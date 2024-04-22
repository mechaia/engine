pub mod resource;
pub mod stage;

mod command;
mod descriptor;
mod material;
mod queues;
mod swapchain;

pub use stage::{Stage, StageArgs};
pub use swapchain::SwapChain;

use ash::vk;
use core::{ffi::CStr, mem, num::NonZeroU32, ops::Deref};
use std::{backtrace::Backtrace, ptr::NonNull};
use vk_mem::{Alloc, AllocatorCreateInfo};

const TIMEOUT: u64 = 100_000_000;

pub type VmaBuffer = (vk::Buffer, vk_mem::Allocation);
type VmaImage = (vk::Image, vk_mem::Allocation);

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message)
        .to_str()
        .unwrap();
    let ty = format!("{:?}", message_type).to_lowercase();

    let (severity, color) = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => ("debug", "90"),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => ("info", "97"),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => ("warn", "93"),
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => ("error", "91"),
        _ => ("????", "90"),
    };

    eprintln!("\x1b[{color}m[{ty}:{severity}] {message}\x1b[39m");
    if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        eprintln!("{:}", Backtrace::force_capture());
        std::process::exit(2);
    }
    vk::FALSE
}

fn env_true(var: &str) -> bool {
    std::env::var(var).is_ok_and(|v| !["", "0"].contains(&&*v))
}

pub struct Dev {
    dev: ash::Device,
    alloc: vk_mem::Allocator,
}

impl Dev {
    pub fn allocate_buffer(
        &mut self,
        size: u64,
        usage: vk::BufferUsageFlags,
        host: bool,
    ) -> VmaBuffer {
        //let queue_family_indices = [self.commands.queues.graphics_index];
        let queue_family_indices = [];
        let b_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_family_indices);
        let mut flags = vk_mem::AllocationCreateFlags::STRATEGY_MIN_MEMORY;
        if host {
            flags |= vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE;
        }
        let c_info = vk_mem::AllocationCreateInfo {
            flags,
            usage: if host {
                vk_mem::MemoryUsage::AutoPreferHost
            } else {
                vk_mem::MemoryUsage::AutoPreferDevice
            },
            required_flags: if host {
                vk::MemoryPropertyFlags::HOST_COHERENT
            } else {
                vk::MemoryPropertyFlags::empty()
            },
            ..Default::default()
        };
        unsafe { self.alloc.create_buffer(&b_info, &c_info).unwrap() }
    }

    pub fn map_buffer(&mut self, buffer: &mut VmaBuffer) -> NonNull<u8> {
        unsafe { NonNull::new(self.alloc.map_memory(&mut buffer.1).unwrap()).unwrap() }
    }

    pub unsafe fn unmap_buffer(&mut self, buffer: &mut VmaBuffer) {
        self.alloc.unmap_memory(&mut buffer.1)
    }

    pub unsafe fn free_buffer(&mut self, mut buffer: VmaBuffer) {
        self.alloc.destroy_buffer(buffer.0, &mut buffer.1);
    }
}

impl Deref for Dev {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.dev
    }
}

pub unsafe trait DropWith {
    unsafe fn drop_with(self, dev: &Dev);
}

pub struct Render {
    dev: Dev,
    entry: ash::Entry,
    instance: ash::Instance,
    debug_utils: ash::extensions::ext::DebugUtils,
    utils_messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    surface_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    surface_format: vk::SurfaceFormatKHR,
    swapchain: swapchain::SwapChain,
    swapchain_draw: swapchain::Draw,
    commands: command::Commands,
    command_buffers: Box<[vk::CommandBuffer]>,
    stages: util::Arena<Box<[Box<dyn Stage>]>>,
}

impl Render {
    fn make_descriptor_pool(
        &mut self,
        sets: u32,
        sizes: &[vk::DescriptorPoolSize],
    ) -> vk::DescriptorPool {
        let info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(sets)
            .pool_sizes(sizes);
        unsafe { self.dev.create_descriptor_pool(&info, None).unwrap() }
    }
}

impl Drop for Render {
    fn drop(&mut self) {
        unsafe {
            let _ = self.dev.dev.device_wait_idle();
            self.swapchain.drop_with(&self.dev.dev);
            self.swapchain_draw.drop_with(&self.dev.dev);
            self.commands.drop_with(&self.dev.alloc);
        }
    }
}

fn select_format(
    surface_loader: &ash::extensions::khr::Surface,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> vk::SurfaceFormatKHR {
    let mut formats = unsafe {
        surface_loader
            .get_physical_device_surface_formats(physical_device, surface)
            .unwrap()
    };
    let f_prio = |f: vk::SurfaceFormatKHR| match f.format {
        vk::Format::B8G8R8A8_SRGB
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::B8G8R8_SRGB
        | vk::Format::R8G8B8_SRGB => 100,
        _ => 0,
    };
    let mut fmt = formats.pop().expect("no surface formats?");
    let mut prio = f_prio(fmt);
    for f in formats {
        let p = f_prio(fmt);
        if p > prio {
            (fmt, prio) = (f, p);
        }
    }
    fmt
}

unsafe fn swapchain_image_count(
    physdev: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_loader: &ash::extensions::khr::Surface,
) -> usize {
    let surface_capabilities = surface_loader
        .get_physical_device_surface_capabilities(physdev, surface)
        .unwrap();
    // TODO I don't understand this tbh
    3.min(surface_capabilities.max_image_count)
        .max(surface_capabilities.min_image_count) as usize
}

impl Render {
    pub fn rebuild_swapchain(&mut self) {
        unsafe {
            self.dev.dev.device_wait_idle().unwrap();
            self.swapchain.drop_with(&self.dev.dev);
        }
        let image_count = unsafe {
            swapchain_image_count(self.physical_device, self.surface, &self.surface_loader)
        };
        self.swapchain = swapchain::SwapChain::new(
            self.physical_device,
            &self.dev.dev,
            &self.surface_loader,
            self.surface,
            self.surface_format,
            self.commands.queues.graphics_index,
            &self.instance,
            image_count as u32,
        );

        for stages in self.stages.values_mut() {
            for stage in stages.iter_mut() {
                unsafe {
                    stage.rebuild_swapchain(&mut self.dev, &mut self.swapchain);
                }
            }
        }

        unsafe {
            self.record_commands();
        }
    }
}

fn select_device(instance: &ash::Instance) -> (vk::PhysicalDevice, vk::PhysicalDeviceProperties) {
    // Select device
    let phys_devs = unsafe { instance.enumerate_physical_devices().unwrap() };
    let mut chosen = None;
    let force_cpu = env_true("MECHAIA_RENDER_FORCE_CPU");
    let mut prio = 0;
    for p in phys_devs {
        let properties = unsafe { instance.get_physical_device_properties(p) };
        match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU if prio < 5 => {
                chosen = Some((p, properties));
                prio = 5;
            }
            vk::PhysicalDeviceType::INTEGRATED_GPU if prio < 4 => {
                chosen = Some((p, properties));
                prio = 4;
            }
            vk::PhysicalDeviceType::VIRTUAL_GPU if prio < 3 => {
                chosen = Some((p, properties));
                prio = 3;
            }
            vk::PhysicalDeviceType::CPU if prio < 2 || force_cpu => {
                chosen = Some((p, properties));
                prio = 2 + u8::from(force_cpu) * 100;
            }
            vk::PhysicalDeviceType::OTHER if prio < 1 => {
                chosen = Some((p, properties));
                prio = 1;
            }
            // skip
            _ => {}
        }
    }
    chosen.unwrap()
}

impl Render {
    unsafe fn record_commands(&self) {
        assert_eq!(self.stages.len(), 1, "COME THE FUCK ON DUDE");

        for stage in self.stages.values() {
            for (index, &cmd) in self.command_buffers.iter().enumerate() {
                let info = vk::CommandBufferBeginInfo::builder();
                self.dev.begin_command_buffer(cmd, &info).unwrap();
                let args = StageArgs {
                    cmd,
                    index,
                    viewport: self.swapchain.extent(),
                };
                for stage in stage.iter() {
                    stage.record_commands(&self.dev, &args);
                }
                self.dev.end_command_buffer(cmd).unwrap();
            }
        }
    }
}

macro_rules! cvt_raw_handle {
    ($fn:ident $arg:ident $ret:ident $($var:ident $varh:ident [$($val:ident)*])*) => {
fn $fn(display: raw_window_handle::$arg) -> old_raw_window_handle::$ret {
    match display.as_raw() { $(
        raw_window_handle::$ret::$var(x) => {
            let mut y = old_raw_window_handle::$varh::empty();
            $(y.$val = x.$val;)*
            old_raw_window_handle::$ret::$var(y)
        }
        c => todo!("{:?}", c),
    )* }
}
    };
}

fn convert_display_handle(
    display: raw_window_handle::DisplayHandle,
) -> old_raw_window_handle::RawDisplayHandle {
    use old_raw_window_handle as b;
    use old_raw_window_handle::RawDisplayHandle as B;
    use raw_window_handle::RawDisplayHandle as A;
    match display.as_raw() {
        A::Xlib(x) => {
            let mut y = b::XlibDisplayHandle::empty();
            y.display = x.display.map_or(core::ptr::null_mut(), |p| p.as_ptr());
            y.screen = x.screen;
            B::Xlib(y)
        }
        c => todo!("{:?}", c),
    }
}

cvt_raw_handle!(
    convert_window_handle WindowHandle RawWindowHandle
    Xlib XlibWindowHandle [window visual_id]
);

fn init(
    window: raw_window_handle::WindowHandle,
    display: raw_window_handle::DisplayHandle,
) -> Render {
    // FIXME fucking ash-window, y u close?
    // https://github.com/ash-rs/ash/pull/826
    let raw_window_handle = convert_window_handle(window);
    let raw_display_handle = convert_display_handle(display);

    let entry = unsafe { ash::Entry::load().unwrap() };

    // Basic setup
    let mut layer_names = vec![];
    let mut extension_names = vec![];

    extension_names
        .extend_from_slice(ash_window::enumerate_required_extensions(raw_display_handle).unwrap());

    if env_true("MECHAIA_RENDER_VALIDATION") {
        layer_names.push(b"VK_LAYER_KHRONOS_validation\0".as_ptr() as *const i8);
    }

    extension_names.push(ash::extensions::ext::DebugUtils::name().as_ptr());

    let app_info = vk::ApplicationInfo::builder()
        .application_name(CStr::from_bytes_with_nul(b"Block Renderer\0").unwrap())
        .application_version(vk::make_api_version(0, 0, 0, 1))
        .engine_name(CStr::from_bytes_with_nul(b"BlockRender\0").unwrap())
        .engine_version(vk::make_api_version(0, 0, 42, 0))
        .api_version(vk::make_api_version(0, 1, 3, 0));
    let mut debugcreateinfo = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(vulkan_debug_utils_callback));
    let instance_create_info = vk::InstanceCreateInfo::builder()
        .push_next(&mut debugcreateinfo)
        .application_info(&app_info)
        .enabled_layer_names(&layer_names)
        .enabled_extension_names(&extension_names);

    let instance = unsafe { entry.create_instance(&instance_create_info, None).unwrap() };
    let debug_utils = ash::extensions::ext::DebugUtils::new(&entry, &instance);
    let utils_messenger = unsafe {
        debug_utils
            .create_debug_utils_messenger(&debugcreateinfo, None)
            .unwrap()
    };

    let (physical_device, _) = select_device(&instance);

    // Surface
    let surface = unsafe {
        ash_window::create_surface(
            &entry,
            &instance,
            raw_display_handle,
            raw_window_handle,
            None,
        )
        .unwrap()
    };
    let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);

    let surface_format = select_format(&surface_loader, physical_device, surface);

    let (dev, queues) = queues::Queues::new(&instance, physical_device);

    let alloc = {
        let info = AllocatorCreateInfo::new(&instance, &dev, physical_device);
        vk_mem::Allocator::new(info).unwrap()
    };

    let image_count = unsafe { swapchain_image_count(physical_device, surface, &surface_loader) };

    let commands = command::Commands::new(&dev, queues);

    let swapchain = swapchain::SwapChain::new(
        physical_device,
        &dev,
        &surface_loader,
        surface,
        surface_format,
        commands.queues.graphics_index,
        &instance,
        image_count as u32,
    );

    let swapchain_draw = swapchain::Draw::new(&dev, image_count);

    let command_buffers = unsafe {
        let info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(commands.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(image_count as u32);
        dev.allocate_command_buffers(&info).unwrap().into()
    };

    // Done
    Render {
        entry,
        dev: Dev { dev, alloc },
        instance,
        debug_utils,
        utils_messenger,
        physical_device,
        surface,
        surface_loader,
        surface_format,
        swapchain,
        swapchain_draw,
        commands,
        command_buffers,
        stages: Default::default(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShaderSetHandle(NonZeroU32);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StageSetHandle(util::ArenaHandle);

impl ShaderSetHandle {
    pub const PBR: Self = Self(NonZeroU32::MAX);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DrawClosureHandle(util::ArenaHandle);

impl Render {
    pub fn new(
        window: raw_window_handle::WindowHandle,
        display: raw_window_handle::DisplayHandle,
    ) -> Self {
        init(window, display)
    }

    pub fn add_stage_set(&mut self, stages: Box<[Box<dyn Stage>]>) -> StageSetHandle {
        let h = self.stages.insert(stages);

        // FIXME add trait method or something to init without rebuilding shit
        self.rebuild_swapchain();

        unsafe { self.record_commands() };
        StageSetHandle(h)
    }

    pub fn draw(&mut self, stage_set: StageSetHandle, update: &mut dyn FnMut(usize)) {
        unsafe {
            self.swapchain
                .draw(&self.dev, &mut self.swapchain_draw, |info| {
                    update(info.index);
                    let cmdbufs = [self.command_buffers[info.index]];
                    let avail = [info.available];
                    let finish = [info.finished];
                    let mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                    let submit_info = vk::SubmitInfo::builder()
                        .command_buffers(&cmdbufs)
                        .wait_semaphores(&avail)
                        .signal_semaphores(&finish)
                        .wait_dst_stage_mask(&mask);
                    self.dev
                        .queue_submit(
                            self.commands.queues.graphics,
                            &[submit_info.build()],
                            info.may_draw,
                        )
                        .unwrap();
                    self.commands.queues.graphics
                });
        }
    }
}

fn f32_to_bytes(slice: &[f32]) -> &[u8] {
    unsafe {
        core::slice::from_raw_parts(
            slice.as_ptr().cast::<u8>(),
            slice.len() * mem::size_of::<f32>(),
        )
    }
}

pub fn box_stage<T: Stage + 'static>(stage: T) -> Box<dyn Stage> {
    Box::new(stage)
}
