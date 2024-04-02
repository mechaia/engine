mod camera;
mod command;
mod descriptor;
mod draw;
mod material;
mod mesh;
mod queues;
mod render_pass;
mod swapchain;

pub use mesh::Mesh;

use ash::vk;
use core::{ffi::CStr, mem, num::NonZeroU32, ptr::NonNull};
use raw_window_handle::HasRawDisplayHandle;
use std::{backtrace::Backtrace, borrow::Cow};
use vk_mem::{Alloc, AllocatorCreateInfo};

const TIMEOUT: u64 = 100_000_000;

pub struct Camera {
    pub translation: glam::Vec3,
    pub rotation: glam::Quat,
    pub fov: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

type VmaBuffer = (vk::Buffer, vk_mem::Allocation);
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

pub struct Vulkan {
    entry: ash::Entry,
    instance: ash::Instance,
    debug_utils: ash::extensions::ext::DebugUtils,
    utils_messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    allocator: vk_mem::Allocator,
    dev: ash::Device,
    surface_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    surface_format: vk::SurfaceFormatKHR,
    swapchain: swapchain::SwapChain,
    swapchain_draw: swapchain::Draw,
    camera: camera::Camera,
    commands: command::Commands,
    draw_function: draw::DrawFunction,
    draw_closures: Vec<Option<draw::DrawClosure>>,
}

/*
 * Who gives a shit? OS will clean up anyway.
impl Drop for Vulkan {
    fn drop(&mut self) {
        unsafe {
            for iv in self.swapchain_imageviews.drain(..) {
                self.dev.destroy_image_view(iv, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.dev.destroy_device(None);
            self.debug_utils
                .destroy_debug_utils_messenger(self.utils_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}
*/
// fucking VMA
impl Drop for Vulkan {
    fn drop(&mut self) {
        unsafe {
            let _ = self.dev.device_wait_idle();
            self.swapchain.drop_with(&self.dev, &self.allocator);
            self.swapchain_draw.drop_with(&self.dev);
            self.camera.drop_with(&self.allocator);
            self.commands.drop_with(&self.allocator);
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

impl Vulkan {
    pub fn rebuild_swapchain(&mut self) {
        unsafe {
            self.dev.device_wait_idle().unwrap();
            self.swapchain.drop_with(&self.dev, &self.allocator);
        }
        let image_count = unsafe {
            swapchain_image_count(self.physical_device, self.surface, &self.surface_loader)
        };
        let extent;
        (self.swapchain, extent) = swapchain::SwapChain::new(
            self.physical_device,
            &self.dev,
            &self.allocator,
            &self.surface_loader,
            self.surface,
            self.surface_format,
            self.commands.queues.graphics_index,
            &self.instance,
            self.draw_function.render_pass(),
            image_count,
        );
        for c in self.draw_closures.iter_mut().flat_map(|c| c) {
            unsafe {
                c.record_command(
                    &self.dev,
                    &self.draw_function,
                    extent,
                    &self.swapchain.framebuffers,
                );
            }
        }
    }
}

fn select_device(instance: &ash::Instance) -> (vk::PhysicalDevice, vk::PhysicalDeviceProperties) {
    // Select device
    let phys_devs = unsafe { instance.enumerate_physical_devices().unwrap() };
    let mut chosen = None;
    for p in phys_devs {
        let properties = unsafe { instance.get_physical_device_properties(p) };
        if properties.device_type == vk::PhysicalDeviceType::CPU {
            //continue;
            chosen = Some((p, properties));
            break;
        } else if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            chosen = Some((p, properties));
        } else if chosen.is_none() {
            // Select any, even if something like llvmpipe
            chosen = Some((p, properties));
        }
    }
    chosen.unwrap()
}

unsafe fn make_descriptor_pool(
    dev: &ash::Device,
    uniform_count: u32,
    storage_count: u32,
) -> vk::DescriptorPool {
    let pool_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: uniform_count,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: storage_count,
        },
    ];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&pool_sizes)
        .max_sets(pool_sizes.iter().map(|p| p.descriptor_count).sum());
    dev.create_descriptor_pool(&info, None).unwrap()
}

unsafe fn allocate_descriptor_sets(
    dev: &ash::Device,
    pool: vk::DescriptorPool,
    layouts: &[vk::DescriptorSetLayout],
) -> Vec<vk::DescriptorSet> {
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool)
        .set_layouts(layouts);
    dev.allocate_descriptor_sets(&info).unwrap()
}

pub fn init_vulkan(window: &winit::window::Window) -> Vulkan {
    let entry = unsafe { ash::Entry::load().unwrap() };

    // Basic setup
    let layer_names = [b"VK_LAYER_KHRONOS_validation\0".as_ptr() as *const i8];
    let mut extension_names = vec![ash::extensions::ext::DebugUtils::name().as_ptr()];
    extension_names.extend_from_slice(
        ash_window::enumerate_required_extensions(window.raw_display_handle()).unwrap(),
    );

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
        use raw_window_handle::HasRawWindowHandle;
        ash_window::create_surface(
            &entry,
            &instance,
            window.raw_display_handle(),
            window.raw_window_handle(),
            None,
        )
        .unwrap()
    };
    let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);

    let surface_format = select_format(&surface_loader, physical_device, surface);

    let (dev, queues) = queues::Queues::new(&instance, physical_device);

    let allocator = {
        let info = AllocatorCreateInfo::new(&instance, &dev, physical_device);
        vk_mem::Allocator::new(info).unwrap()
    };

    let image_count = unsafe { swapchain_image_count(physical_device, surface, &surface_loader) };

    let commands = command::Commands::new(&dev, queues);

    let draw_function = unsafe { draw::DrawFunction::new(&dev, surface_format.format) };

    let draw_closures = Vec::new();

    let (swapchain, extent) = swapchain::SwapChain::new(
        physical_device,
        &dev,
        &allocator,
        &surface_loader,
        surface,
        surface_format,
        commands.queues.graphics_index,
        &instance,
        draw_function.render_pass(),
        image_count,
    );

    let camera = unsafe { camera::Camera::new(&allocator, image_count) };

    let swapchain_draw = swapchain::Draw::new(&dev, image_count);

    // Done
    Vulkan {
        entry,
        allocator,
        instance,
        debug_utils,
        utils_messenger,
        physical_device,
        dev,
        surface,
        surface_loader,
        surface_format,
        swapchain,
        swapchain_draw,
        commands,
        camera,
        draw_function,
        draw_closures,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeshSetHandle(NonZeroU32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShaderSetHandle(NonZeroU32);

impl ShaderSetHandle {
    pub const PBR: Self = Self(NonZeroU32::MAX);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DrawSetHandle(NonZeroU32);

pub struct MemoryWrite {
    alloc: vk_mem::Allocation,
}

impl Drop for MemoryWrite {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        eprintln!("== Leaking memory! ==");
        eprintln!("{}", Backtrace::force_capture());
    }
}

pub struct MemoryWriter<'a> {
    allocator: &'a vk_mem::Allocator,
    alloc: &'a mut vk_mem::Allocation,
    cur: NonNull<u8>,
    end: NonNull<u8>,
}

impl<'a> MemoryWriter<'a> {
    fn new(mem: &'a mut MemoryWrite, allocator: &'a vk_mem::Allocator) -> Self {
        let info = allocator.get_allocation_info(&mem.alloc);
        unsafe {
            let ptr = allocator.map_memory(&mut mem.alloc).unwrap();
            Self {
                allocator,
                alloc: &mut mem.alloc,
                cur: NonNull::new_unchecked(ptr),
                end: NonNull::new_unchecked(ptr.add(info.size as usize)),
            }
        }
    }

    pub fn write(&mut self, data: &[u8]) {
        unsafe {
            // use debug_assert to still allow coalescing writes by the compiler
            // in release builds.
            let max = self.end.as_ptr().offset_from(self.cur.as_ptr()) as usize;
            debug_assert!(max <= data.len(), "writing out of bounds");
            self.write_unchecked(&data[..data.len().min(max)]);
        }
    }

    // Only make public if someone
    // - can demonstrate it causes performance issues
    // - has a real, valid usecase. Not memeshit like writing byte-by-byte
    unsafe fn write_unchecked(&mut self, data: &[u8]) {
        self.cur
            .as_ptr()
            .copy_from_nonoverlapping(data.as_ptr(), data.len());
    }
}

impl Drop for MemoryWriter<'_> {
    fn drop(&mut self) {
        unsafe {
            self.allocator.unmap_memory(self.alloc);
            self.allocator
                .flush_allocation(&self.alloc, 0, ash::vk::WHOLE_SIZE as usize)
                .unwrap();
        }
    }
}

pub struct MemoryRead {
    alloc: vk_mem::Allocation,
}

pub struct MakeFunction<'a> {
    inputs: (),
    outputs: (),
    stages: &'a [FunctionStage<'a>],
    edges: &'a [(usize, usize)],
}

pub enum FunctionStage<'a> {
    Compute {
        shader: &'a [u32],
    },
    Graphics {
        vertex_shader: &'a [u32],
        fragment_shader: &'a [u32],
    },
}

/// Parameters for the default PBR function.
pub const PBR: MakeFunction<'static> = MakeFunction {
};

impl Vulkan {
    /// Allocate memory to transfer data to the GPU.
    ///
    /// # Warning
    ///
    /// Must be manually freed with [`Self::free_memory`].
    ///
    /// # Details
    ///
    /// See https://www.khronos.org/assets/uploads/developers/library/2018-vulkan-devday/03-Memory.pdf
    /// slide 13
    #[allow(unused)]
    pub fn allocate_memory_read(&mut self, size: usize, alignment: usize) -> MemoryRead {
        todo!()
    }

    /// Allocate memory that is accessible by the GPU as a buffer.
    ///
    /// Intended for:
    /// - Data to transfer to GPU memory.
    /// - Data that is frequently updated by the CPU (e.g. object with rigidbody)
    ///
    /// If `host_cached` is `false`, then the region is optimized for CPU to GPU writes.
    /// If `host_cached` is `true`, then the region is optimized for GPU to CPU writes.
    ///
    /// AVOID reading from CPU if `host_cached` is `false`.
    /// AVOID reading from GPU if `host_cached` is `true`.
    ///
    /// AVOID partial writes. Prefer writing contiguous blocks.
    ///
    /// # Warning
    ///
    /// Must be manually freed with [`Self::free_memory`].
    ///
    /// # Details
    ///
    /// See https://www.khronos.org/assets/uploads/developers/library/2018-vulkan-devday/03-Memory.pdf
    /// slide 13
    pub fn allocate_memory_write(
        &mut self,
        size: usize,
        alignment: u32,
        host_cached: bool,
    ) -> MemoryWrite {
        let reqs = vk::MemoryRequirements {
            size: size.try_into().unwrap(),
            alignment: alignment.into(),
            memory_type_bits: 0,
        };
        let info = vk_mem::AllocationCreateInfo {
            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE,
            preferred_flags: if host_cached {
                vk::MemoryPropertyFlags::HOST_CACHED
            } else {
                vk::MemoryPropertyFlags::empty()
            },
            ..Default::default()
        };
        let alloc = unsafe { self.allocator.allocate_memory(&reqs, &info).unwrap() };
        MemoryWrite { alloc }
    }

    pub fn memory_start_write<'a>(&'a mut self, memory: &'a mut MemoryWrite) -> MemoryWriter<'a> {
        MemoryWriter::new(memory, &self.allocator)
    }

    pub fn add_meshes(&mut self, meshes: &[Mesh], max_instances: u32) -> MeshSetHandle {
        let meshes = mesh::MeshCollectionBuilder { meshes }.finish(self);
        let closure = unsafe {
            draw::DrawClosure::new(
                &self.dev,
                &self.allocator,
                self.commands.pool,
                &self.draw_function,
                self.swapchain.framebuffers.len() as u32,
                meshes,
                max_instances,
                &self.camera,
                self.commands.queues.graphics,
            )
        };
        self.draw_closures.push(Some(closure));
        return MeshSetHandle(NonZeroU32::new(self.draw_closures.len() as u32).unwrap());
    }

    pub fn make_draw_set(
        &mut self,
        mesh_set: MeshSetHandle,
        shader_set: ShaderSetHandle,
    ) -> DrawSetHandle {
        assert_eq!(shader_set, ShaderSetHandle::PBR, "TODO: custom shaders");
        DrawSetHandle(mesh_set.0)
    }

    pub fn draw(&mut self, camera: &Camera, draw_set: DrawSetHandle) {
        unsafe {
            self.swapchain
                .draw(&self.dev, &mut self.swapchain_draw, |info| {
                    self.camera.set(info.index, camera);
                    self.draw_closures[0].as_ref().unwrap().submit(
                        &self.dev,
                        &mut self.commands,
                        &info,
                    );
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
