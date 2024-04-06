mod camera;
mod command;
mod descriptor;
mod draw;
mod material;
mod mesh;
mod queues;
mod swapchain;

use glam::{Quat, UVec2, UVec3, Vec3};
pub use mesh::Mesh;

use ash::vk;
use core::{ffi::CStr, mem, num::NonZeroU32};
use std::backtrace::Backtrace;
use vk_mem::{Alloc, AllocatorCreateInfo};

const TIMEOUT: u64 = 100_000_000;

pub struct Camera {
    pub translation: glam::Vec3,
    pub rotation: glam::Quat,
    pub projection: CameraProjection,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

pub enum CameraProjection {
    Perspective { fov: f32 },
    Orthographic { scale: f32 },
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
    render_pass: vk::RenderPass,
}

pub struct Render {
    vulkan: Vulkan,
    material_set: MaterialSet,
    mesh_sets: util::Arena<mesh::MeshCollection>,
    draw_function: draw::DrawFunction,
    draw_closures: util::Arena<draw::DrawClosure>,
    textures: util::Arena<(VmaImage, vk::ImageView)>,
    materials: util::Arena<()>,
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
    fn rebuild_swapchain(&mut self) -> vk::Extent2D {
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
            self.render_pass,
            image_count,
        );
        extent
    }
}

fn select_device(instance: &ash::Instance) -> (vk::PhysicalDevice, vk::PhysicalDeviceProperties) {
    // Select device
    let phys_devs = unsafe { instance.enumerate_physical_devices().unwrap() };
    let mut chosen = None;
    for p in phys_devs {
        let properties = unsafe { instance.get_physical_device_properties(p) };
        if properties.device_type == vk::PhysicalDeviceType::CPU {
            continue;
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

unsafe fn make_render_pass(dev: &ash::Device, format: vk::Format) -> vk::RenderPass {
    let attachments = [
        vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            samples: vk::SampleCountFlags::TYPE_1,
        },
        vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: vk::Format::D32_SFLOAT,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            samples: vk::SampleCountFlags::TYPE_1,
        },
    ];
    let color_attachments = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let depth_attachment = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    let subpasses = [vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachments)
        .depth_stencil_attachment(&depth_attachment)
        .build()];
    let subpass_dependencies = [vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_subpass(0)
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )
        .build()];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&subpass_dependencies);
    dev.create_render_pass(&info, None).unwrap()
}

struct MaterialSet {
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    set: vk::DescriptorSet,
    sampler: vk::Sampler,
    materials: VmaBuffer,
}

unsafe fn make_material_set(
    dev: &ash::Device,
    alloc: &vk_mem::Allocator,
    max_textures: u32,
    max_materials: u32,
) -> MaterialSet {
    let layout = {
        let f = |binding| {
            vk::DescriptorSetLayoutBinding::builder()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(max_textures)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build()
        };
        let bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(max_textures)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
        ];
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        dev.create_descriptor_set_layout(&info, None).unwrap()
    };

    let pool = {
        let sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: max_textures,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
            },
        ];
        let info = vk::DescriptorPoolCreateInfo::builder()
            // TODO we don't need UPDATE_AFTER_BIND, do we?
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(1)
            .pool_sizes(&sizes);
        dev.create_descriptor_pool(&info, None).unwrap()
    };

    let set = {
        let layouts = [layout];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        dev.allocate_descriptor_sets(&info).unwrap()[0]
    };

    let sampler = {
        let info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .mip_lod_bias(0.0)
            .anisotropy_enable(false)
            .compare_enable(false)
            .unnormalized_coordinates(false);
        dev.create_sampler(&info, None).unwrap()
    };

    let materials = {
        let info = vk::BufferCreateInfo::builder()
            .size(mem::size_of::<material::PbrMaterial>() as u64 * u64::from(max_materials))
            .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let c_info = vk_mem::AllocationCreateInfo {
            flags: vk_mem::AllocationCreateFlags::STRATEGY_MIN_MEMORY,
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };
        alloc.create_buffer(&info, &c_info).unwrap()
    };

    let info = [vk::DescriptorBufferInfo {
        buffer: materials.0,
        offset: 0,
        range: vk::WHOLE_SIZE,
    }];
    let writes = [vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(1)
        .dst_array_element(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&info)
        .build()];
    dev.update_descriptor_sets(&writes, &[]);

    MaterialSet {
        pool,
        layout,
        set,
        sampler,
        materials,
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

fn init_vulkan(
    window: raw_window_handle::WindowHandle,
    display: raw_window_handle::DisplayHandle,
) -> Vulkan {
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

    layer_names.push(b"VK_LAYER_KHRONOS_validation\0".as_ptr() as *const i8);
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
        use raw_window_handle::HasRawWindowHandle;
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

    let allocator = {
        let info = AllocatorCreateInfo::new(&instance, &dev, physical_device);
        vk_mem::Allocator::new(info).unwrap()
    };

    let image_count = unsafe { swapchain_image_count(physical_device, surface, &surface_loader) };

    let commands = command::Commands::new(&dev, queues);

    let render_pass = unsafe { make_render_pass(&dev, surface_format.format) };

    let (swapchain, extent) = swapchain::SwapChain::new(
        physical_device,
        &dev,
        &allocator,
        &surface_loader,
        surface,
        surface_format,
        commands.queues.graphics_index,
        &instance,
        render_pass,
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
        render_pass,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeshSetHandle(util::ArenaHandle);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureHandle(util::ArenaHandle);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PbrMaterialHandle(util::ArenaHandle);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShaderSetHandle(NonZeroU32);

impl ShaderSetHandle {
    pub const PBR: Self = Self(NonZeroU32::MAX);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DrawClosureHandle(util::ArenaHandle);

pub mod function {
    pub struct New<'a> {
        pub buffers: &'a [Buffer<'a>],
        pub inputs: &'a [Input],
        pub outputs: &'a [Output],
        pub stages: &'a [Stage<'a>],
        pub flow: &'a [(usize, usize)],
    }

    pub enum Input {
        Buffer(usize),
    }

    pub enum Output {
        Buffer(usize),
        Display(usize),
    }

    pub enum Buffer<'a> {
        Fixed { ty: &'a [Type] },
        Array { ty: &'a [Type] },
        DrawParameters,
        MeshCollection,
    }

    pub enum Type {
        F32,
        F32_2,
        F32_3,
        F32_4,
        F32_4_4,
        U32,
    }

    pub enum Stage<'a> {
        Compute {
            shader: &'a [u32],
            inputs: &'a [usize],
            outputs: &'a [usize],
            uniforms: &'a [usize],
        },
        Graphics {
            vertex_shader: &'a [u32],
            fragment_shader: &'a [u32],
            vertex_inputs: &'a [usize],
            instance_inputs: &'a [usize],
            uniforms: &'a [usize],
        },
    }
}

/// Parameters for the default PBR function.
pub const PBR: function::New<'static> = {
    use function::*;

    New {
        buffers: &[
            Buffer::DrawParameters,
            Buffer::MeshCollection,
            Buffer::Array {
                ty: &[Type::F32_3, Type::F32_3, Type::F32_3, Type::U32],
            },
            Buffer::Array {
                ty: &[Type::F32_4_4],
            },
        ],
        inputs: &[],
        outputs: &[],
        stages: &[
            Stage::Compute {
                shader: vk_shader_macros::include_glsl!("instance.glsl", kind: comp),
                inputs: &[],
                outputs: &[],
                uniforms: &[],
            },
            Stage::Graphics {
                vertex_shader: vk_shader_macros::include_glsl!("shader/pbr.vert.glsl", kind: vert),
                fragment_shader: vk_shader_macros::include_glsl!("shader/pbr.frag.glsl", kind: frag),
                vertex_inputs: &[4],
                instance_inputs: &[],
                uniforms: &[],
            },
        ],
        flow: &[(0, 1)],
    }
};

#[derive(Clone, Copy, Debug)]
pub enum TextureFormat {
    Rgba8Unorm,
    Gray8Unorm,
}

#[derive(Clone, Copy, Debug)]
pub struct Rgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Rgb {
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    pub const fn to_array(&self) -> [f32; 3] {
        [self.r, self.g, self.b]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PbrMaterial {
    pub albedo: Rgb,
    pub roughness: f32,
    pub metallic: f32,
    pub ambient_occlusion: f32,
    pub albedo_texture: TextureHandle,
    pub roughness_texture: TextureHandle,
    pub metallic_texture: TextureHandle,
    pub ambient_occlusion_texture: TextureHandle,
}

#[derive(Clone, Copy, Debug)]
pub struct InstanceData {
    pub translation: Vec3,
    pub rotation: Quat,
    pub material: PbrMaterialHandle,
}

impl Render {
    pub fn new(
        window: raw_window_handle::WindowHandle,
        display: raw_window_handle::DisplayHandle,
    ) -> Self {
        let vulkan = init_vulkan(window, display);
        let material_set = unsafe { make_material_set(&vulkan.dev, &vulkan.allocator, 1024, 1024) };
        let draw_function = unsafe {
            draw::DrawFunction::new(
                &vulkan.dev,
                vulkan.surface_format.format,
                vulkan.render_pass,
                material_set.layout,
            )
        };
        Self {
            vulkan,
            mesh_sets: Default::default(),
            draw_function,
            draw_closures: Default::default(),
            textures: Default::default(),
            material_set,
            materials: Default::default(),
        }
    }

    pub fn add_meshes(&mut self, meshes: &[Mesh], max_instances: u32) -> MeshSetHandle {
        let mesh_set = mesh::MeshCollectionBuilder { meshes }.finish(&mut self.vulkan);
        MeshSetHandle(self.mesh_sets.insert(mesh_set))
    }

    pub fn add_texture_2d(
        &mut self,
        dimensions: UVec2,
        format: TextureFormat,
        reader: &mut dyn FnMut(&mut [u8]),
    ) -> TextureHandle {
        let fmt = match format {
            TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            TextureFormat::Gray8Unorm => vk::Format::R8_UNORM,
        };

        let img = unsafe {
            let info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(fmt)
                .extent(vk::Extent3D {
                    width: dimensions.x,
                    height: dimensions.y,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let c_info = vk_mem::AllocationCreateInfo {
                flags: vk_mem::AllocationCreateFlags::STRATEGY_MIN_MEMORY,
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };
            self.vulkan.allocator.create_image(&info, &c_info).unwrap()
        };

        unsafe {
            self.vulkan.commands.transfer_to_image_with(
                &self.vulkan.dev,
                &self.vulkan.allocator,
                img.0,
                UVec3::ZERO,
                reader,
                UVec3::new(dimensions.x, dimensions.y, 1),
                fmt,
            );
        }

        let view = unsafe {
            let info = vk::ImageViewCreateInfo::builder()
                .image(img.0)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(fmt)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            self.vulkan.dev.create_image_view(&info, None).unwrap()
        };

        let h = self.textures.insert((img, view));

        unsafe {
            let info = [vk::DescriptorImageInfo {
                sampler: self.material_set.sampler,
                image_view: view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }];
            let writes = [vk::WriteDescriptorSet::builder()
                .dst_set(self.material_set.set)
                .dst_binding(match format {
                    TextureFormat::Rgba8Unorm => 0,
                    TextureFormat::Gray8Unorm => todo!(),
                })
                .dst_array_element(h.as_u32())
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&info)
                .build()];
            self.vulkan.dev.update_descriptor_sets(&writes, &[]);
        }

        TextureHandle(h)
    }

    pub fn add_pbr_material(&mut self, material: &PbrMaterial) -> PbrMaterialHandle {
        let h = self.materials.insert(());

        let mat = material::PbrMaterial {
            albedo: material.albedo.to_array(),
            albedo_texture_index: material.albedo_texture.0.as_u32(),
            roughness: material.roughness,
            roughness_texture_id: material.roughness_texture.0.as_u32(),
            metallic: material.metallic,
            metallic_texture_id: material.metallic_texture.0.as_u32(),
            ambient_occlusion: material.ambient_occlusion,
            ambient_occlusion_texture_id: material.ambient_occlusion_texture.0.as_u32(),
        };

        unsafe {
            self.vulkan.commands.transfer_to(
                &self.vulkan.dev,
                &self.vulkan.allocator,
                self.material_set.materials.0,
                u64::try_from(mem::size_of_val(&mat)).unwrap() * u64::from(h.as_u32()),
                (&mat as *const material::PbrMaterial).cast(),
                mem::size_of_val(&mat),
            );
        }

        PbrMaterialHandle(h)
    }

    pub fn make_draw_closure(
        &mut self,
        mesh_set: MeshSetHandle,
        shader_set: ShaderSetHandle,
        max_instances: u32,
    ) -> DrawClosureHandle {
        assert_eq!(shader_set, ShaderSetHandle::PBR, "TODO: custom shaders");
        let meshes = &self.mesh_sets[mesh_set.0];
        let closure = unsafe {
            draw::DrawClosure::new(
                &self.vulkan.dev,
                &self.vulkan.allocator,
                self.vulkan.commands.pool,
                &self.draw_function,
                self.vulkan.swapchain.framebuffers.len() as u32,
                meshes,
                mesh_set,
                max_instances,
                &self.vulkan.camera,
            )
        };
        DrawClosureHandle(self.draw_closures.insert(closure))
    }

    pub fn draw(
        &mut self,
        camera: &Camera,
        draw_closure: DrawClosureHandle,
        instances_counts: &[u32],
        instances_data: &mut dyn Iterator<Item = InstanceData>,
    ) {
        let closure = &mut self.draw_closures[draw_closure.0];

        unsafe {
            self.vulkan
                .swapchain
                .draw(&self.vulkan.dev, &mut self.vulkan.swapchain_draw, |info| {
                    let meshes = &self.mesh_sets[closure.mesh_handle.0];
                    closure.set_instance_data(info.index, meshes, instances_counts, instances_data);
                    self.vulkan.camera.set(info.index, camera);
                    closure.submit(&self.vulkan.dev, &mut self.vulkan.commands, &info);
                    self.vulkan.commands.queues.graphics
                });
        }
    }

    pub fn rebuild_swapchain(&mut self) {
        let extent = self.vulkan.rebuild_swapchain();
        for c in self.draw_closures.iter_mut() {
            let meshes = &self.mesh_sets[c.mesh_handle.0];
            unsafe {
                c.record_command(
                    &self.vulkan.dev,
                    &self.draw_function,
                    meshes,
                    self.material_set.set,
                    extent,
                    &self.vulkan.swapchain.framebuffers,
                    self.vulkan.render_pass,
                );
            }
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
