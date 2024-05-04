/**
 * "Standard" 3D render stage.
 *
 * Designed to handle common models in a relatively efficient manner.
 *
 * Features:
 * - Dynamic instancing
 * - Metallic-roughness PBR shading
 * - Skinning with up to 4 joints per vertex
 *
 * How to use:
 * - Load transforms into buffer
 * - Load instance data into buffer
 * - ???
 * - profit
 */
mod compute;
mod descriptor;
mod graphics;

pub use compute::ComputeStage;

use super::renderpass::RenderPassBuilder;
use crate::resource::camera::{Camera, CameraView};
use crate::resource::texture::TextureView;
use crate::resource::Shared;
use crate::resource::{material::pbr::PbrMaterialSet, mesh::MeshSet};
use crate::{DropWith, Render, VmaBuffer};
use ash::vk;
use core::{ffi::CStr, mem, ptr::NonNull};
use descriptor::Descriptors;
use glam::{Vec3, Vec4};
use std::sync::{Arc, Mutex};
use util::TransformScale;

const ENTRY_POINT: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

/// vec3 (position) + uint (material) + vec4 (rotation)
const COMPUTE_INSTANCE_DATA_SIZE: u32 = (3 + 1 + 4) * 4;
/// mat4 (projection)
const GRAPHICS_INSTANCE_DATA_SIZE: u32 = 4 * 4 * 4;

pub struct Standard3D {
    shared: Data,
}

pub struct Configuration {
    pub max_transform_count: u32,
    pub max_instance_count: u32,
    pub max_texture_count: u32,
    pub max_material_count: u32,
    pub transparent: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureHandle(util::ArenaHandle);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeshSetHandle(util::ArenaHandle);

type Data = Arc<Mutex<SharedData>>;

struct SharedData {
    max_transform_count: u32,
    max_instance_count: u32,
    camera: Shared<Camera>,
    set_data: util::Arena<SetData>,
    per_image: Box<[DataOne]>,
    descriptors: Descriptors,
    textures: util::Arena<Shared<TextureView>>,
    max_texture_count: u32,
    material_set: Shared<PbrMaterialSet>,
}

struct SetData {
    graphics: graphics::SetData,
    mesh_set: Shared<MeshSet>,
}

struct DataOne {
    compute: compute::Data,
    graphics: graphics::Data,
    directional_lights_data: DirectionalLight,
    directional_lights: VmaBuffer,
    directional_lights_ptr: NonNull<DirectionalLight>,
}

#[derive(Clone, Copy, Debug)]
pub struct Instance {
    pub transforms_offset: u32,
    pub material: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct DirectionalLight {
    direction: Vec3,
    _padding_0: f32,
    color: Vec3,
    _padding_1: f32,
}

fn compute_data_size(max_instance_count: u32) -> u64 {
    u64::from(COMPUTE_INSTANCE_DATA_SIZE) * u64::from(max_instance_count)
}

fn graphics_data_instance_size(max_instance_count: u32) -> u64 {
    u64::from(GRAPHICS_INSTANCE_DATA_SIZE) * u64::from(max_instance_count)
}

fn graphics_data_transform_size(max_instance_count: u32) -> u64 {
    u64::from(GRAPHICS_INSTANCE_DATA_SIZE) * u64::from(max_instance_count)
}

fn buffer_size<T>(count: u32) -> u64 {
    mem::size_of::<T>() as u64 * u64::from(count)
}

impl Standard3D {
    pub fn new(
        render: &mut crate::Render,
        render_pass: &mut RenderPassBuilder,
        camera: Shared<Camera>,
        material_set: Shared<PbrMaterialSet>,
        config: &Configuration,
    ) -> (Self, ComputeStage) {
        let shared = SharedData::new(render, camera, material_set, config);

        let slf = Self {
            shared: Arc::new(Mutex::new(shared)),
        };

        unsafe {
            let compute = compute::new(slf.shared.clone(), &render.dev);
            graphics::push(render, render_pass, slf.shared.clone(), config.transparent);
            (slf, compute)
        }
    }

    pub fn set_transform_data(
        &mut self,
        index: usize,
        transform_data: &mut dyn Iterator<Item = TransformScale>,
    ) {
        let mut shared = self.shared.lock().unwrap();
        let mut transform_data = transform_data.take(shared.max_transform_count as usize);
        unsafe {
            shared.per_image[index]
                .compute
                .set_transform_data(&mut transform_data)
        };
    }

    pub fn set_instance_data(
        &mut self,
        index: usize,
        instances_counts: &[u32],
        instances_data: &mut dyn Iterator<Item = Instance>,
    ) {
        let shared = &mut *self.shared.lock().unwrap();

        assert_eq!(
            instances_counts.len(),
            shared.set_data.values().map(|v| v.mesh_set.len()).sum()
        );
        assert!(instances_counts.iter().sum::<u32>() <= shared.max_instance_count);

        unsafe {
            shared.per_image[index].graphics.set_instance_data(
                index,
                &shared.set_data,
                instances_counts,
                instances_data,
            )
        };
    }

    pub fn set_directional_light(&mut self, index: usize, direction: Vec3, color: Vec3) {
        let mut sh = self.shared.lock().unwrap();
        sh.per_image[index].directional_lights_data = DirectionalLight {
            direction,
            _padding_0: 0.0,
            color,
            _padding_1: 0.0,
        };
    }

    pub fn set_camera(&mut self, index: usize, camera: &CameraView) {
        let mut sh = self.shared.lock().unwrap();
        let (m2w, w2p) = sh.camera.set_shared(index, camera);
        let pi = &mut sh.per_image[index];

        let direction = (m2w * Vec4::from((pi.directional_lights_data.direction, 0.0))).truncate();

        unsafe {
            pi.directional_lights_ptr.as_ptr().write(DirectionalLight {
                direction,
                ..pi.directional_lights_data
            });
        }
    }

    pub fn add_texture(
        &mut self,
        render: &mut Render,
        texture: Shared<TextureView>,
    ) -> TextureHandle {
        let sampler = render.dev.nearest_sampler();
        let infos = [texture.bind_info(sampler)];

        let mut shared = self.shared.lock().unwrap();
        let h = shared.textures.insert(texture);
        assert!(h.as_u32() < shared.max_texture_count);

        let writes = shared
            .descriptors
            .sets
            .iter()
            .map(|set| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(*set)
                    .dst_binding(Descriptors::BINDING_TEXTURES)
                    .dst_array_element(h.as_u32())
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&infos)
            })
            .map(|w| w.build())
            .collect::<Vec<_>>();

        unsafe { render.dev.update_descriptor_sets(&writes, &[]) };

        TextureHandle(h)
    }

    pub fn add_mesh_set(
        &mut self,
        render: &mut Render,
        mesh_set: Shared<MeshSet>,
    ) -> MeshSetHandle {
        let graphics = graphics::SetData::new(render, &mesh_set);
        let mut shared = self.shared.lock().unwrap();
        let h = shared.set_data.insert(SetData { graphics, mesh_set });
        MeshSetHandle(h)
    }

    pub fn remove_mesh_set(&mut self, render: &mut Render, mesh_set: MeshSetHandle) {
        let mut shared = self.shared.lock().unwrap();
        let set_data = shared
            .set_data
            .remove(mesh_set.0)
            .expect("no set with handle");
        set_data.drop_with(&mut render.dev);
    }
}

impl SharedData {
    fn new(
        render: &mut crate::Render,
        camera: Shared<Camera>,
        material_set: Shared<PbrMaterialSet>,
        config: &Configuration,
    ) -> Self {
        let descriptors = Descriptors::new(render, config);

        let comp_data_transform_size = compute_data_size(config.max_transform_count);
        let gfx_data_transform_size = graphics_data_transform_size(config.max_transform_count);
        let gfx_data_instance_size = graphics_data_instance_size(config.max_instance_count);

        let per_image = descriptors
            .sets
            .iter()
            .enumerate()
            .map(|(index, &set)| {
                let mut compute_data = render.dev.allocate_buffer(
                    comp_data_transform_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
                    true,
                );
                let mut compute_parameters = render.dev.allocate_buffer(
                    mem::size_of::<vk::DispatchIndirectCommand>() as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                    true,
                );
                let compute_data_ptr = render.dev.map_buffer(&mut compute_data);
                let compute_parameters_ptr = render.dev.map_buffer(&mut compute_parameters);

                let graphics_data_transform = render.dev.allocate_buffer(
                    gfx_data_transform_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    false,
                );
                let mut graphics_data_instance = render.dev.allocate_buffer(
                    gfx_data_instance_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
                    true,
                );
                let graphics_data_instance_ptr = render.dev.map_buffer(&mut graphics_data_instance);

                let mut directional_lights = render.dev.allocate_buffer(
                    mem::size_of::<DirectionalLight>() as _,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    true,
                );
                let directional_lights_ptr = render
                    .dev
                    .map_buffer(&mut directional_lights)
                    .cast::<DirectionalLight>();

                {
                    let info_camera = [vk::DescriptorBufferInfo::builder()
                        .buffer(camera.buffer(index))
                        .offset(0)
                        .range(128)
                        .build()];
                    let info_input = [vk::DescriptorBufferInfo::builder()
                        .buffer(compute_data.0)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                        .build()];
                    let info_output_trf = [vk::DescriptorBufferInfo::builder()
                        .buffer(graphics_data_transform.0)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                        .build()];
                    let info_directional_lights = [vk::DescriptorBufferInfo::builder()
                        .buffer(directional_lights.0)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                        .build()];
                    let info_material_set = [vk::DescriptorBufferInfo::builder()
                        .buffer(material_set.buffer())
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                        .build()];
                    let write = [
                        // COMPUTE
                        // camera
                        vk::WriteDescriptorSet::builder()
                            .dst_set(set)
                            .dst_binding(0)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(&info_camera),
                        // transforms in
                        vk::WriteDescriptorSet::builder()
                            .dst_set(set)
                            .dst_binding(1)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&info_input),
                        // transforms out
                        vk::WriteDescriptorSet::builder()
                            .dst_set(set)
                            .dst_binding(2)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&info_output_trf),
                        // directional lights
                        vk::WriteDescriptorSet::builder()
                            .dst_set(set)
                            .dst_binding(3)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&info_directional_lights),
                        // materials
                        vk::WriteDescriptorSet::builder()
                            .dst_set(set)
                            .dst_binding(4)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(&info_material_set),
                    ]
                    .map(|x| x.build());
                    unsafe { render.dev.update_descriptor_sets(&write, &[]) }
                }

                DataOne {
                    compute: compute::Data {
                        data: compute_data,
                        data_ptr: compute_data_ptr.cast(),
                        parameters: compute_parameters,
                        parameters_ptr: compute_parameters_ptr.cast(),
                    },
                    graphics: graphics::Data {
                        data_instances: graphics_data_instance,
                        data_instances_ptr: graphics_data_instance_ptr.cast(),
                        data_transforms: graphics_data_transform,
                    },
                    directional_lights_data: DirectionalLight {
                        direction: Vec3::NEG_Z,
                        color: Vec3::ZERO,
                        _padding_0: 0.0,
                        _padding_1: 0.0,
                    },
                    directional_lights,
                    directional_lights_ptr,
                }
            })
            .collect();

        Self {
            max_transform_count: config.max_transform_count,
            max_instance_count: config.max_instance_count,
            camera,
            per_image,
            set_data: Default::default(),
            descriptors,
            textures: Default::default(),
            max_texture_count: config.max_texture_count,
            material_set,
        }
    }
}

unsafe impl DropWith for SetData {
    fn drop_with(self, dev: &mut crate::Dev) {
        self.graphics.drop_with(dev);
        self.mesh_set.drop_with(dev);
    }
}

unsafe fn make_shader(dev: &ash::Device, code: &[u32]) -> vk::ShaderModule {
    let info = vk::ShaderModuleCreateInfo::builder().code(code);
    dev.create_shader_module(&info, None).unwrap()
}
