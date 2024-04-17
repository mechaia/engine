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
mod graphics;

pub use compute::ComputeStage;
use glam::{Vec3, Vec4};

use super::renderpass::RenderPassBuilder;
use crate::resource::camera::{Camera, CameraView};
use crate::resource::{material::pbr::PbrMaterialSet, mesh::MeshSet, texture::TextureSet};
use crate::{Render, VmaBuffer};
use ash::vk;
use core::{ffi::CStr, mem, ptr::NonNull};
use std::sync::{Arc, Mutex};
use util::Transform;

const ENTRY_POINT: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

/// vec3 (position) + uint (material) + vec4 (rotation)
const COMPUTE_INSTANCE_DATA_SIZE: u32 = (3 + 1 + 4) * 4;
/// mat4 (projection)
const GRAPHICS_INSTANCE_DATA_SIZE: u32 = 4 * 4 * 4;

pub struct Standard3D {
    shared: Shared,
}

type Shared = Arc<Mutex<SharedData>>;

struct SharedData {
    compute_descriptor_set_layout: vk::DescriptorSetLayout,
    graphics_descriptor_set_layout: vk::DescriptorSetLayout,
    data_sets: Vec<Data>,
}

struct Data {
    max_transform_count: u32,
    max_instance_count: u32,
    descriptor_pool: vk::DescriptorPool,
    meshes: MeshSet,
    texture_set: TextureSet,
    material_set: PbrMaterialSet,
    camera: Camera,
    per_image: Box<[DataOne]>,
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
        texture_set: TextureSet,
        material_set: PbrMaterialSet,
        mesh_set: MeshSet,
        camera: Camera,
        transparent: bool,
        max_transform_count: u32,
        max_instance_count: u32,
    ) -> (Self, ComputeStage) {
        let mut slf = Self {
            shared: Arc::new(Mutex::new(SharedData {
                data_sets: Vec::new(),
                compute_descriptor_set_layout: vk::DescriptorSetLayout::null(),
                graphics_descriptor_set_layout: vk::DescriptorSetLayout::null(),
            })),
        };

        unsafe {
            let compute = compute::new(slf.shared.clone(), &render.dev);
            graphics::push(
                render,
                render_pass,
                slf.shared.clone(),
                material_set.layout(),
                texture_set.layout(),
                transparent,
            );

            slf.add_meshes(
                render,
                max_transform_count,
                max_instance_count,
                mesh_set,
                texture_set,
                material_set,
                camera,
            );

            (slf, compute)
        }
    }

    pub fn set_transform_data(
        &mut self,
        index: usize,
        transform_data: &mut dyn Iterator<Item = Transform>,
    ) {
        let mut shared = self.shared.lock().unwrap();
        let data = &mut shared.data_sets[0 /* FIXME */];
        let per = &mut data.per_image[index];

        let mut transform_data = transform_data.take(data.max_transform_count as usize);

        unsafe {
            per.compute
                .set_transform_data(&data.meshes, &mut transform_data)
        };
    }

    pub fn set_instance_data(
        &mut self,
        index: usize,
        instances_counts: &[u32],
        instances_data: &mut dyn Iterator<Item = Instance>,
    ) {
        let mut shared = self.shared.lock().unwrap();
        let data = &mut shared.data_sets[0 /* FIXME */];
        let per = &mut data.per_image[index];

        assert_eq!(instances_counts.len(), data.meshes.len());
        assert!(instances_counts.iter().sum::<u32>() <= data.max_instance_count);

        unsafe {
            per.graphics
                .set_instance_data(&data.meshes, instances_counts, instances_data)
        };
    }

    pub fn set_directional_light(&mut self, index: usize, direction: Vec3, color: Vec3) {
        let mut sh = self.shared.lock().unwrap();
        let dt = &mut sh.data_sets[0].per_image[index];
        let direction = Vec3::NEG_Z;
        /*
        let direction = Vec3::NEG_ONE.normalize();
        let direction = Vec3::new(-1.0, -2.0, -2.0).normalize();
        let direction = Vec3::new(-1.0, 0.0, 0.0).normalize();
        */
        let direction = Vec3::new(-1.0, 0.0, -1.0).normalize();
        dt.directional_lights_data = DirectionalLight {
            direction,
            _padding_0: 0.0,
            color,
            _padding_1: 0.0,
        };
    }

    pub fn set_camera(&mut self, index: usize, camera: &CameraView) {
        let mut sh = self.shared.lock().unwrap();
        let dt = &mut sh.data_sets[0];
        let (m2w, w2p) = dt.camera.set(index, camera);
        let pi = &mut dt.per_image[index];

        let direction = (m2w * Vec4::from((pi.directional_lights_data.direction, 0.0))).truncate();

        unsafe {
            pi.directional_lights_ptr.as_ptr().write(DirectionalLight {
                direction,
                ..pi.directional_lights_data
            });
        }
    }

    unsafe fn add_meshes(
        &mut self,
        render: &mut Render,
        max_transform_count: u32,
        max_instance_count: u32,
        meshes: MeshSet,
        texture_set: TextureSet,
        material_set: PbrMaterialSet,
        camera: Camera,
    ) {
        let dev = &mut render.dev;
        let image_count = render.swapchain.image_count() as u32;

        let descriptor_pool = {
            let pool_sizes = [
                // COMPUTE
                // camera (per image)
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: image_count * 30,
                },
                // instance data (per image)
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: image_count * 30,
                },
                // GRAPHICS
                // camera (per image)
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: image_count * 30,
                },
            ];
            let info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&pool_sizes)
                .max_sets(pool_sizes.iter().map(|p| p.descriptor_count).sum());
            dev.create_descriptor_pool(&info, None).unwrap()
        };

        let comp_data_transform_size = compute_data_size(max_transform_count);
        let gfx_data_transform_size = graphics_data_transform_size(max_transform_count);
        let gfx_data_instance_size = graphics_data_instance_size(max_instance_count);

        let mut shared = self.shared.lock().unwrap();
        let per_image = (0..image_count as usize)
            .map(|index| {
                let mut compute_data = dev.allocate_buffer(
                    comp_data_transform_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
                    true,
                );
                let mut compute_parameters = dev.allocate_buffer(
                    mem::size_of::<vk::DispatchIndirectCommand>() as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                    true,
                );
                let compute_data_ptr = dev.map_buffer(&mut compute_data);
                let compute_parameters_ptr = dev.map_buffer(&mut compute_parameters);

                let graphics_data_transform = dev.allocate_buffer(
                    gfx_data_transform_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    false,
                );
                let mut graphics_data_instance = dev.allocate_buffer(
                    gfx_data_instance_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
                    true,
                );
                let graphics_data_instance_ptr = dev.map_buffer(&mut graphics_data_instance);
                let mut graphics_parameters = dev.allocate_buffer(
                    4 + buffer_size::<vk::DrawIndexedIndirectCommand>(
                        u32::try_from(meshes.len()).unwrap(),
                    ),
                    vk::BufferUsageFlags::INDIRECT_BUFFER,
                    true,
                );
                let graphics_parameters_ptr = dev.map_buffer(&mut graphics_parameters);

                let [compute_descriptor_set, graphics_descriptor_set] = {
                    let layouts = [
                        shared.compute_descriptor_set_layout,
                        shared.graphics_descriptor_set_layout,
                    ];
                    let info = vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&layouts);
                    dev.allocate_descriptor_sets(&info)
                        .unwrap()
                        .try_into()
                        .unwrap()
                };

                let mut directional_lights = dev.allocate_buffer(
                    mem::size_of::<DirectionalLight>() as _,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    true,
                );
                let directional_lights_ptr = dev
                    .map_buffer(&mut directional_lights)
                    .cast::<DirectionalLight>();

                {
                    let info_camera = [vk::DescriptorBufferInfo::builder()
                        .buffer(camera.buffer(index))
                        .offset(0)
                        .range(64)
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
                    let info_camera_inv = [vk::DescriptorBufferInfo::builder()
                        .buffer(camera.buffer(index))
                        .offset(64)
                        .range(64)
                        .build()];
                    let info_directional_lights = [vk::DescriptorBufferInfo::builder()
                        .buffer(directional_lights.0)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                        .build()];
                    let write = [
                        // COMPUTE
                        // camera
                        vk::WriteDescriptorSet::builder()
                            .dst_set(compute_descriptor_set)
                            .dst_binding(0)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(&info_camera),
                        // input
                        vk::WriteDescriptorSet::builder()
                            .dst_set(compute_descriptor_set)
                            .dst_binding(1)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&info_input),
                        // output
                        vk::WriteDescriptorSet::builder()
                            .dst_set(compute_descriptor_set)
                            .dst_binding(2)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&info_output_trf),
                        // GRAPHICS - VERTEX
                        // camera
                        vk::WriteDescriptorSet::builder()
                            .dst_set(graphics_descriptor_set)
                            .dst_binding(0)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(&info_camera_inv),
                        // transforms
                        vk::WriteDescriptorSet::builder()
                            .dst_set(graphics_descriptor_set)
                            .dst_binding(1)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&info_output_trf),
                        // GRAPHICS - FRAGMENT
                        // directional lights
                        vk::WriteDescriptorSet::builder()
                            .dst_set(graphics_descriptor_set)
                            .dst_binding(2)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&info_directional_lights),
                    ]
                    .map(|x| x.build());
                    dev.update_descriptor_sets(&write, &[]);
                }

                DataOne {
                    compute: compute::Data {
                        data: compute_data,
                        data_ptr: compute_data_ptr.cast(),
                        parameters: compute_parameters,
                        parameters_ptr: compute_parameters_ptr.cast(),
                        descriptor_set: compute_descriptor_set,
                    },
                    graphics: graphics::Data {
                        data_instances: graphics_data_instance,
                        data_instances_ptr: graphics_data_instance_ptr.cast(),
                        data_transforms: graphics_data_transform,
                        parameters: graphics_parameters,
                        parameters_ptr: graphics_parameters_ptr,
                        descriptor_set: graphics_descriptor_set,
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
        shared.data_sets.push(Data {
            max_transform_count,
            max_instance_count,
            descriptor_pool,
            per_image,
            meshes,
            texture_set,
            material_set,
            camera,
        });
    }
}

unsafe fn make_shader(dev: &ash::Device, code: &[u32]) -> vk::ShaderModule {
    let info = vk::ShaderModuleCreateInfo::builder().code(code);
    dev.create_shader_module(&info, None).unwrap()
}
