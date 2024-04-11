mod compute;
mod graphics;

pub use compute::ComputeStage;

use super::renderpass::{RenderPassBuilder, SubpassAttachmentReferences};
use crate::resource::material::pbr::PbrMaterialSet;
use crate::resource::texture::TextureSet;
use crate::{mesh::MeshCollection, Dev};
use crate::{Render, VmaBuffer};
use ash::vk;
use core::{ffi::CStr, mem};
use glam::{Quat, Vec3};
use std::sync::{Arc, Mutex};
use vk_mem::Alloc;

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
    max_instance_count: u32,
    descriptor_pool: vk::DescriptorPool,
    meshes: MeshCollection,
    texture_set: TextureSet,
    material_set: PbrMaterialSet,
    per_image: Box<[DataOne]>,
}

struct DataOne {
    compute: compute::Data,
    graphics: graphics::Data,
}

#[derive(Clone, Copy, Debug)]
pub struct Instance {
    pub translation: Vec3,
    pub rotation: Quat,
    pub material: u32,
}

fn compute_data_size(max_instance_count: u32) -> u64 {
    u64::from(COMPUTE_INSTANCE_DATA_SIZE) * u64::from(max_instance_count)
}

fn graphics_data_size(max_instance_count: u32) -> u64 {
    u64::from(GRAPHICS_INSTANCE_DATA_SIZE) * u64::from(max_instance_count)
}

unsafe fn alloc_storage(
    dev: &Dev,
    size: u64,
    as_parameters: bool,
    as_vertex: bool,
    host_visible: bool,
) -> VmaBuffer {
    let b_info = vk::BufferCreateInfo::builder()
        .usage(if as_parameters {
            vk::BufferUsageFlags::INDIRECT_BUFFER
        } else if as_vertex {
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER
        } else {
            vk::BufferUsageFlags::STORAGE_BUFFER
        })
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .size(size);
    let c_info = vk_mem::AllocationCreateInfo {
        flags: vk_mem::AllocationCreateFlags::STRATEGY_MIN_MEMORY
            | if host_visible {
                vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
            } else {
                vk_mem::AllocationCreateFlags::empty()
            },
        usage: if host_visible {
            vk_mem::MemoryUsage::Auto
        } else {
            vk_mem::MemoryUsage::AutoPreferDevice
        },
        required_flags: if host_visible {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::empty()
        },
        ..Default::default()
    };
    dev.alloc.create_buffer(&b_info, &c_info).unwrap()
}

impl Standard3D {
    pub fn new(
        render: &mut crate::Render,
        render_pass: &mut RenderPassBuilder,
        texture_set: TextureSet,
        material_set: PbrMaterialSet,
        meshes: &[crate::Mesh],
        transparent: bool,
    ) -> (Self, ComputeStage) {
        let meshes = crate::mesh::MeshCollectionBuilder { meshes }.finish(render);

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

            slf.add_meshes(render, 1024, meshes, texture_set, material_set);

            (slf, compute)
        }
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
            per.compute.set_instance_data(
                &data.meshes,
                instances_counts,
                instances_data,
                &mut per.graphics,
            )
        };
    }

    unsafe fn add_meshes(
        &mut self,
        render: &mut Render,
        max_instance_count: u32,
        meshes: MeshCollection,
        texture_set: TextureSet,
        material_set: PbrMaterialSet,
    ) {
        let dev = &render.dev;
        let alloc = &dev.alloc;
        let camera = &render.camera;
        let image_count = render.swapchain.image_count() as u32;

        let descriptor_pool = {
            let pool_sizes = [
                // COMPUTE
                // camera (per image)
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: image_count * 10,
                },
                // instance data (per image)
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: image_count * 10,
                },
                // GRAPHICS
                // camera (per image)
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: image_count * 10,
                },
            ];
            let info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&pool_sizes)
                .max_sets(pool_sizes.iter().map(|p| p.descriptor_count).sum());
            dev.create_descriptor_pool(&info, None).unwrap()
        };

        let mut shared = self.shared.lock().unwrap();
        let per_image = (0..image_count as usize)
            .map(|index| {
                let mut compute_data = alloc_storage(
                    dev,
                    compute_data_size(max_instance_count),
                    false,
                    true, //false,
                    true,
                );
                let mut compute_parameters = alloc_storage(
                    dev,
                    graphics_data_size(max_instance_count),
                    true,
                    false,
                    true,
                );
                let compute_data_ptr = alloc.map_memory(&mut compute_data.1).unwrap();
                let compute_parameters_ptr = alloc.map_memory(&mut compute_parameters.1).unwrap();

                let graphics_data = alloc_storage(
                    dev,
                    graphics_data_size(max_instance_count),
                    false,
                    true,
                    false,
                );
                let mut graphics_parameters = alloc_storage(
                    dev,
                    graphics_data_size(max_instance_count),
                    true,
                    false,
                    true,
                );
                let graphics_parameters_ptr = alloc.map_memory(&mut graphics_parameters.1).unwrap();

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

                {
                    let info_camera = vk::DescriptorBufferInfo::builder()
                        .buffer(camera.buffer(index))
                        .offset(0)
                        .range(64)
                        .build();
                    let info_input = [vk::DescriptorBufferInfo::builder()
                        .buffer(compute_data.0)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                        .build()];
                    let info_output = [vk::DescriptorBufferInfo::builder()
                        .buffer(graphics_data.0)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                        .build()];
                    let info_camera_inv = [vk::DescriptorBufferInfo::builder()
                        .buffer(camera.buffer(index))
                        .offset(64)
                        .range(64)
                        .build()];
                    let info_directional_lights = [vk::DescriptorBufferInfo::builder()
                        .buffer({
                            let b_info = vk::BufferCreateInfo::builder()
                                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                                .size(mem::size_of::<[[f32; 4]; 2]>() as _);
                            let c_info = vk_mem::AllocationCreateInfo {
                                flags: vk_mem::AllocationCreateFlags::STRATEGY_MIN_MEMORY
                                    | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                                usage: vk_mem::MemoryUsage::Auto,
                                required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                                    | vk::MemoryPropertyFlags::HOST_COHERENT,
                                ..Default::default()
                            };
                            let mut buf = alloc.create_buffer(&b_info, &c_info).unwrap();
                            alloc
                                .map_memory(&mut buf.1)
                                .unwrap()
                                .cast::<[[f32; 4]; 2]>()
                                .write([
                                    glam::Vec4::new(0.0, 0.0, -1.0, 0.0).normalize().to_array(),
                                    (glam::Vec4::new(1.0, 1.0, 1.0, 0.0) * 5.0).to_array(),
                                ]);
                            buf.0
                        })
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
                            .buffer_info(&[info_camera]),
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
                            .buffer_info(&info_output),
                        // GRAPHICS - VERTEX
                        // camera
                        vk::WriteDescriptorSet::builder()
                            .dst_set(graphics_descriptor_set)
                            .dst_binding(0)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(&info_camera_inv),
                        // GRAPHICS - FRAGMENT
                        // directional lights
                        vk::WriteDescriptorSet::builder()
                            .dst_set(graphics_descriptor_set)
                            .dst_binding(1)
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
                        data_ptr: compute_data_ptr,
                        parameters: compute_parameters,
                        parameters_ptr: compute_parameters_ptr,
                        descriptor_set: compute_descriptor_set,
                    },
                    graphics: graphics::Data {
                        data: graphics_data,
                        parameters: graphics_parameters,
                        parameters_ptr: graphics_parameters_ptr,
                        descriptor_set: graphics_descriptor_set,
                    },
                }
            })
            .collect();
        shared.data_sets.push(Data {
            max_instance_count,
            descriptor_pool,
            per_image,
            meshes,
            texture_set,
            material_set,
        });
    }
}

unsafe fn make_shader(dev: &ash::Device, code: &[u32]) -> vk::ShaderModule {
    let info = vk::ShaderModuleCreateInfo::builder().code(code);
    dev.create_shader_module(&info, None).unwrap()
}
