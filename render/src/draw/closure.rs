use crate::{material::PbrMaterialView, mesh::MeshCollection, VmaBuffer};
use ash::vk::{self, PipelineLayoutCreateFlags};
use core::{ffi::CStr, mem};
use glam::{Vec2, Vec3};
use vk_mem::Alloc;
use vk_shader_macros::include_glsl;

/// vec3 (position) + float (scale) + vec4 (rotation)
const COMPUTE_INSTANCE_DATA_SIZE: u32 = (3 + 1 + 4) * 4;
/// mat4 (projection)
const GRAPHICS_INSTANCE_DATA_SIZE: u32 = 4 * 4 * 4;

pub struct DrawClosure {
    descriptor_pool: vk::DescriptorPool,
    meshes: MeshCollection,
    command: Box<[DrawCommand]>,
    max_instances: u32,
}

struct DrawCommand {
    command: vk::CommandBuffer,
    compute: DrawCommandCompute,
    graphics: DrawCommandGraphics,
}

struct DrawCommandCompute {
    /// Instance transforms + Dispatch parameters.
    ///
    /// - instance data
    ///
    /// Set by host, so host-visible.
    data: VmaBuffer,
    /// Dispatch parameters
    ///
    /// - parameters
    ///
    /// Set by host, so host-visible.
    parameters: VmaBuffer,
    /// - camera mat4 (uniform)
    /// - compute_data (storage)
    /// - graphics_data (storage)
    descriptor_set: vk::DescriptorSet,
}

struct DrawCommandGraphics {
    /// Projected instance transforms + Draw parameters
    ///
    /// - instance data
    ///
    /// Set by compute shader, so device-local.
    data: VmaBuffer,
    /// Draw parameters
    ///
    /// - parameters count
    /// - parameters array
    ///
    /// (currently) Set by host, so host-visible.
    parameters: VmaBuffer,
    /// - camera near/far (uniform)
    descriptor_set: vk::DescriptorSet,
}

impl DrawClosure {
    pub unsafe fn new(
        dev: &ash::Device,
        alloc: &vk_mem::Allocator,
        cmd_pool: vk::CommandPool,
        function: &super::DrawFunction,
        image_count: u32,
        meshes: MeshCollection,
        max_instances: u32,
        camera: &crate::camera::Camera,

        // FIXME BRO BRO BRO NONONO
        Q__queue: vk::Queue,
    ) -> Self {
        let descriptor_pool = make_descriptor_pool(dev, image_count);
        let command = make_command(
            dev,
            alloc,
            cmd_pool,
            descriptor_pool,
            &function,
            camera,
            image_count,
            max_instances,
            &meshes,
            Q__queue,
        );
        Self {
            descriptor_pool,
            command,
            meshes,
            max_instances,
        }
    }

    pub unsafe fn drop_with(&mut self, dev: &ash::Device, cmd_pool: vk::CommandPool) {
        /*
        dev.free_command_buffers(cmd_pool, &self.command);
        for &sem in self.semaphores.iter() {
            dev.destroy_semaphore(sem, None);
        }
        for &(pl, pll) in self.pipelines.iter() {
            dev.destroy_pipeline(pl, None);
            dev.destroy_pipeline_layout(pll, None);
        }
        for &rp in self.render_passes.iter() {
            dev.destroy_render_pass(rp, None);
        }
        */
    }
}

unsafe fn make_descriptor_pool(dev: &ash::Device, image_count: u32) -> vk::DescriptorPool {
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
}

fn compute_data_size(max_instance_count: u32) -> u64 {
    u64::from(COMPUTE_INSTANCE_DATA_SIZE) * u64::from(max_instance_count)
}

fn graphics_data_size(max_instance_count: u32) -> u64 {
    u64::from(GRAPHICS_INSTANCE_DATA_SIZE) * u64::from(max_instance_count)
}

unsafe fn alloc_storage(
    alloc: &vk_mem::Allocator,
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
    alloc.create_buffer(&b_info, &c_info).unwrap()
}

unsafe fn make_command(
    dev: &ash::Device,
    alloc: &vk_mem::Allocator,
    cmdpool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    function: &super::DrawFunction,
    camera: &crate::camera::Camera,
    image_count: u32,
    max_instance_count: u32,
    meshes: &MeshCollection,
    Q__queue: vk::Queue,
) -> Box<[DrawCommand]> {
    let info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(cmdpool)
        .command_buffer_count(image_count);
    dev.allocate_command_buffers(&info)
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(index, command)| {
            let mut compute_data = alloc_storage(
                alloc,
                compute_data_size(max_instance_count),
                false,
                true, //false,
                true,
            );
            let mut compute_parameters = alloc_storage(
                alloc,
                graphics_data_size(max_instance_count),
                true,
                false,
                true,
            );
            let graphics_data = alloc_storage(
                alloc,
                graphics_data_size(max_instance_count),
                false,
                true,
                false,
            );
            let mut graphics_parameters = alloc_storage(
                alloc,
                graphics_data_size(max_instance_count),
                true,
                false,
                true,
            );
            let [compute_descriptor_set, graphics_descriptor_set] = {
                let layouts = [
                    function.compute_pipeline.descriptor_set_layout,
                    function.graphics_pipeline.descriptor_set_layout,
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
                #[repr(C)]
                struct Instance {
                    pos: [f32; 3],
                    scale: f32,
                    rot: [f32; 3],
                    material: u32,
                }
                let p = alloc.map_memory(&mut compute_data.1).unwrap();
                let r2a = |mut q: glam::Quat| {
                    if q.w < 0.0 {
                        q = -q;
                    }
                    q.xyz().to_array()
                };
                p.cast::<[Instance; 7]>().write([
                    Instance {
                        pos: [0.0, -2.0, 0.0],
                        scale: 0.5,
                        rot: [0.0; 3],
                        material: 1,
                    },
                    Instance {
                        pos: [0.0; 3],
                        scale: 1.0,
                        rot: r2a(glam::Quat::from_rotation_x(-1.0)),
                        material: 0,
                    },
                    Instance {
                        pos: [0.0, 4.0, 0.0],
                        scale: 1.5,
                        rot: r2a(glam::Quat::from_rotation_z(1.0)),
                        material: 0,
                    },
                    Instance {
                        pos: [-3.0, -2.0, 0.0],
                        scale: 1.5,
                        rot: [0.0; 3],
                        material: 1,
                    },
                    Instance {
                        pos: [-3.0, 2.0, 0.0],
                        scale: 1.0,
                        rot: r2a(glam::Quat::from_rotation_y(2.0)),
                        material: 0,
                    },
                    Instance {
                        pos: [-3.0, 4.0, 0.0],
                        scale: 0.5,
                        rot: [0.0; 3],
                        material: 0,
                    },
                    Instance {
                        pos: [0.0, 0.0, 5.0],
                        scale: 1.0,
                        rot: [0.0; 3],
                        material: 1,
                    },
                ]);
                alloc.unmap_memory(&mut compute_data.1);
            }

            {
                let p = alloc.map_memory(&mut compute_parameters.1).unwrap();
                p.cast::<vk::DispatchIndirectCommand>()
                    .write_unaligned(vk::DispatchIndirectCommand { x: 7, y: 1, z: 1 });
                alloc.unmap_memory(&mut compute_parameters.1);
            }

            {
                let monkey = meshes.mesh(0);
                let donut = meshes.mesh(1);
                let cube = meshes.mesh(2);
                let p = alloc.map_memory(&mut graphics_parameters.1).unwrap();
                p.cast::<u32>().write_unaligned(3);
                p.add(4)
                    .cast::<[vk::DrawIndexedIndirectCommand; 3]>()
                    .write([
                        vk::DrawIndexedIndirectCommand {
                            vertex_offset: monkey.vertex_offset.try_into().unwrap(),
                            first_index: monkey.index_offset,
                            index_count: monkey.index_count,
                            instance_count: 3,
                            first_instance: 0,
                        },
                        vk::DrawIndexedIndirectCommand {
                            vertex_offset: donut.vertex_offset.try_into().unwrap(),
                            first_index: donut.index_offset,
                            index_count: donut.index_count,
                            instance_count: 3,
                            first_instance: 3,
                        },
                        vk::DrawIndexedIndirectCommand {
                            vertex_offset: cube.vertex_offset.try_into().unwrap(),
                            first_index: cube.index_offset,
                            index_count: cube.index_count,
                            instance_count: 1,
                            first_instance: 6,
                        },
                    ]);
                alloc.unmap_memory(&mut graphics_parameters.1);
            }

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
                let info_stub_texture =
                    [vk::DescriptorImageInfo::builder()
                        .sampler({
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
                                .compare_op(vk::CompareOp::NEVER)
                                //.border_color(vk::BorderColor::INT_OPAQUE_BLACK)
                                .unnormalized_coordinates(false);
                            dev.create_sampler(&info, None).unwrap()
                        })
                        .image_view({
                            let mut image = alloc.create_image(
                            &vk::ImageCreateInfo::builder()
                                .image_type(vk::ImageType::TYPE_2D)
                                .format(vk::Format::R8G8B8A8_UNORM)
                                .mip_levels(1)
                                .array_layers(1)
                                .samples(vk::SampleCountFlags::TYPE_1)
                                //.tiling(vk::ImageTiling::OPTIMAL)
                                .tiling(vk::ImageTiling::LINEAR)
                                .usage(vk::ImageUsageFlags::SAMPLED)
                                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                                .queue_family_indices(&[])
                                .extent(vk::Extent3D {
                                    width: 16,
                                    height: 16,
                                    depth: 1,
                                })
                                .initial_layout(vk::ImageLayout::PREINITIALIZED),
                            &vk_mem::AllocationCreateInfo {
                                flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                                ..Default::default()
                            },
                        ).unwrap();

                                let bitmap = [
                                    0b0000_0000_0000_0000,
                                    0b0000_0000_0000_0000,
                                    0b0000_1100_0011_0000,
                                    0b0001_1110_0111_1000,
                                    0b0001_1110_0111_1000,
                                    0b0000_1100_0011_0000,
                                    0b0000_0000_0000_0000,
                                    0b0000_0000_0000_0000,
                                    0b0010_0000_0000_0100,
                                    0b0011_0000_0000_1100,
                                    0b0001_1000_0001_1000,
                                    0b0000_1111_1111_0000,
                                    0b0000_0111_1110_0000,
                                    0b0000_0000_0000_0000,
                                    0b0000_0000_0000_0000,
                                    0b0000_0000_0000_0000,
                                ];

                                let mut p = alloc.map_memory(&mut image.1).unwrap().cast::<u32>();
                                for x in 0..16 { for y in 0..16 {
                                *p = u32::MAX * (!(x / 4 ^ y / 4) & 1);
                                *p = u32::MAX * (!(bitmap[15-y as usize] >> x) & 1);
                                p = p.add(1);
                    }}
                            alloc.unmap_memory(&mut image.1);
                            alloc.flush_allocation(&image.1, 0, 16*16*4).unwrap();

            {
    let info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(cmdpool)
        .command_buffer_count(image_count);
    let cmdbuf = dev.allocate_command_buffers(&info).unwrap()[0];
    let info = vk::CommandBufferBeginInfo::builder();
    dev.begin_command_buffer(cmdbuf, &info).unwrap();
                let image_barriers = [vk::ImageMemoryBarrier::builder()
                    .image(image.0)
                    .src_access_mask(vk::AccessFlags::HOST_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::PREINITIALIZED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 })
                    .build()];
                dev.cmd_pipeline_barrier(
                    cmdbuf,
                    vk::PipelineStageFlags::HOST,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &image_barriers,
                );
                dev.end_command_buffer(cmdbuf).unwrap();
                let cmdbufs = [cmdbuf];
                let submits = [vk::SubmitInfo::builder()
                .command_buffers(&cmdbufs).build()];
                dev.queue_submit(Q__queue, &submits, vk::Fence::null()).unwrap();
                dev.device_wait_idle().unwrap();
            }

                            let info = vk::ImageViewCreateInfo::builder()
                                .image(image.0)
                                .view_type(vk::ImageViewType::TYPE_2D)
                                .format(vk::Format::R8G8B8A8_UNORM)
                                .subresource_range(vk::ImageSubresourceRange {
                                    aspect_mask: vk::ImageAspectFlags::COLOR,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                });
                            dev.create_image_view(&info, None).unwrap()
                        })
                        .image_layout(vk::ImageLayout::GENERAL)
                        .build()];
                let info_materials = [vk::DescriptorBufferInfo::builder()
                    .buffer({
                        let b_info = vk::BufferCreateInfo::builder()
                            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                            .sharing_mode(vk::SharingMode::EXCLUSIVE)
                            .size(mem::size_of::<[PbrMaterialView; 2]>() as _);
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
                            .cast::<[PbrMaterialView; 2]>()
                            .write([PbrMaterialView {
                                uv_offset: [0.0; 2],
                                uv_scale: [0.0; 2],
                                albedo: [1.0; 3],
                                albedo_texture_index: 0,
                                roughness: 0.5,
                                roughness_texture_id: 0,
                                metallic: 0.5,
                                metallic_texture_id: 0,
                                ambient_occlusion: 1.0,
                                ambient_occlusion_texture_id: 0,
                                _padding: [0; 2],
                            }, PbrMaterialView {
                                uv_offset: [0.0; 2],
                                uv_scale: [1.0; 2],
                                albedo: [1.0, 1.0, 0.0],
                                albedo_texture_index: 0,
                                roughness: 0.5,
                                roughness_texture_id: 0,
                                metallic: 0.5,
                                metallic_texture_id: 0,
                                ambient_occlusion: 1.0,
                                ambient_occlusion_texture_id: 0,
                                _padding: [0; 2],
                            }]);
                        buf.0
                    })
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
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
                                //glam::Vec4::new(0.0, 1.0, 1.0, 0.0).normalize().to_array(),
                                glam::Vec4::new(0.0, 0.0, -1.0, 0.0).normalize().to_array(),
                                //glam::Vec4::new(0.0, 0.0, 1.0, 0.0).normalize().to_array(),
                                //glam::Vec4::new(-1.0, -1.0, -1.0, 0.0).normalize().to_array(),
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
                        .buffer_info(&info_camera)
                        .build(),
                    // input
                    vk::WriteDescriptorSet::builder()
                        .dst_set(compute_descriptor_set)
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&info_input)
                        .build(),
                    // output
                    vk::WriteDescriptorSet::builder()
                        .dst_set(compute_descriptor_set)
                        .dst_binding(2)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&info_output)
                        .build(),
                    // GRAPHICS - VERTEX
                    // camera
                    vk::WriteDescriptorSet::builder()
                        .dst_set(graphics_descriptor_set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&info_camera_inv)
                        .build(),
                    // GRAPHICS - FRAGMENT
                    // albedo texture
                    vk::WriteDescriptorSet::builder()
                        .dst_set(graphics_descriptor_set)
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&info_stub_texture)
                        .build(),
                    // roughness texture
                    vk::WriteDescriptorSet::builder()
                        .dst_set(graphics_descriptor_set)
                        .dst_binding(2)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&info_stub_texture)
                        .build(),
                    // metallic texture
                    vk::WriteDescriptorSet::builder()
                        .dst_set(graphics_descriptor_set)
                        .dst_binding(3)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&info_stub_texture)
                        .build(),
                    // ambient occlusion texture
                    vk::WriteDescriptorSet::builder()
                        .dst_set(graphics_descriptor_set)
                        .dst_binding(4)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&info_stub_texture)
                        .build(),
                    // material
                    vk::WriteDescriptorSet::builder()
                        .dst_set(graphics_descriptor_set)
                        .dst_binding(5)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&info_materials)
                        .build(),
                    // directional lights
                    vk::WriteDescriptorSet::builder()
                        .dst_set(graphics_descriptor_set)
                        .dst_binding(6)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&info_directional_lights)
                        .build(),
                ];
                dev.update_descriptor_sets(&write, &[]);
            }

            DrawCommand {
                command,
                compute: DrawCommandCompute {
                    data: compute_data,
                    parameters: compute_parameters,
                    descriptor_set: compute_descriptor_set,
                },
                graphics: DrawCommandGraphics {
                    data: graphics_data,
                    parameters: graphics_parameters,
                    descriptor_set: graphics_descriptor_set,
                },
            }
        })
        .collect()
}

impl DrawClosure {
    /// Recorded command requires a queue that supports both GRAPHICS and COMPUTE.
    pub unsafe fn record_command(
        &mut self,
        dev: &ash::Device,
        function: &super::DrawFunction,
        viewport: vk::Extent2D,
        framebuffers: &[vk::Framebuffer],
    ) {
        assert_eq!(self.command.len(), framebuffers.len());
        for (cmd, &fb) in self.command.iter().zip(framebuffers) {
            let info = vk::CommandBufferBeginInfo::builder();
            dev.begin_command_buffer(cmd.command, &info).unwrap();

            // compute
            dev.cmd_bind_pipeline(
                cmd.command,
                vk::PipelineBindPoint::COMPUTE,
                function.compute_pipeline.pipeline,
            );
            dev.cmd_bind_descriptor_sets(
                cmd.command,
                vk::PipelineBindPoint::COMPUTE,
                function.compute_pipeline.layout,
                0,
                &[cmd.compute.descriptor_set],
                &[],
            );
            dev.cmd_dispatch_indirect(cmd.command, cmd.compute.parameters.0, 0);

            // sync
            // TODO consider using buffer barrier, which has higher granulity
            {
                let memory_barriers = [vk::MemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::VERTEX_ATTRIBUTE_READ)
                    .build()];
                dev.cmd_pipeline_barrier(
                    cmd.command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::VERTEX_INPUT,
                    vk::DependencyFlags::empty(),
                    &memory_barriers,
                    &[],
                    &[],
                );
            }

            // graphics
            let clearvalues = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0., 0., 0., 1.],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        //depth: 0.0,
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];
            let info = vk::RenderPassBeginInfo::builder()
                .render_pass(function.render_pass)
                .framebuffer(fb)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: viewport,
                })
                .clear_values(&clearvalues);
            dev.cmd_begin_render_pass(cmd.command, &info, vk::SubpassContents::INLINE);
            dev.cmd_bind_pipeline(
                cmd.command,
                vk::PipelineBindPoint::GRAPHICS,
                function.graphics_pipeline.pipeline,
            );
            dev.cmd_bind_descriptor_sets(
                cmd.command,
                vk::PipelineBindPoint::GRAPHICS,
                function.graphics_pipeline.layout,
                0,
                &[cmd.graphics.descriptor_set],
                &[],
            );
            dev.cmd_bind_vertex_buffers(
                cmd.command,
                0,
                &[
                    self.meshes.vertex_data.0,
                    self.meshes.vertex_data.0,
                    self.meshes.vertex_data.0,
                    // FIXME lol, lmao
                    cmd.compute.data.0,
                    cmd.graphics.data.0,
                ],
                &[
                    self.meshes.positions_offset,
                    self.meshes.normals_offset,
                    self.meshes.uvs_offset,
                    0,
                    0,
                ],
            );
            let inv_viewport = {
                let vp = Vec2::new(viewport.width as f32, viewport.height as f32);
                let inv_vp = 1.0 / vp;
                Vec3::new(inv_vp.x, inv_vp.y, vp.y * inv_vp.x)
            };
            dev.cmd_push_constants(
                cmd.command,
                function.graphics_pipeline.layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                crate::f32_to_bytes(&inv_viewport.to_array()),
            );
            dev.cmd_bind_index_buffer(
                cmd.command,
                self.meshes.index_data.0,
                0,
                vk::IndexType::UINT32,
            );
            let viewports = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: viewport.width as f32,
                height: viewport.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let scissors = [vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: viewport,
            }];
            dev.cmd_set_viewport(cmd.command, 0, &viewports);
            dev.cmd_set_scissor(cmd.command, 0, &scissors);
            dev.cmd_draw_indexed_indirect_count(
                cmd.command,
                cmd.graphics.parameters.0,
                4,
                cmd.graphics.parameters.0,
                0,
                self.meshes.len() as u32,
                mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
            dev.cmd_end_render_pass(cmd.command);

            dev.end_command_buffer(cmd.command).unwrap();
        }
    }
}

impl DrawClosure {
    pub unsafe fn submit(
        &self,
        dev: &ash::Device,
        commands: &mut crate::command::Commands,
        info: &crate::swapchain::DrawInfo,
    ) {
        let wait = [info.available];
        let waiting_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal = [info.finished];
        let command = [self.command[info.index].command];
        let submit_info = [vk::SubmitInfo::builder()
            .wait_semaphores(&wait)
            .wait_dst_stage_mask(&waiting_stages)
            .command_buffers(&command)
            .signal_semaphores(&signal)
            .build()];
        unsafe {
            dev.queue_submit(commands.queues.graphics, &submit_info, info.may_draw)
                .unwrap()
        };
    }
}
