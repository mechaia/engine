use ash::vk;
use core::mem;
use glam::{U64Vec3, UVec3};
use vk_mem::Alloc;

const DRAW_PARAMETERS_SIZE: u32 = mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32;

pub struct Commands {
    pub queues: super::queues::Queues,
    pub pool: vk::CommandPool,
}

impl Commands {
    pub fn new(dev: &ash::Device, queues: super::queues::Queues) -> Self {
        let mkpool = |index| {
            let info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            unsafe { dev.create_command_pool(&info, None).unwrap() }
        };
        let pool = mkpool(queues.graphics_index);
        Self { queues, pool }
    }

    unsafe fn oneshot(
        &mut self,
        dev: &ash::Device,
        f: &mut dyn FnMut(&ash::Device, vk::CommandBuffer),
    ) {
        // alloc
        let info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.pool)
            .command_buffer_count(1);
        let cmdbuf = dev.allocate_command_buffers(&info).unwrap()[0];
        // record
        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        dev.begin_command_buffer(cmdbuf, &info).unwrap();
        f(dev, cmdbuf);
        dev.end_command_buffer(cmdbuf).unwrap();
        // submit
        let command_buffers = [cmdbuf];
        let submit_info = [vk::SubmitInfo::builder()
            .command_buffers(&command_buffers)
            .build()];
        dev.queue_submit(self.queues.graphics, &submit_info, vk::Fence::null())
            .unwrap();
        dev.queue_wait_idle(self.queues.graphics).unwrap();
        // clean
        dev.free_command_buffers(self.pool, &[cmdbuf]);
    }

    pub unsafe fn transfer_between(
        &mut self,
        dev: &ash::Device,
        dst: vk::Buffer,
        dst_offset: u64,
        src: vk::Buffer,
        src_offset: u64,
        amount: usize,
    ) {
        self.oneshot(dev, &mut |dev, cmdbuf| {
            dev.cmd_copy_buffer(
                cmdbuf,
                src,
                dst,
                &[vk::BufferCopy {
                    src_offset,
                    dst_offset,
                    size: amount as u64,
                }],
            );
        });
    }

    pub unsafe fn transfer_to(
        &mut self,
        dev: &ash::Device,
        alloc: &vk_mem::Allocator,
        dst: vk::Buffer,
        dst_offset: u64,
        src: *const u8,
        amount: usize,
    ) {
        with_buffer(dev, alloc, amount, &mut |dev, buf, ptr| {
            ptr.copy_from_nonoverlapping(src, amount);
            self.transfer_between(dev, dst, dst_offset, buf, 0, amount);
        });
    }

    unsafe fn transfer_between_images(
        &mut self,
        dev: &ash::Device,
        dst: vk::Image,
        dst_offset: UVec3,
        src: vk::Image,
        src_offset: UVec3,
        dimensions: UVec3,
    ) {
        let info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.pool)
            .command_buffer_count(1);
        let cmdbuf = dev.allocate_command_buffers(&info).unwrap()[0];

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        dev.begin_command_buffer(cmdbuf, &info).unwrap();
        dev.cmd_copy_image(
            cmdbuf,
            src,
            vk::ImageLayout::GENERAL,
            dst,
            vk::ImageLayout::PREINITIALIZED,
            &[vk::ImageCopy {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 0,
                },
                src_offset: uvec3_to_offset3d(src_offset),
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 0,
                },
                dst_offset: uvec3_to_offset3d(dst_offset),
                extent: uvec3_to_extent3d(dimensions),
            }],
        );
        dev.end_command_buffer(cmdbuf).unwrap();

        let command_buffers = [cmdbuf];
        let submit_info = [vk::SubmitInfo::builder()
            .command_buffers(&command_buffers)
            .build()];
        dev.queue_submit(self.queues.graphics, &submit_info, vk::Fence::null())
            .unwrap();
        dev.queue_wait_idle(self.queues.graphics).unwrap();

        dev.free_command_buffers(self.pool, &[cmdbuf]);
    }

    pub unsafe fn transfer_to_image_with(
        &mut self,
        dev: &ash::Device,
        alloc: &vk_mem::Allocator,
        dst: vk::Image,
        dst_offset: UVec3,
        src_f: &mut dyn FnMut(&mut [u8]),
        dimensions: UVec3,
        format: vk::Format,
    ) {
        let elem_size = match format {
            vk::Format::R8G8B8A8_UNORM => 4,
            _ => todo!(),
        };
        let size =
            usize::try_from(U64Vec3::from(dimensions).element_product() * elem_size).unwrap();
        with_buffer(dev, alloc, size, &mut |dev, buf, ptr| {
            src_f(core::slice::from_raw_parts_mut(ptr, size));
            self.oneshot(dev, &mut |dev, cmdbuf| {
                dev.cmd_pipeline_barrier(
                    cmdbuf,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::builder()
                        .image(dst)
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            layer_count: 1,
                            level_count: 1,
                            base_array_layer: 0,
                        })
                        .build()],
                );
                dev.cmd_copy_buffer_to_image(
                    cmdbuf,
                    buf,
                    dst,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: uvec3_to_offset3d(dst_offset),
                        image_extent: uvec3_to_extent3d(dimensions),
                    }],
                );
                dev.cmd_pipeline_barrier(
                    cmdbuf,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::builder()
                        .image(dst)
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            layer_count: 1,
                            level_count: 1,
                            base_array_layer: 0,
                        })
                        .build()],
                );
            });
        });
    }

    pub unsafe fn drop_with(&mut self, alloc: &vk_mem::Allocator) {}
}

fn uvec3_to_extent3d(v: UVec3) -> vk::Extent3D {
    vk::Extent3D {
        width: v.x,
        height: v.y,
        depth: v.z,
    }
}

fn uvec3_to_offset3d(v: UVec3) -> vk::Offset3D {
    vk::Offset3D {
        x: v.x as i32,
        y: v.y as i32,
        z: v.z as i32,
    }
}

unsafe fn with_buffer(
    dev: &ash::Device,
    alloc: &vk_mem::Allocator,
    size: usize,
    f: &mut dyn FnMut(&ash::Device, vk::Buffer, *mut u8),
) {
    let mut buf = alloc
        .create_buffer(
            &vk::BufferCreateInfo::builder()
                .size(size as u64)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferHost,
                flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                    | vk_mem::AllocationCreateFlags::STRATEGY_MIN_TIME,
                required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
                ..Default::default()
            },
        )
        .unwrap();
    let ptr = alloc.map_memory(&mut buf.1).unwrap();
    f(dev, buf.0, ptr);
    alloc.unmap_memory(&mut buf.1);
    //alloc.flush_allocation(&buf.1, 0, size).unwrap();
    alloc.destroy_buffer(buf.0, &mut buf.1);
}
