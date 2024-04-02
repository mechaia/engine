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

    pub unsafe fn transfer_between(
        &mut self,
        dev: &ash::Device,
        dst: vk::Buffer,
        dst_offset: u64,
        src: vk::Buffer,
        src_offset: u64,
        amount: usize,
    ) {
        let info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.pool)
            .command_buffer_count(1);
        let cmdbuf = dev.allocate_command_buffers(&info).unwrap()[0];

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        dev.begin_command_buffer(cmdbuf, &info).unwrap();
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

    pub unsafe fn transfer_to(
        &mut self,
        dev: &ash::Device,
        alloc: &vk_mem::Allocator,
        dst: vk::Buffer,
        dst_offset: u64,
        src: *const u8,
        amount: usize,
    ) {
        let mut buf = alloc
            .create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(amount as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::AutoPreferHost,
                    flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                        | vk_mem::AllocationCreateFlags::STRATEGY_MIN_TIME,
                    ..Default::default()
                },
            )
            .unwrap();
        alloc
            .map_memory(&mut buf.1)
            .unwrap()
            .copy_from_nonoverlapping(src, amount);
        alloc.flush_allocation(&buf.1, 0, amount).unwrap();
        alloc.unmap_memory(&mut buf.1);
        self.transfer_between(dev, dst, dst_offset, buf.0, 0, amount);
        alloc.destroy_buffer(buf.0, &mut buf.1);
    }

    #[allow(unused)]
    pub unsafe fn transfer_from(
        &mut self,
        dev: &ash::Device,
        alloc: &vk_mem::Allocator,
        dst: *mut u8,
        src: vk::Buffer,
        src_offset: u64,
        amount: usize,
    ) {
        todo!()
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

    pub unsafe fn transfer_to_image(
        &mut self,
        dev: &ash::Device,
        alloc: &vk_mem::Allocator,
        dst: vk::Image,
        dst_offset: UVec3,
        src: *const u8,
        dimensions: UVec3,
    ) {
        let mut img = alloc
            .create_image(
                &vk::ImageCreateInfo::builder()
                    // FIXME don't hardcode image properties
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UINT)
                    // TODO do we need mip_levels and array_layers?
                    .mip_levels(1)
                    .array_layers(1)
                    .extent(uvec3_to_extent3d(dimensions))
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::LINEAR)
                    .usage(vk::ImageUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .queue_family_indices(&[self.queues.graphics_index])
                    .initial_layout(vk::ImageLayout::PREINITIALIZED),
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::AutoPreferHost,
                    flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                        | vk_mem::AllocationCreateFlags::STRATEGY_MIN_TIME,
                    ..Default::default()
                },
            )
            .unwrap();
        let amount = usize::try_from(U64Vec3::from(dimensions).element_product()).unwrap();
        alloc
            .map_memory(&mut img.1)
            .unwrap()
            .copy_from_nonoverlapping(src, amount);
        alloc.flush_allocation(&img.1, 0, amount).unwrap();
        alloc.unmap_memory(&mut img.1);
        self.transfer_between_images(dev, dst, dst_offset, img.0, UVec3::ZERO, dimensions);
        alloc.destroy_image(img.0, &mut img.1);
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
