use ash::vk;
use core::mem;
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
        dev.destroy_buffer(buf.0, None);
        alloc.free_memory(&mut buf.1);
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

    pub unsafe fn drop_with(&mut self, alloc: &vk_mem::Allocator) {}
}
