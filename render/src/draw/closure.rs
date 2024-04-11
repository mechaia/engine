use crate::{mesh::MeshCollection, Dev, VmaBuffer};
use ash::vk;
use core::mem;
use glam::{Vec2, Vec3};
use vk_mem::Alloc;

pub struct DrawClosure {
    descriptor_pool: vk::DescriptorPool,
    command: Box<[DrawCommand]>,
    max_instances: u32,
}

struct DrawCommand {
    command: vk::CommandBuffer,
    graphics: DrawCommandGraphics,
}

struct DrawCommandGraphics {}

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
