pub mod renderpass;
pub mod standard3d;

use ash::vk;

pub struct StageArgs {
    pub cmd: vk::CommandBuffer,
    pub index: usize,
    pub viewport: vk::Extent2D,
}

pub unsafe trait Stage {
    unsafe fn record_commands(&self, dev: &ash::Device, args: &StageArgs);

    unsafe fn rebuild_swapchain(
        &mut self,
        dev: &mut crate::Dev,
        swapchain: &crate::swapchain::SwapChain,
    );
}
