pub struct Compute {}

unsafe impl render::Stage for Compute {
    unsafe fn record_commands(&self, dev: &render::Dev, args: &render::StageArgs) {
        /* TODO */
	}

    unsafe fn rebuild_swapchain(
        &mut self,
        dev: &mut render::Dev,
        swapchain: &render::SwapChain,
    ) {
    }
}
