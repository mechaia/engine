struct GraphicsPass {
}

unsafe impl render::stage::renderpass::RenderSubpass for GraphicsPass {
    unsafe fn record_commands(&self, dev: &ash::Device, args: &render::StageArgs) {
        
    }

    unsafe fn rebuild_swapchain(&mut self, dev: &mut render::Dev, swapchain: &render::SwapChain) {
        
    }
}
