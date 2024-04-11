use ash::vk;

pub struct DrawInfo {
    pub index: usize,
    pub available: vk::Semaphore,
    pub finished: vk::Semaphore,
    pub may_draw: vk::Fence,
}

struct DrawOne {
    available: vk::Semaphore,
    finish: vk::Semaphore,
    may_begin: vk::Fence,
}

pub struct Draw {
    current_image: usize,
    sync: Box<[DrawOne]>,
}

impl Draw {
    pub fn new(dev: &ash::Device, count: usize) -> Self {
        let mut sync = Vec::with_capacity(count);
        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        for _ in 0..count {
            let available = unsafe { dev.create_semaphore(&semaphore_info, None).unwrap() };
            let finish = unsafe { dev.create_semaphore(&semaphore_info, None).unwrap() };
            let may_begin = unsafe { dev.create_fence(&fence_info, None).unwrap() };
            sync.push(DrawOne {
                available,
                finish,
                may_begin,
            });
        }
        Self {
            current_image: 0,
            sync: sync.into(),
        }
    }

    pub unsafe fn drop_with(&mut self, dev: &ash::Device) {
        for one in self.sync.iter() {
            dev.destroy_semaphore(one.available, None);
            dev.destroy_semaphore(one.finish, None);
            dev.destroy_fence(one.may_begin, None);
        }
    }
}

pub struct SwapChain {
    loader: ash::extensions::khr::Swapchain,
    khr: vk::SwapchainKHR,
    surface_capabilities: vk::SurfaceCapabilitiesKHR,
    surface_format: vk::SurfaceFormatKHR,
}

impl SwapChain {
    pub fn new(
        physical_device: vk::PhysicalDevice,
        dev: &ash::Device,
        surface_loader: &ash::extensions::khr::Surface,
        surface: vk::SurfaceKHR,
        surface_format: vk::SurfaceFormatKHR,
        graphics_queue_index: u32,
        instance: &ash::Instance,
        image_count: u32,
    ) -> Self {
        // Swapchain
        let surface_capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap()
        };
        let extent = surface_capabilities.current_extent;
        let format = surface_format.format;

        let queuefamilies = [graphics_queue_index];
        let create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(surface_capabilities.current_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queuefamilies)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO);
        let loader = ash::extensions::khr::Swapchain::new(instance, dev);
        let khr = unsafe { loader.create_swapchain(&create_info, None).unwrap() };

        Self {
            loader,
            khr,
            surface_capabilities,
            surface_format,
        }
    }

    pub unsafe fn drop_with(&mut self, dev: &ash::Device) {
        self.loader.destroy_swapchain(self.khr, None);
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.surface_capabilities.current_extent
    }

    pub fn format(&self) -> vk::Format {
        self.surface_format.format
    }

    pub unsafe fn draw(
        &mut self,
        dev: &ash::Device,
        draw: &mut Draw,
        f: impl FnOnce(DrawInfo) -> vk::Queue,
    ) -> bool {
        let cur = &draw.sync[draw.current_image];
        draw.current_image += 1;
        draw.current_image %= draw.sync.len();

        dev.wait_for_fences(&[cur.may_begin], true, super::TIMEOUT)
            .unwrap();
        dev.reset_fences(&[cur.may_begin]).unwrap();

        let (image_index, _) = self
            .loader
            .acquire_next_image(self.khr, super::TIMEOUT, cur.available, vk::Fence::null())
            .unwrap();

        let queue = f(DrawInfo {
            index: image_index as usize,
            available: cur.available,
            finished: cur.finish,
            may_draw: cur.may_begin,
        });

        let sem_finish = [cur.finish];
        let swapchains = [self.khr];
        let indices = [image_index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&sem_finish)
            .swapchains(&swapchains)
            .image_indices(&indices);
        let suboptimal = self.loader.queue_present(queue, &present_info).unwrap();

        suboptimal
    }

    pub unsafe fn image_count(&self) -> usize {
        // FIXME avoid allocation
        self.loader.get_swapchain_images(self.khr).unwrap().len()
    }

    pub unsafe fn images(&self) -> Vec<vk::Image> {
        self.loader.get_swapchain_images(self.khr).unwrap()
    }
}
