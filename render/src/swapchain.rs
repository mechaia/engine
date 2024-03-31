use ash::vk;
use vk_mem::Alloc;

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
    image_views: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,
    images: Vec<super::VmaImage>,
}

impl SwapChain {
    pub fn new(
        physical_device: vk::PhysicalDevice,
        dev: &ash::Device,
        alloc: &vk_mem::Allocator,
        surface_loader: &ash::extensions::khr::Surface,
        surface: vk::SurfaceKHR,
        surface_format: vk::SurfaceFormatKHR,
        graphics_queue_index: u32,
        instance: &ash::Instance,
        render_pass: vk::RenderPass,
        image_count: usize,
    ) -> (Self, vk::Extent2D) {
        // Swapchain
        let surface_capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap()
        };

        let queuefamilies = [graphics_queue_index];
        let create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .min_image_count(image_count as u32)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(surface_capabilities.current_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queuefamilies)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::IMMEDIATE);
        //.present_mode(vk::PresentModeKHR::MAILBOX);
        //.present_mode(vk::PresentModeKHR::FIFO);
        let loader = ash::extensions::khr::Swapchain::new(&instance, &dev);
        let khr = unsafe { loader.create_swapchain(&create_info, None).unwrap() };

        let mut image_views = Vec::with_capacity(image_count * 2);
        let mut images = Vec::with_capacity(image_count);

        // Swapchain images (buffers)
        let display_images = unsafe { loader.get_swapchain_images(khr).unwrap() };
        assert_eq!(display_images.len(), image_count);
        for &display_image in &display_images {
            // display
            image_views.push(unsafe {
                let subresource_range = vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);
                let info = vk::ImageViewCreateInfo::builder()
                    .image(display_image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .subresource_range(*subresource_range);
                dev.create_image_view(&info, None).unwrap()
            });

            // depth
            image_views.push(unsafe {
                let info = vk::ImageCreateInfo::builder()
                    .image_type(vk::ImageType::TYPE_2D)
                    .extent(vk::Extent3D {
                        width: surface_capabilities.current_extent.width,
                        height: surface_capabilities.current_extent.height,
                        depth: 1,
                    })
                    .array_layers(1)
                    .mip_levels(1)
                    .format(vk::Format::D32_SFLOAT)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE);
                let (depth_image, depth_alloc) = alloc
                    .create_image(
                        &info,
                        &vk_mem::AllocationCreateInfo {
                            flags: vk_mem::AllocationCreateFlags::STRATEGY_MIN_MEMORY,
                            usage: vk_mem::MemoryUsage::AutoPreferDevice,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                images.push((depth_image, depth_alloc));

                let subresource_range = vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);
                let info = vk::ImageViewCreateInfo::builder()
                    .image(depth_image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::D32_SFLOAT)
                    .subresource_range(*subresource_range);
                dev.create_image_view(&info, None).unwrap()
            });
        }

        // Framebuffers
        let mut framebuffers = Vec::with_capacity(image_count);
        for iviews in image_views.chunks(2) {
            let info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&iviews)
                .width(surface_capabilities.current_extent.width)
                .height(surface_capabilities.current_extent.height)
                .layers(1);
            let fb = unsafe { dev.create_framebuffer(&info, None).unwrap() };
            framebuffers.push(fb);
        }

        let sc = SwapChain {
            loader,
            khr,
            image_views,
            framebuffers,
            images,
        };
        (sc, surface_capabilities.current_extent)
    }

    pub unsafe fn drop_with(&mut self, dev: &ash::Device, alloc: &vk_mem::Allocator) {
        for fb in self.framebuffers.drain(..) {
            dev.destroy_framebuffer(fb, None);
        }
        for iv in self.image_views.drain(..) {
            dev.destroy_image_view(iv, None);
        }
        for (buf, mut mem) in self.images.drain(..) {
            alloc.destroy_image(buf, &mut mem);
        }
        self.loader.destroy_swapchain(self.khr, None);
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
}
