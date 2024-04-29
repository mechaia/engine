use super::{Stage, StageArgs};
use crate::{Dev, VmaImage};
use ash::vk::{self, AttachmentReference};
use core::mem;
use vk_mem::Alloc;

pub unsafe trait RenderSubpass {
    unsafe fn record_commands(&self, dev: &crate::Dev, args: &StageArgs);

    unsafe fn rebuild_swapchain(&mut self, dev: &mut Dev, swapchain: &crate::swapchain::SwapChain);
}

pub unsafe trait RenderSubpassBuilder {
    unsafe fn build(
        self: Box<Self>,
        dev: &crate::Dev,
        render_pass: vk::RenderPass,
        subpass: u32,
    ) -> Box<dyn RenderSubpass>;
}

pub struct RenderPass {
    render_pass: vk::RenderPass,
    clear_values: Box<[vk::ClearValue]>,
    framebuffers: Box<[RenderPassFramebuffer]>,
    subpasses: Box<[Box<dyn RenderSubpass>]>,
}

pub struct RenderPassBuilder {
    attachments: Vec<vk::AttachmentDescription>,
    subpasses: Vec<SubpassAttachmentReferences>,
    dependencies: Vec<vk::SubpassDependency>,
    subpass_builders: Vec<Box<dyn RenderSubpassBuilder>>,
}

pub struct SubpassAttachmentReferences {
    pub color: Box<[vk::AttachmentReference]>,
    pub depth_stencil: Option<vk::AttachmentReference>,
}

struct RenderPassFramebuffer {
    framebuffer: vk::Framebuffer,
    // some images, such as those from get_swapchain_images, cannot
    // be freed by us.
    // Use separate lists for image and imageviews to avoid accidents.
    image_views: Box<[vk::ImageView]>,
    images: Box<[VmaImage]>,
}

impl RenderPass {
    pub fn builder(render: &mut crate::Render) -> RenderPassBuilder {
        RenderPassBuilder {
            attachments: Vec::new(),
            subpasses: Vec::new(),
            dependencies: Vec::new(),
            subpass_builders: Vec::new(),
        }
    }

    pub fn render_pass(&self) -> vk::RenderPass {
        self.render_pass
    }
}

impl RenderPassBuilder {
    pub fn push<T: RenderSubpassBuilder + 'static>(
        &mut self,
        subpass_builder: T,
        subpass: SubpassAttachmentReferences,
    ) -> u32 {
        self.subpass_builders.push(Box::new(subpass_builder));
        self.subpasses.push(subpass);
        u32::try_from(self.subpasses.len() - 1).unwrap()
    }

    pub unsafe fn add_attachment(&mut self, attachment: vk::AttachmentDescription) {
        self.attachments.push(attachment)
    }

    pub unsafe fn add_dependency(&mut self, dependency: vk::SubpassDependency) {
        self.dependencies.push(dependency);
    }

    pub fn build(self, render: &mut crate::Render) -> RenderPass {
        assert_eq!(self.subpasses.len(), self.subpass_builders.len());

        unsafe {
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0; 4] },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ]
            .into();

            let render_pass = {
                let subpasses = self
                    .subpasses
                    .iter()
                    .map(|v| {
                        let mut b = vk::SubpassDescription::builder()
                            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                            .color_attachments(&v.color);
                        if let Some(a) = v.depth_stencil.as_ref() {
                            b = b.depth_stencil_attachment(a);
                        }
                        b.build()
                    })
                    .collect::<Vec<_>>();
                let info = vk::RenderPassCreateInfo::builder()
                    .attachments(&self.attachments)
                    .subpasses(&subpasses)
                    .dependencies(&self.dependencies);
                render.dev.create_render_pass(&info, None).unwrap()
            };

            let subpasses = self
                .subpass_builders
                .into_iter()
                .enumerate()
                .map(|(i, s)| s.build(&render.dev, render_pass, i.try_into().unwrap()))
                .collect();

            let framebuffers = render
                .swapchain
                .images()
                .into_iter()
                .map(|img| {
                    make_framebuffer(
                        &render.dev,
                        render_pass,
                        img,
                        render.swapchain.format(),
                        render.swapchain.extent(),
                    )
                })
                .collect();
            RenderPass {
                clear_values,
                render_pass,
                framebuffers,
                subpasses,
            }
        }
    }
}

unsafe impl Stage for RenderPass {
    unsafe fn record_commands(&self, dev: &crate::Dev, args: &StageArgs) {
        let fb = &self.framebuffers[args.index];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(fb.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: args.viewport,
            })
            .clear_values(&self.clear_values);
        dev.cmd_begin_render_pass(args.cmd, &info, vk::SubpassContents::INLINE);
        for (k, subpass) in self.subpasses.iter().enumerate() {
            if k > 0 {
                dev.cmd_next_subpass(args.cmd, vk::SubpassContents::INLINE);
            }
            subpass.record_commands(dev, args);
        }
        dev.cmd_end_render_pass(args.cmd);
    }

    unsafe fn rebuild_swapchain(&mut self, dev: &mut Dev, swapchain: &crate::swapchain::SwapChain) {
        for fb in mem::take(&mut self.framebuffers).into_vec() {
            fb.drop_with(dev);
        }
        self.framebuffers = swapchain
            .images()
            .into_iter()
            .map(|img| {
                make_framebuffer(
                    dev,
                    self.render_pass,
                    img,
                    swapchain.format(),
                    swapchain.extent(),
                )
            })
            .collect();
        for subpass in self.subpasses.iter_mut() {
            subpass.rebuild_swapchain(dev, swapchain);
        }
    }
}

impl RenderPassFramebuffer {
    unsafe fn drop_with(mut self, dev: &Dev) {
        dev.destroy_framebuffer(self.framebuffer, None);
        for iv in mem::take(&mut self.image_views).into_vec() {
            dev.destroy_image_view(iv, None);
        }
        for (img, mut mem) in mem::take(&mut self.images).into_vec() {
            dev.alloc.destroy_image(img, &mut mem);
        }
    }
}

unsafe fn make_framebuffer(
    dev: &Dev,
    render_pass: vk::RenderPass,
    out: vk::Image,
    format: vk::Format,
    viewport: vk::Extent2D,
) -> RenderPassFramebuffer {
    let mut image_views = Vec::with_capacity(1 + 1);
    let mut images = Vec::with_capacity(1);

    // display
    image_views.push({
        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let info = vk::ImageViewCreateInfo::builder()
            .image(out)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(*subresource_range);
        dev.create_image_view(&info, None).unwrap()
    });

    // depth
    image_views.push({
        let info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: viewport.width,
                height: viewport.height,
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
        let (depth_image, depth_alloc) = dev
            .alloc
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

    let info = vk::FramebufferCreateInfo::builder()
        .render_pass(render_pass)
        .attachments(&image_views)
        .width(viewport.width)
        .height(viewport.height)
        .layers(1);
    let framebuffer = unsafe { dev.create_framebuffer(&info, None).unwrap() };

    RenderPassFramebuffer {
        framebuffer,
        images: images.into(),
        image_views: image_views.into(),
    }
}
