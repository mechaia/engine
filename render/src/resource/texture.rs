use crate::{Render, VmaImage};
use ash::vk;
use glam::{UVec2, UVec3};
use vk_mem::Alloc;

use super::Shared;

pub struct Texture {
    image: VmaImage,
    format: vk::Format,
}

pub struct TextureView {
    texture: Shared<Texture>,
    image_view: vk::ImageView,
}

#[derive(Clone, Copy, Debug)]
pub enum TextureFormat {
    Rgba8Unorm,
    Gray8Unorm,
}

impl Texture {
    pub fn new(
        render: &mut Render,
        dimensions: UVec2,
        format: TextureFormat,
        reader: &mut dyn FnMut(&mut [u8]),
    ) -> Self {
        let format = match format {
            TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            TextureFormat::Gray8Unorm => vk::Format::R8_UNORM,
        };

        let image = unsafe {
            let info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: dimensions.x,
                    height: dimensions.y,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let c_info = vk_mem::AllocationCreateInfo {
                flags: vk_mem::AllocationCreateFlags::STRATEGY_MIN_MEMORY,
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };
            render.dev.alloc.create_image(&info, &c_info).unwrap()
        };

        unsafe {
            render.commands.transfer_to_image_with(
                &render.dev,
                image.0,
                UVec3::ZERO,
                reader,
                UVec3::new(dimensions.x, dimensions.y, 1),
                format,
            );
        }

        Self { image, format }
    }
}

impl TextureView {
    pub fn new(render: &mut Render, texture: Shared<Texture>) -> Self {
        let image_view = unsafe {
            let info = vk::ImageViewCreateInfo::builder()
                .image(texture.image.0)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(texture.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            render.dev.create_image_view(&info, None).unwrap()
        };
        Self {
            texture,
            image_view,
        }
    }

    pub fn bind_info(&self, sampler: vk::Sampler) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo {
            sampler,
            image_view: self.image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }
    }
}
