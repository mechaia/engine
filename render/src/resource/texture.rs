use crate::{Render, VmaImage};
use ash::vk;
use glam::{UVec2, UVec3};
use vk_mem::Alloc;

pub struct TextureSet {
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    set: vk::DescriptorSet,
    sampler: vk::Sampler,
    images: Box<[(VmaImage, vk::ImageView)]>,
    format: TextureFormat,
}

pub struct TextureSetBuilder<'a> {
    render: &'a mut Render,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    set: vk::DescriptorSet,
    sampler: vk::Sampler,
    images: Vec<(VmaImage, vk::ImageView)>,
    image_count: usize,
    format: TextureFormat,
}

#[derive(Clone, Copy, Debug)]
pub enum TextureFormat {
    Rgba8Unorm,
    Gray8Unorm,
}

impl TextureSet {
    pub fn builder(
        render: &mut Render,
        format: TextureFormat,
        texture_count: usize,
    ) -> TextureSetBuilder {
        let pool = render.make_descriptor_pool(
            1,
            &[vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: texture_count.try_into().unwrap(),
            }],
        );
        let layout = unsafe {
            let bindings = [vk::DescriptorSetLayoutBinding::builder()
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(texture_count.try_into().unwrap())
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build()];
            let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
            render
                .dev
                .create_descriptor_set_layout(&info, None)
                .unwrap()
        };
        let set = unsafe {
            let layouts = [layout];
            let info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            render.dev.allocate_descriptor_sets(&info).unwrap()[0]
        };
        let sampler = unsafe {
            let info = vk::SamplerCreateInfo::builder()
                /*
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                */
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .mip_lod_bias(0.0)
                .anisotropy_enable(false)
                .compare_enable(false)
                .unnormalized_coordinates(false);
            render.dev.create_sampler(&info, None).unwrap()
        };

        TextureSetBuilder {
            render,
            pool,
            layout,
            set,
            sampler,
            images: Vec::with_capacity(texture_count),
            image_count: texture_count,
            format,
        }
    }

    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }

    pub fn set(&self) -> vk::DescriptorSet {
        self.set
    }
}

impl<'a> TextureSetBuilder<'a> {
    pub fn push(mut self, dimensions: UVec2, reader: &mut dyn FnMut(&mut [u8])) -> Self {
        assert!(self.images.len() < self.image_count);

        let fmt = match self.format {
            TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            TextureFormat::Gray8Unorm => vk::Format::R8_UNORM,
        };

        let img = unsafe {
            let info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(fmt)
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
            self.render.dev.alloc.create_image(&info, &c_info).unwrap()
        };

        unsafe {
            self.render.commands.transfer_to_image_with(
                &self.render.dev,
                img.0,
                UVec3::ZERO,
                reader,
                UVec3::new(dimensions.x, dimensions.y, 1),
                fmt,
            );
        }

        let view = unsafe {
            let info = vk::ImageViewCreateInfo::builder()
                .image(img.0)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(fmt)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            self.render.dev.create_image_view(&info, None).unwrap()
        };

        unsafe {
            let info = [vk::DescriptorImageInfo {
                sampler: self.sampler,
                image_view: view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }];
            let writes = [vk::WriteDescriptorSet::builder()
                .dst_set(self.set)
                .dst_binding(0)
                .dst_array_element(self.images.len().try_into().unwrap())
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&info)
                .build()];
            self.render.dev.update_descriptor_sets(&writes, &[]);
        }

        self.images.push((img, view));
        self
    }

    pub fn build(self) -> TextureSet {
        assert_eq!(self.image_count, self.images.len());
        TextureSet {
            pool: self.pool,
            layout: self.layout,
            set: self.set,
            sampler: self.sampler,
            images: self.images.into(),
            format: self.format,
        }
    }
}
