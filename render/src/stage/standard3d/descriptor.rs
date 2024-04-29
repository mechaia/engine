use crate::DropWith;

use super::Configuration;
use ash::vk;

pub(super) struct Descriptors {
    pub layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub sets: Box<[vk::DescriptorSet]>,
}

impl Descriptors {
    pub const BINDING_TEXTURES: u32 = 5;

    pub fn new(render: &mut crate::Render, config: &Configuration) -> Self {
        let layout = make_layout(&render.dev, config);
        let image_count = unsafe { render.swapchain.image_count() as u32 };
        let pool = make_pool(&render.dev, config, image_count);
        let sets = unsafe {
            let layouts = (0..image_count).map(|_| layout).collect::<Vec<_>>();
            let info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            render
                .dev
                .allocate_descriptor_sets(&info)
                .unwrap()
                .try_into()
                .unwrap()
        };
        Self { pool, sets, layout }
    }
}

unsafe impl DropWith for Descriptors {
    fn drop_with(self, dev: &mut crate::Dev) {
        unsafe {
            dev.destroy_descriptor_set_layout(self.layout, None);
            dev.destroy_descriptor_pool(self.pool, None);
        }
    }
}

fn make_layout(dev: &crate::Dev, config: &Configuration) -> vk::DescriptorSetLayout {
    let binding = core::cell::Cell::new(0);

    let bind = |ty, count, stage_flags| {
        let v = vk::DescriptorSetLayoutBinding::builder()
            .binding(binding.get())
            .descriptor_type(ty)
            .descriptor_count(count)
            .stage_flags(stage_flags);
        binding.set(binding.get() + 1);
        v
    };
    let comp = |ty, count| bind(ty, count, vk::ShaderStageFlags::COMPUTE);
    let comp_s = || comp(vk::DescriptorType::STORAGE_BUFFER, 1);

    let frag = |ty, count| bind(ty, count, vk::ShaderStageFlags::FRAGMENT);
    let frag_i = |tex_count| frag(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, tex_count);
    let frag_u = || frag(vk::DescriptorType::UNIFORM_BUFFER, 1);
    let frag_s = || frag(vk::DescriptorType::STORAGE_BUFFER, 1);

    let bindings = [
        // camera
        bind(
            vk::DescriptorType::UNIFORM_BUFFER,
            1,
            vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX,
        ),
        // transforms in
        comp_s(),
        // transforms out
        bind(
            vk::DescriptorType::STORAGE_BUFFER,
            1,
            vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX,
        ),
        // directional lights
        frag_s(),
        // materials
        frag_u(),
        // textures
        frag_i(config.max_texture_count),
    ]
    .map(|x| x.build());
    let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    unsafe { dev.create_descriptor_set_layout(&info, None).unwrap() }
}

fn make_pool(dev: &crate::Dev, config: &Configuration, image_count: u32) -> vk::DescriptorPool {
    let size = |ty, descriptor_count| vk::DescriptorPoolSize {
        ty,
        descriptor_count,
    };
    let sizes = [
        // camera
        size(vk::DescriptorType::UNIFORM_BUFFER, image_count),
        // transforms in
        size(vk::DescriptorType::STORAGE_BUFFER, image_count),
        // transforms out
        size(vk::DescriptorType::STORAGE_BUFFER, image_count),
        // directional lights
        size(vk::DescriptorType::STORAGE_BUFFER, image_count),
        // materials
        size(vk::DescriptorType::UNIFORM_BUFFER, image_count),
        // textures
        size(
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            image_count * config.max_texture_count,
        ),
    ];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&sizes)
        .max_sets(image_count);
    unsafe { dev.create_descriptor_pool(&info, None).unwrap() }
}
