use ash::vk;

pub struct DescriptorManager {
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    descriptor_pool: vk::DescriptorPool,
    uniform_descriptor_sets: Vec<vk::DescriptorSet>,
    storage_descriptor_set: vk::DescriptorSet,
}
