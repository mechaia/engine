use ash::vk;

pub struct Queues {
    pub graphics_index: u32,
    pub graphics: vk::Queue,
}

fn create_device(
    instance: &ash::Instance,
    physdev: vk::PhysicalDevice,
    queue_infos: &[vk::DeviceQueueCreateInfo],
) -> ash::Device {
    let device_extension_names = [
        ash::extensions::khr::Swapchain::name().as_ptr(),
        vk::KhrShaderClockFn::name().as_ptr(),
    ];
    let mut vk12 = vk::PhysicalDeviceVulkan12Features::builder()
        .draw_indirect_count(true)
        .runtime_descriptor_array(true)
        .shader_sampled_image_array_non_uniform_indexing(true);
    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(queue_infos)
        .enabled_extension_names(&device_extension_names)
        .push_next(&mut vk12);
    unsafe {
        instance
            .create_device(physdev, &device_create_info, None)
            .unwrap()
    }
}

impl Queues {
    pub fn new(instance: &ash::Instance, physdev: vk::PhysicalDevice) -> (ash::Device, Self) {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physdev) };
        let graphics_index = {
            let mut found_graphics_q_index = None;
            for (index, qfam) in queue_family_properties.iter().enumerate() {
                if qfam.queue_count > 0 && qfam.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    found_graphics_q_index = Some(index as u32);
                }
            }
            found_graphics_q_index.unwrap()
        };
        let priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(graphics_index)
            .queue_priorities(&priorities)
            .build()];

        let dev = create_device(instance, physdev, &queue_infos);

        let queues = Self {
            graphics_index,
            graphics: unsafe { dev.get_device_queue(graphics_index, 0) },
        };
        (dev, queues)
    }
}
