use ash::vk;
use core::{mem, ptr::NonNull};
use glam::{Mat4, Vec2};
use vk_mem::Alloc;

#[repr(C)]
pub struct CameraData {
    matrix: [f32; 16],
    inv_matrix: [f32; 16],
}

pub struct Camera {
    buffers: Vec<(vk::Buffer, vk_mem::Allocation, NonNull<CameraData>)>,
}

impl Camera {
    pub unsafe fn new(alloc: &vk_mem::Allocator, swapchain_count: usize) -> Self {
        let mut buffers = Vec::with_capacity(swapchain_count);
        for _ in 0..swapchain_count {
            let (buf, mut mem) = alloc
                .create_buffer_with_alignment(
                    &vk::BufferCreateInfo::builder()
                        .size(mem::size_of::<CameraData>() as u64)
                        .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    &vk_mem::AllocationCreateInfo {
                        usage: vk_mem::MemoryUsage::Auto,
                        flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                        required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE,
                        ..Default::default()
                    },
                    4,
                )
                .unwrap();
            let ptr = alloc.map_memory(&mut mem).unwrap();
            buffers.push((buf, mem, NonNull::new_unchecked(ptr.cast())));
        }
        Self { buffers }
    }

    pub fn set(&mut self, index: usize, camera: &super::Camera) {
        let t = Mat4::from_translation(-camera.translation);
        let r = Mat4::from_quat(camera.rotation);
        let x = Mat4::from_cols_array_2d(&[
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let p = Mat4::perspective_rh(camera.fov, camera.aspect, camera.near, camera.far);
        let m = p * x * r * t;

        unsafe {
            self.buffers[index].2.as_ptr().write(CameraData {
                matrix: m.to_cols_array(),
                inv_matrix: m.inverse().to_cols_array(),
            });
        };
    }

    pub unsafe fn bind(
        &mut self,
        dev: &ash::Device,
        uniform_descriptor_sets: &[vk::DescriptorSet],
    ) {
        for (buf, &set) in self.buffers.iter().zip(uniform_descriptor_sets) {
            let info = [vk::DescriptorBufferInfo::builder()
                .buffer(buf.0)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build()];
            let mut write = vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&info);
            // waiting for https://github.com/ash-rs/ash/pull/809
            write.descriptor_count = 1;
            dev.update_descriptor_sets(&[write.build()], &[]);
        }
    }

    pub fn buffer(&self, index: usize) -> vk::Buffer {
        self.buffers[index].0
    }

    pub unsafe fn drop_with(&mut self, alloc: &vk_mem::Allocator) {
        for (buf, mut mem, _) in self.buffers.drain(..) {
            alloc.unmap_memory(&mut mem);
            alloc.destroy_buffer(buf, &mut mem);
        }
    }
}
