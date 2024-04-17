use ash::vk;
use core::{mem, ptr::NonNull};
use glam::Mat4;
use vk_mem::Alloc;

pub struct CameraView {
    pub translation: glam::Vec3,
    pub rotation: glam::Quat,
    pub projection: CameraProjection,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

pub enum CameraProjection {
    Perspective { fov: f32 },
    Orthographic { scale: f32 },
}

#[repr(C)]
struct CameraData {
    world_to_view: [f32; 16],
    view_to_project: [f32; 16],
}

pub struct Camera {
    buffers: Vec<(crate::VmaBuffer, NonNull<CameraData>)>,
}

impl Camera {
    pub fn new(render: &mut crate::Render) -> Self {
        let mut s = Self {
            buffers: Vec::new(),
        };
        unsafe { s.rebuild_swapchain(&mut render.dev, &render.swapchain) };
        s
    }

    pub fn set(&mut self, index: usize, camera: &CameraView) -> (Mat4, Mat4) {
        let t = Mat4::from_translation(-camera.translation);
        let r = Mat4::from_quat(camera.rotation);
        let x = Mat4::from_cols_array_2d(&[
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        //let x = Mat4::IDENTITY;
        let p = match &camera.projection {
            &CameraProjection::Perspective { fov } => {
                Mat4::perspective_rh(fov, camera.aspect, camera.near, camera.far)
            }
            &CameraProjection::Orthographic { scale } => {
                let x = scale;
                let y = x * camera.aspect;
                Mat4::orthographic_rh(-x, x, -y, y, camera.near, camera.far)
            }
        };

        let world_to_view = x * r * t;
        let view_to_project = p;

        unsafe {
            self.buffers[index].1.as_ptr().write(CameraData {
                world_to_view: world_to_view.to_cols_array(),
                // TODO decompose so we can eliminate 0 factors in the shader
                view_to_project: view_to_project.to_cols_array(),
            });
        };

        (world_to_view, view_to_project)
    }

    pub unsafe fn bind(
        &mut self,
        dev: &ash::Device,
        uniform_descriptor_sets: &[vk::DescriptorSet],
    ) {
        for (buf, &set) in self.buffers.iter().zip(uniform_descriptor_sets) {
            let info = [vk::DescriptorBufferInfo::builder()
                .buffer(buf.0 .0)
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

    pub(crate) fn buffer(&self, index: usize) -> vk::Buffer {
        self.buffers[index].0 .0
    }

    pub unsafe fn drop_with(self, dev: &mut crate::Dev) {
        for (mut buf, _) in self.buffers {
            dev.unmap_buffer(&mut buf);
            dev.free_buffer(buf);
        }
    }

    pub unsafe fn rebuild_swapchain(&mut self, dev: &mut crate::Dev, swapchain: &crate::SwapChain) {
        for _ in self.buffers.len()..swapchain.image_count() {
            let mut buf = dev.allocate_buffer(
                mem::size_of::<CameraData>() as _,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                true,
            );
            let ptr = dev.map_buffer(&mut buf);
            self.buffers.push((buf, ptr.cast()));
        }
        for (mut buf, _) in self.buffers.drain(swapchain.image_count()..) {
            dev.unmap_buffer(&mut buf);
            dev.free_buffer(buf);
        }
    }
}
