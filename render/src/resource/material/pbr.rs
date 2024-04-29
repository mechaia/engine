use crate::{Render, VmaBuffer};
use ash::vk;
use core::mem;
use glam::Vec4;
use vk_mem::Alloc;

pub struct PbrMaterialSet {
    buffer: VmaBuffer,
    len: u32,
    capacity: u32,
}

pub type TextureHandle = u32;

#[derive(Clone, Copy, Debug)]
pub struct PbrMaterial {
    pub albedo: Vec4,
    pub roughness: f32,
    pub metallic: f32,
    pub ambient_occlusion: f32,
    pub albedo_texture: TextureHandle,
    pub roughness_texture: TextureHandle,
    pub metallic_texture: TextureHandle,
    pub ambient_occlusion_texture: TextureHandle,
}

#[repr(C, align(16))]
struct PbrMaterialData {
    albedo: [f32; 4],
    roughness: f32,
    metallic: f32,
    ambient_occlusion: f32,
    _padding: f32,
    albedo_texture_index: u32,
    roughness_texture_id: u32,
    metallic_texture_id: u32,
    ambient_occlusion_texture_id: u32,
}

impl PbrMaterialSet {
    pub fn new(render: &mut Render, capacity: u32) -> Self {
        let buffer = unsafe {
            let info = vk::BufferCreateInfo::builder()
                .size(mem::size_of::<PbrMaterialData>() as u64 * u64::from(capacity))
                .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::UNIFORM_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let c_info = vk_mem::AllocationCreateInfo {
                flags: vk_mem::AllocationCreateFlags::STRATEGY_MIN_MEMORY,
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };
            render.dev.alloc.create_buffer(&info, &c_info).unwrap()
        };

        Self {
            buffer,
            len: 0,
            capacity,
        }
    }

    pub fn push(&mut self, render: &mut Render, material: &PbrMaterial) {
        assert!(self.len < self.capacity);

        let mat = PbrMaterialData {
            albedo: material.albedo.to_array(),
            roughness: material.roughness,
            metallic: material.metallic,
            ambient_occlusion: material.ambient_occlusion,
            _padding: 0.0,
            albedo_texture_index: material.albedo_texture,
            roughness_texture_id: material.roughness_texture,
            metallic_texture_id: material.metallic_texture,
            ambient_occlusion_texture_id: material.ambient_occlusion_texture,
        };

        unsafe {
            render.commands.transfer_to(
                &render.dev,
                self.buffer.0,
                u64::try_from(mem::size_of_val(&mat)).unwrap() * u64::from(self.len),
                (&mat as *const PbrMaterialData).cast(),
                mem::size_of_val(&mat),
            );
        }

        self.len += 1;
    }

    pub fn bind_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.buffer.0,
            offset: 0,
            range: vk::WHOLE_SIZE,
        }
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer.0
    }
}
