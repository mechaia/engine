use crate::{Render, VmaBuffer};
use ash::vk;
use core::mem;
use glam::Vec4;
use vk_mem::Alloc;

pub struct PbrMaterialSet {
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    set: vk::DescriptorSet,
    buffer: VmaBuffer,
}

pub struct PbrMaterialSetBuilder<'a> {
    render: &'a mut Render,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    set: vk::DescriptorSet,
    buffer: VmaBuffer,
    material_index: u32,
    material_count: u32,
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
    pub fn builder(render: &mut Render, count: u32) -> PbrMaterialSetBuilder {
        let pool = render.make_descriptor_pool(
            1,
            &[vk::DescriptorPoolSize {
                //ty: vk::DescriptorType::STORAGE_BUFFER,
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            }],
        );
        let layout = unsafe {
            let bindings = [vk::DescriptorSetLayoutBinding::builder()
                //.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)]
            .map(|x| x.build());
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
        let buffer = unsafe {
            let info = vk::BufferCreateInfo::builder()
                .size(mem::size_of::<PbrMaterialData>() as u64 * u64::from(count))
                //.usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER)
                .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::UNIFORM_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let c_info = vk_mem::AllocationCreateInfo {
                flags: vk_mem::AllocationCreateFlags::STRATEGY_MIN_MEMORY,
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };
            render.dev.alloc.create_buffer(&info, &c_info).unwrap()
        };

        PbrMaterialSetBuilder {
            render,
            pool,
            layout,
            set,
            buffer,
            material_index: 0,
            material_count: count,
        }
    }

    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }

    pub fn set(&self) -> vk::DescriptorSet {
        self.set
    }
}

impl<'a> PbrMaterialSetBuilder<'a> {
    pub fn push(mut self, material: &PbrMaterial) -> Self {
        assert!(self.material_index < self.material_count);

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
            self.render.commands.transfer_to(
                &self.render.dev,
                self.buffer.0,
                u64::try_from(mem::size_of_val(&mat)).unwrap() * u64::from(self.material_index),
                (&mat as *const PbrMaterialData).cast(),
                mem::size_of_val(&mat),
            );
        }

        self.material_index += 1;
        self
    }

    pub fn build(self) -> PbrMaterialSet {
        assert_eq!(self.material_index, self.material_count);

        unsafe {
            let info = [vk::DescriptorBufferInfo {
                buffer: self.buffer.0,
                offset: 0,
                range: vk::WHOLE_SIZE,
            }];
            let writes = [vk::WriteDescriptorSet::builder()
                .dst_set(self.set)
                .dst_binding(0)
                .dst_array_element(0)
                //.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&info)]
            .map(|x| x.build());
            self.render.dev.update_descriptor_sets(&writes, &[]);
        }

        PbrMaterialSet {
            pool: self.pool,
            layout: self.layout,
            set: self.set,
            buffer: self.buffer,
        }
    }
}
