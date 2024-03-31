//! # References
//!
//! https://learnopengl.com/PBR/Theory

pub struct PbrMaterialCollection {
    pub textures: Box<[VmaBuffer]>,
    pub views: Box<[PbrMaterialView]>,
}

#[repr(C)]
pub(crate) struct PbrMaterialView {
    pub color: [f32; 4],
    pub uv_offset: [f32; 2],
    pub uv_size: [f32; 2],
    pub roughness: TextureOrValue<f32>,
    pub metallicness: TextureOrValue<f32>,
    pub texture_index: u32,
}

pub(crate) enum TextureOrValue<T> {
    Texture(u32),
    Value(T),
}

pub struct PbrMaterial {
}
