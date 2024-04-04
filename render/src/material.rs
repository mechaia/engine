//! # References
//!
//! https://learnopengl.com/PBR/Theory

use ash::vk;
use glam::{U16Vec2, U64Vec2, UVec2, UVec3};
use vk_mem::Alloc;

use crate::{VmaImage, Vulkan};

pub struct PbrMaterialCollection {
    /// all textures are put into an atlas to reduce descriptor usage
    pub textures: TextureAtlas,
    pub views: Box<[PbrMaterialView]>,
}

#[repr(C, align(16))]
pub(crate) struct PbrMaterialView {
    pub albedo: [f32; 3],
    pub albedo_texture_index: u32,
    pub roughness: f32,
    pub roughness_texture_id: u32,
    pub metallic: f32,
    pub metallic_texture_id: u32,
    pub ambient_occlusion: f32,
    pub ambient_occlusion_texture_id: u32,
}

pub type PbrMaterial = PbrMaterialView;

pub(crate) struct TextureAtlas {
    texture: VmaImage,
    mapping: Box<[U16Vec2]>,
}

impl TextureAtlas {
    pub unsafe fn pack_textures(vulkan: &mut Vulkan, textures: &mut [Texture]) -> Self {
        // sort large to small
        // TODO indirect to preserve the order of the original textures
        let f = |t: &Texture| U64Vec2::from(t.dim).element_product();
        textures.sort_unstable_by(|x, y| f(x).cmp(&f(y)).reverse());

        // TODO try smaller sizes
        let size_p2 = 12;
        let mapping = pack_textures(size_p2, textures);
        let texture = create_texture_atlas(vulkan, size_p2, textures, &mapping);
        Self { texture, mapping }
    }
}

/// Pack textures tightly
fn pack_textures(size_p2: u8, textures: &[Texture]) -> Box<[U16Vec2]> {
    // scan until a fitting place is found.
    //
    // Use a quadtree to save memory.
    // This has good memory efficiency while allowing fast queries in both dimensions.
    //
    // To make it work, make a jagged array of rows.
    // When inserting a new row, give it the same length as all other rows.
    // When inserting a new column, grow all rows by +1.
    //
    // Each element is an index in the textures array, or u32::MAX if empty.
    let mut filled = quad_bitmap::QuadBitmap::new(size_p2);

    let mut mapping = (0..textures.len())
        .map(|_| U16Vec2::MAX)
        .collect::<Box<_>>();

    for (tex_id, tex) in textures.iter().enumerate() {
        // search
        let mut pos = UVec2::ZERO;
        'search: while pos.y < filled.size() {
            pos.x = 0;
            while let Some(x) = filled.next_row_gap(pos.y, pos.x) {
                pos.x = x;
                if !filled.any(pos, pos + tex.dim) {
                    break 'search;
                }
            }
            // TODO check height of previous blocks to skip faster
            pos.y += 1;
            assert!(pos.y < filled.size(), "atlas creation failed");
        }

        // insert
        filled.fill(pos, pos + tex.dim);
        mapping[tex_id] = U16Vec2::try_from(pos).unwrap();
    }

    mapping
}

unsafe fn create_texture_atlas(
    vulkan: &mut Vulkan,
    size_p2: u8,
    textures: &[Texture],
    mapping: &[U16Vec2],
) -> VmaImage {
    let dim = 1 << size_p2;

    let img = vulkan
        .allocator
        .create_image(
            &vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UINT)
                .mip_levels(1)
                .array_layers(1)
                .extent(vk::Extent3D {
                    width: dim,
                    height: dim,
                    depth: 1,
                })
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .queue_family_indices(&[vulkan.commands.queues.graphics_index])
                .initial_layout(vk::ImageLayout::UNDEFINED),
            &vk_mem::AllocationCreateInfo {
                flags: vk_mem::AllocationCreateFlags::STRATEGY_MIN_MEMORY,
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            },
        )
        .unwrap();

    for (tex, pos) in textures.iter().zip(mapping) {
        let data = tex.decode_to_vec();
        assert_eq!(data.len(), usize::from(pos.x) * usize::from(pos.y));
        let f = |v: UVec2| UVec3::new(v.x.into(), v.y.into(), 0);
        vulkan.commands.transfer_to_image_with(
            &vulkan.dev,
            &vulkan.allocator,
            img.0,
            f(UVec2::from(*pos)),
            &mut |s| s.copy_from_slice(&data),
            f(tex.dim),
            todo!(),
        );
    }

    img
}

mod quad_bitmap {
    use glam::UVec2;

    pub struct QuadBitmap {
        size: u32,
        root: Node,
    }

    impl QuadBitmap {
        pub fn new(size_p2: u8) -> Self {
            Self {
                size: 1 << size_p2,
                root: Node::Empty,
            }
        }

        pub fn size(&self) -> u32 {
            self.size
        }

        pub fn any(&self, start: UVec2, end: UVec2) -> bool {
            if start.x >= end.x || start.y >= end.y {
                return false;
            }
            self.root.any(start, end, self.size)
        }

        pub fn fill(&mut self, start: UVec2, end: UVec2) {
            self.root.fill(start, end, self.size);
        }

        pub fn next_row_gap(&self, y: u32, start_x: u32) -> Option<u32> {
            self.root.next_row_gap(y, start_x, self.size)
        }
    }

    enum Node {
        Empty,
        Full,
        /// ```
        ///     |
        ///   0 | 1
        /// ----+---> x
        ///   2 | 3
        ///     v
        ///     y
        /// ```
        Partial(Box<[Node; 4]>),
    }

    fn any_node(nodes: &[Node; 4], start: UVec2, end: UVec2, dim: u32) -> bool {
        // try to avoid recursing
        if nodes.iter().any(|e| matches!(e, Node::Full)) {
            return true;
        }
        if nodes.iter().all(|e| matches!(e, Node::Empty)) {
            return false;
        }
        let d = dim >> 1;
        for (i, e) in nodes.iter().enumerate() {
            let Node::Partial(l) = e else { continue };
            if any_node(l, start + i_to_uvec(i) * d, end - i_to_uvec(!i) * d, d) {
                return true;
            }
        }
        false
    }

    impl Node {
        fn any(&self, start: UVec2, end: UVec2, dim: u32) -> bool {
            let d = dim >> 1;
            match self {
                Node::Empty => false,
                Node::Full => true,
                Node::Partial(l) if start.x < d && start.y < d && end.x > d && end.y > d => true,
                Node::Partial(l) => {
                    let d = dim >> 1;

                    if start.y < d {
                        if start.x < d && l[0].any(start, end, d) {
                            return true;
                        }
                        let v = UVec2::new(d, 0);
                        if end.x >= d && l[1].any(start + v, end - v, d) {
                            return true;
                        }
                    }

                    if end.y >= d {
                        let v = UVec2::new(0, d);
                        if start.x < d && l[2].any(start + v, end - v, d) {
                            return true;
                        }
                        let v = UVec2::new(d, d);
                        if end.x >= d && l[3].any(start + v, end - d, d) {
                            return true;
                        }
                    }
                    false
                }
            }
        }

        fn next_row_gap(&self, y: u32, start_x: u32, dim: u32) -> Option<u32> {
            let d = dim >> 1;
            match self {
                Self::Empty => Some(0),
                Self::Full => None,
                Self::Partial(l) => {
                    let (i, yd) = if y < dim { (0, y) } else { (2, y - d) };
                    l[i].next_row_gap(y - d, start_x, d)
                        .or_else(|| l[i + 1].next_row_gap(y - d, start_x - d, d).map(|x| x + d))
                }
            }
        }

        fn fill(&mut self, start: UVec2, end: UVec2, dim: u32) {
            if matches!(self, Self::Full) || dim <= start.x && dim <= start.y {
                return;
            }
            if start == UVec2::ZERO && end.x <= dim && end.y <= dim {
                *self = Self::Full;
                return;
            }
            if matches!(self, Self::Empty) {
                const E: Node = Node::Empty;
                *self = Self::Partial(Box::new([E; 4]));
            }
            let Self::Partial(l) = self else {
                unreachable!()
            };
            let d = dim >> 1;
            for (i, e) in l.iter_mut().enumerate() {
                e.fill(start + i_to_uvec(i) * d, end - i_to_uvec(!i) * d, d);
            }
            if l.iter().all(|e| matches!(e, Self::Full)) {
                *self = Self::Full;
            }
        }
    }

    fn i_to_uvec(i: usize) -> UVec2 {
        let i = i as u32;
        UVec2::new(i & 1, (i >> 1) & 1)
    }
}

pub struct Texture {
    dim: UVec2,
    ty: TextureType,
    data: Vec<u8>,
}

enum TextureType {
    Raw,
    Png,
    Jpeg,
}

impl Texture {
    pub fn from_array_2d<const W: usize, const H: usize>(array: [[u32; W]; H]) -> Self {
        Self {
            dim: UVec2::new(W as u32, H as u32),
            ty: TextureType::Raw,
            data: array
                .iter()
                .flat_map(|r| r)
                .flat_map(|p| p.to_le_bytes())
                .collect(),
        }
    }

    fn decode_to_vec(&self) -> Vec<u8> {
        match &self.ty {
            TextureType::Raw => self.data.clone(),
            TextureType::Png => todo!(),
            TextureType::Jpeg => todo!(),
        }
    }
}
