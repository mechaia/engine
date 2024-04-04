use crate::{VmaBuffer, Vulkan};
use ash::vk;
use core::mem;
use vk_mem::Alloc;

/// Build a collection of meshes.
///
/// Once built it cannot be modified.
pub struct MeshCollectionBuilder<'a> {
    pub meshes: &'a [Mesh],
}

unsafe fn alloc_buf(vulkan: &mut Vulkan, size: u64, as_index: bool) -> VmaBuffer {
    let usage = if as_index {
        vk::BufferUsageFlags::INDEX_BUFFER
    } else {
        vk::BufferUsageFlags::VERTEX_BUFFER
    };
    vulkan
        .allocator
        .create_buffer_with_alignment(
            &vk::BufferCreateInfo::builder()
                .size(size)
                .usage(usage | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                flags: vk_mem::AllocationCreateFlags::STRATEGY_MIN_MEMORY,
                preferred_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                ..Default::default()
            },
            //4,
            64,
        )
        .unwrap()
}

unsafe fn merge(
    vulkan: &mut Vulkan,
    buf: vk::Buffer,
    offt: &mut u64,
    it: &mut dyn Iterator<Item = (*const u8, usize)>,
) {
    for (ptr, len) in it {
        vulkan
            .commands
            .transfer_to(&vulkan.dev, &vulkan.allocator, buf, *offt, ptr, len);
        *offt += len as u64;
    }
}

fn as_bytes<T>(slice: &[T]) -> (*const u8, usize) {
    (slice.as_ptr().cast(), slice.len() * mem::size_of::<T>())
}

impl MeshCollectionBuilder<'_> {
    pub fn finish(self, vulkan: &mut Vulkan) -> MeshCollection {
        let mut index_count @ mut vertex_count = 0;
        let positions_offset @ mut normals_offset @ mut uvs_offset = 0;
        for m in self.meshes.iter() {
            index_count += m.indices.len() as u64;
            vertex_count += u64::from(m.vertex_count());
            normals_offset += m.positions.len() as u64;
            uvs_offset += m.positions.len() as u64;
        }
        uvs_offset += normals_offset;
        normals_offset *= 12;
        uvs_offset *= 12;

        let index_data_size = index_count * 4;
        let vertex_data_size = vertex_count * (3 + 3 + 2) * 4;

        unsafe {
            let index_data = alloc_buf(vulkan, index_data_size, true);
            let vertex_data = alloc_buf(vulkan, vertex_data_size, false);

            let it = || self.meshes.iter();
            merge(
                vulkan,
                index_data.0,
                &mut 0,
                &mut it().map(|m| as_bytes(&m.indices)),
            );
            let mut offt = 0;
            let mut v_merge = |it: &mut dyn Iterator<Item = _>| {
                merge(vulkan, vertex_data.0, &mut offt, it);
            };
            v_merge(&mut it().map(|m| as_bytes(&m.positions)));
            v_merge(&mut it().map(|m| as_bytes(&m.normals)));
            v_merge(&mut it().map(|m| as_bytes(&m.uvs)));

            vulkan.dev.device_wait_idle().unwrap();
            MeshCollection {
                index_data,
                vertex_data,
                positions_offset,
                normals_offset,
                uvs_offset,
                mesh_index_offsets: [MeshOffsets::default()]
                    .into_iter()
                    .chain(self.meshes.iter().scan(MeshOffsets::default(), |o, m| {
                        o.vertex += m.vertex_count();
                        o.index += m.indices.len() as u32;
                        Some(*o)
                    }))
                    .collect(),
            }
        }
    }
}

/// Collection of meshes.
pub struct MeshCollection {
    pub index_data: VmaBuffer,
    pub vertex_data: VmaBuffer,
    pub positions_offset: u64,
    pub normals_offset: u64,
    pub uvs_offset: u64,
    mesh_index_offsets: Box<[MeshOffsets]>,
}

#[derive(Clone, Copy, Default)]
struct MeshOffsets {
    vertex: u32,
    index: u32,
}

pub struct MeshView {
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub index_count: u32,
}

impl MeshCollection {
    pub fn len(&self) -> usize {
        self.mesh_index_offsets.len()
    }

    pub fn mesh(&self, index: usize) -> MeshView {
        assert!(index < self.mesh_index_offsets.len() - 1);
        let offsets = self.mesh_index_offsets[index];
        let index_count = self.mesh_index_offsets[index + 1].index - offsets.index;
        MeshView {
            vertex_offset: offsets.vertex,
            index_offset: offsets.index,
            index_count,
        }
    }

    pub unsafe fn drop_with(mut self, alloc: &vk_mem::Allocator) {
        alloc.free_memory(&mut self.index_data.1);
        alloc.free_memory(&mut self.vertex_data.1);
    }
}

/// A single mesh.
#[derive(Debug)]
pub struct Mesh {
    indices: Vec<u32>,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
}

impl Mesh {
    pub fn from_glb_slice(data: &[u8]) -> Vec<Self> {
        let glb = gltf::Glb::from_slice(data).unwrap();
        let gltf = gltf::Gltf::from_slice(&glb.json).unwrap();
        let bin = &glb.bin.as_deref().unwrap();
        let mut meshes = vec![];
        for mesh in gltf.meshes() {
            for p in mesh.primitives() {
                let r = p.reader(|buffer| {
                    assert!(matches!(buffer.source(), gltf::buffer::Source::Bin));
                    Some(bin)
                });
                let indices = r.read_indices().unwrap().into_u32().collect();
                let positions = r.read_positions().unwrap().collect::<Vec<_>>();
                let normals = r.read_normals().unwrap().collect();
                let uvs = r.read_tex_coords(0).map_or_else(
                    || core::iter::repeat([0.0; 2]).take(positions.len()).collect(),
                    |t| t.into_f32().collect(),
                );
                meshes.push(Self {
                    indices,
                    positions,
                    normals,
                    uvs,
                })
            }
        }
        for m in meshes.iter() {
            assert_eq!(m.positions.len(), m.normals.len());
            assert_eq!(m.positions.len(), m.uvs.len());
        }
        meshes
    }

    fn vertex_count(&self) -> u32 {
        self.positions.len() as u32
    }
}
