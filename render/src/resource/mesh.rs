use crate::{Render, VmaBuffer};
use ash::vk;
use core::mem;
use glam::{Vec2, Vec3};
use vk_mem::Alloc;

/// Collection of meshes.
pub struct MeshSet {
    pub index_data: VmaBuffer,
    pub vertex_data: VmaBuffer,
    pub positions_offset: u64,
    pub normals_offset: u64,
    pub uvs_offset: u64,
    pub joints_offset: u64,
    pub weights_offset: u64,
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

pub struct Mesh<'a> {
    pub indices: &'a [u32],
    /// translation, normal, uv, joints, weights
    pub vertices: util::soa::Slice5<'a, Vec3, Vec3, Vec2, [u16; 4], [f32; 4]>,
}

unsafe fn alloc_buf(render: &mut Render, size: u64, as_index: bool) -> VmaBuffer {
    let usage = if as_index {
        vk::BufferUsageFlags::INDEX_BUFFER
    } else {
        vk::BufferUsageFlags::VERTEX_BUFFER
    };
    render
        .dev
        .alloc
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
    render: &mut Render,
    buf: vk::Buffer,
    offt: &mut u64,
    it: &mut dyn Iterator<Item = (*const u8, usize)>,
) {
    for (ptr, len) in it {
        render
            .commands
            .transfer_to(&render.dev, buf, *offt, ptr, len);
        *offt += len as u64;
    }
}

fn as_bytes<T>(slice: &[T]) -> (*const u8, usize) {
    (slice.as_ptr().cast(), slice.len() * mem::size_of::<T>())
}

impl MeshSet {
    pub fn new(render: &mut Render, meshes: &[Mesh<'_>]) -> MeshSet {
        let mut index_count @ mut vertex_count = 0;
        for m in meshes.iter() {
            index_count += u64::try_from(m.indices.len()).unwrap();
            vertex_count += u64::try_from(m.vertices.len()).unwrap();
        }

        let positions_len = vertex_count * 12;
        let normals_len = vertex_count * 12;
        let uvs_len = vertex_count * 8;
        let joints_len = vertex_count * 8;
        let weights_len = vertex_count * 16;

        let positions_offset = 0;
        let normals_offset = positions_offset + positions_len;
        let uvs_offset = normals_offset + normals_len;
        let joints_offset = uvs_offset + uvs_len;
        let weights_offset = joints_offset + joints_len;

        let index_data_size = index_count * 4;
        let vertex_data_size = positions_len + normals_len + uvs_len + joints_len + weights_len;

        unsafe {
            let index_data = alloc_buf(render, index_data_size, true);
            let vertex_data = alloc_buf(render, vertex_data_size, false);

            let it = || meshes.iter();
            merge(
                render,
                index_data.0,
                &mut 0,
                &mut it().map(|m| as_bytes(&m.indices)),
            );
            let mut offt = 0;
            let mut v_merge = |it: &mut dyn Iterator<Item = _>| {
                merge(render, vertex_data.0, &mut offt, it);
            };
            v_merge(&mut it().map(|m| as_bytes(&m.vertices.as_slices().0)));
            v_merge(&mut it().map(|m| as_bytes(&m.vertices.as_slices().1)));
            v_merge(&mut it().map(|m| as_bytes(&m.vertices.as_slices().2)));
            v_merge(&mut it().map(|m| as_bytes(&m.vertices.as_slices().3)));
            v_merge(&mut it().map(|m| as_bytes(&m.vertices.as_slices().4)));

            render.dev.device_wait_idle().unwrap();
            MeshSet {
                index_data,
                vertex_data,
                positions_offset,
                normals_offset,
                uvs_offset,
                joints_offset,
                weights_offset,
                mesh_index_offsets: [MeshOffsets::default()]
                    .into_iter()
                    .chain(meshes.iter().scan(MeshOffsets::default(), |o, m| {
                        o.vertex += u32::try_from(m.vertices.len()).unwrap();
                        o.index += u32::try_from(m.indices.len()).unwrap();
                        Some(*o)
                    }))
                    .collect(),
            }
        }
    }

    pub fn len(&self) -> usize {
        self.mesh_index_offsets.len() - 1
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
}

unsafe impl crate::DropWith for MeshSet {
    fn drop_with(mut self, dev: &mut crate::Dev) {
        unsafe {
            dev.alloc.free_memory(&mut self.index_data.1);
            dev.alloc.free_memory(&mut self.vertex_data.1);
        }
    }
}
