/*!
 * This renderer supports not only cubes but also:
 *
 * - wedges
 * - TODO: corners ("tetras")
 * - TODO: inner corners ("tetra" but opposite)
 *
 * Other shapes might get added at some point.
 *
 * # How it works
 *
 * Rendering is split in a compute and graphics stage:
 *
 * - A compute shader takes a grid, determines which block faces are occluded and emits compact vertex data.
 * - The vertex shader converts vertex data to translation, normal, UV... and transforms to view space.
 *   It takes material data from a spatial grid.
 * - The fragment shader is a standard PBR shader, with materials and multiple lights.
 *
 * There is a CPU-side stage which manages the data on the GPU and performs basic culling.
 *
 * TODO: actually use a compute shader.
 *
 * # Issues and limitations.
 *
 * There is no way to directly "stream" data from a compute shader to a vertex shader,
 * so a large intermediate buffer is required.
 * This can be worked around by splitting up the work in pieces, but without GPU-side dispatch
 * involves more CPU work.
 *
 * Coordinates are limited to the range [0;2**12[ to keep per-vertex data within 64 bits,
 * i.e. about 2000 voxels can be visible at any time in each direction.
 *
 * TODO: consider using mesh shaders. Old cards do not support it but newer cards will benefit substantially.
 */

use glam::{Quat, U64Vec3, UVec3, Vec3};
use render::stage::renderpass::RenderPassBuilder;
use util::{
    bit::{BitBox, BitVec},
    Transform,
};

mod compute;
mod graphics;

pub struct VoxelRender {}

enum Axis {
    X,
    Y,
    Z,
}

enum Direction {
    Pos,
    Neg,
}

enum Shape {
    Block,
    //Wedge,
    //Tetra,
    //InnerTetra,
    //Penta,
    //InnerPenta,
}

pub struct Voxel {
    pub shape: Shape,
}

impl VoxelRender {
    pub fn new(
        render_pass: &mut RenderPassBuilder,
        max_grids: u32,
        voxel_buffer_size: u64,
    ) -> Self {
        todo!()
    }

    /// Reset grids list
    pub fn reset(&mut self, index: usize) {}

    /// Push a grid to render
    pub fn push<F>(
        &mut self,
        index: usize,
        transform: &Transform,
        size: UVec3,
        edges_occluded: bool,
        query: F,
    ) where
        F: Fn(UVec3) -> Option<Voxel>,
    {
        // TODO this is obviously inefficient, though it might be efficient if we'd push it directly
        // to the GPU instead
        let len = U64Vec3::from(size);
        let len = len.z * len.y * len.x;
        let mut map = BitVec::with_capacity(len.try_into().unwrap());

        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    match query(UVec3::new(x, y, z)) {
                        None => map.push(false),
                        Some(_) => map.push(true),
                    }
                }
            }
        }

        let test = todo!();
    }
}
