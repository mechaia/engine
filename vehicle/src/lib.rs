mod block;

pub use block::{Block, BlockDB, Connections};
use glam::{UVec3, Vec3};

use std::{collections::HashMap, num::NonZeroU16};

type Map<K, V> = HashMap<K, V>;

#[derive(PartialEq, Eq, Hash)]
struct Pos4(u16);

impl Pos4 {
    fn new(x: u8, y: u8, z: u8) -> Self {
        Self(u16::from(x) << 8 | u16::from(y) << 4 | u16::from(z))
    }

    fn x(&self) -> u8 {
        ((self.0 >> 8) & 15) as u8
    }

    fn y(&self) -> u8 {
        ((self.0 >> 4) & 15) as u8
    }

    fn z(&self) -> u8 {
        ((self.0 >> 0) & 15) as u8
    }
}

enum Chunk<V> {
    Sparse(Map<Pos4, V>),
}

impl<V> Default for Chunk<V> {
    fn default() -> Self {
        Self::Sparse(Default::default())
    }
}

type ChunkMap<V> = Map<Pos4, Chunk<V>>;

type BlockId = NonZeroU16;
type BlockHealth = NonZeroU16;

pub struct Meta {
    id: BlockId,
    bitmap: util::BitMap8,
}

impl Meta {
    fn new(id: BlockId, mirror_x: bool, mirror_y: bool, mirror_z: bool) -> Self {
        let mut bitmap = util::BitMap8::default();
        bitmap.set(0, mirror_x);
        bitmap.set(1, mirror_y);
        bitmap.set(2, mirror_z);
        Self { id, bitmap }
    }

    fn mirror_x(&self) -> bool {
        self.bitmap.get(0)
    }

    fn mirror_y(&self) -> bool {
        self.bitmap.get(1)
    }

    fn mirror_z(&self) -> bool {
        self.bitmap.get(2)
    }
}

pub struct BlockMap<V> {
    chunks: Map<UVec3, Chunk<V>>,
}

impl<V> BlockMap<V> {
    pub fn insert(&mut self, position: UVec3, value: V) {
        let (pos_h, pos_l) = self.split_pos(position);
        match self.chunks.entry(pos_h).or_default() {
            Chunk::Sparse(v) => {
                v.insert(pos_l, value);
            }
        }
    }

    pub fn get(&self, position: UVec3) -> Option<&V> {
        let (pos_h, pos_l) = self.split_pos(position);
        match self.chunks.get(&pos_h)? {
            Chunk::Sparse(v) => v.get(&pos_l),
        }
    }

    pub fn cast_ray(&mut self, start: Vec3, direction: Vec3) -> Option<(f32, &V)> {
        self.step_ray(start, direction).next()
    }

    pub fn step_ray(
        &mut self,
        start: Vec3,
        direction: Vec3,
    ) -> impl Iterator<Item = (f32, &V)> + '_ {
        todo!();
        [].into_iter()
    }

    pub fn step_sphere(&mut self, origin: Vec3) -> impl Iterator<Item = (f32, &V)> + '_ {
        todo!();
        [].into_iter()
    }

    pub fn step_connected(&mut self) -> impl Iterator<Item = &V> + '_ {
        todo!();
        [].into_iter()
    }

    fn split_pos(&self, position: UVec3) -> (UVec3, Pos4) {
        let pos_h = position / 16;
        let pos_l = position % 16;
        let pos_l = Pos4::new(pos_l.x as u8, pos_l.y as u8, pos_l.z as u8);
        (pos_h, pos_l)
    }
}
