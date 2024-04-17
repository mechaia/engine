use std::collections::HashMap;

use glam::{IVec3, U16Vec3};

const SIDE: usize = 16;

enum Chunk<T> {
    Grid(Box<[[[Option<T>; SIDE]; SIDE]; SIDE]>),
}

pub struct Map<T> {
    chunks: HashMap<ChunkKey, Chunk<T>>,
}

impl<T> Map<T> {
    pub fn new() -> Self {
        Self {
            chunks: Default::default(),
        }
    }

    pub fn insert(&mut self, key: IVec3, value: T) {
        let (ck, bk) = split_key(key);
        let chunk = self.chunks.entry(ck).or_insert_with(|| {
            Chunk::Grid([const { [const { [const { None }; 16] }; 16] }; 16].into())
        });
        match chunk {
            Chunk::Grid(v) => v[bk.z][bk.y][bk.x] = Some(value),
        }
    }

    pub fn remove(&mut self, key: IVec3) -> Option<T> {
        let (ck, bk) = split_key(key);
        let chunk = self.chunks.get_mut(&ck)?;
        match chunk {
            Chunk::Grid(v) => v[bk.z][bk.y][bk.x].take(),
        }
    }

    pub fn get(&self, key: IVec3) -> Option<&T> {
        let (ck, bk) = split_key(key);
        let chunk = self.chunks.get(&ck)?;
        match chunk {
            Chunk::Grid(v) => v[bk.z][bk.y][bk.x].as_ref(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (IVec3, &T)> {
        self.chunks.iter().flat_map(|(&k, c)| match c {
            Chunk::Grid(v) => v
                .iter()
                .enumerate()
                .flat_map(move |(z, v)| {
                    v.iter().enumerate().flat_map(move |(y, v)| {
                        v.iter()
                            .enumerate()
                            .map(move |(x, v)| (merge_key(k, UsizeVec3 { x, y, z }), v))
                    })
                })
                .flat_map(|(k, v)| v.as_ref().map(move |v| (k, v))),
        })
    }

    pub fn keys(&self) -> impl Iterator<Item = IVec3> + '_ {
        self.iter().map(|v| v.0)
    }

    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.iter().map(|v| v.1)
    }
}

/// To help prevent errors
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ChunkKey(IVec3);

struct UsizeVec3 {
    x: usize,
    y: usize,
    z: usize,
}

impl From<U16Vec3> for UsizeVec3 {
    fn from(v: U16Vec3) -> Self {
        Self {
            x: v.x.into(),
            y: v.y.into(),
            z: v.z.into(),
        }
    }
}

fn split_key(key: IVec3) -> (ChunkKey, UsizeVec3) {
    let s = IVec3::splat(i32::try_from(SIDE).unwrap());
    (
        ChunkKey(key.div_euclid(s)),
        U16Vec3::try_from(key.rem_euclid(s)).unwrap().into(),
    )
}

fn merge_key(chunk_key: ChunkKey, sub_key: UsizeVec3) -> IVec3 {
    debug_assert!(sub_key.x < 16);
    debug_assert!(sub_key.y < 16);
    debug_assert!(sub_key.z < 16);
    let sub_key = IVec3::new(sub_key.x as _, sub_key.y as _, sub_key.z as _);
    (chunk_key.0 * i32::try_from(SIDE).unwrap()) + sub_key
}
