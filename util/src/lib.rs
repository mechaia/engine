pub mod bit;
pub mod soa;

use core::{
    mem,
    num::NonZeroU32,
    ops::{Index, IndexMut},
};
use glam::{Quat, Vec3};

#[derive(Clone, Copy, Debug, Default)]
pub struct BitMap8(u8);

impl BitMap8 {
    pub fn get(&self, index: u8) -> bool {
        debug_assert!(index < 8);
        (self.0 >> index) & 1 != 0
    }

    pub fn set(&mut self, index: u8, value: bool) {
        debug_assert!(index < 8);
        self.0 &= !(1 << index);
        self.0 |= u8::from(value) << index;
    }
}

pub struct Arena<T> {
    buf: Vec<Option<T>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArenaHandle(NonZeroU32);

impl ArenaHandle {
    pub fn as_u32(&self) -> u32 {
        self.0.get() - 1
    }

    fn as_index(&self) -> usize {
        usize::try_from(self.0.get()).unwrap() - 1
    }
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self {
            buf: Default::default(),
        }
    }
}

impl<T> Index<ArenaHandle> for Arena<T> {
    type Output = T;

    fn index(&self, index: ArenaHandle) -> &Self::Output {
        self.buf[usize::try_from(index.0.get()).unwrap() - 1]
            .as_ref()
            .unwrap()
    }
}

impl<T> IndexMut<ArenaHandle> for Arena<T> {
    fn index_mut(&mut self, index: ArenaHandle) -> &mut Self::Output {
        self.buf[usize::try_from(index.0.get()).unwrap() - 1]
            .as_mut()
            .unwrap()
    }
}

impl<T> Arena<T> {
    pub fn insert(&mut self, value: T) -> ArenaHandle {
        if let Some(i) = self.buf.iter_mut().position(|e| e.is_none()) {
            self.buf[i] = Some(value);
            ArenaHandle(NonZeroU32::new((i + 1).try_into().unwrap()).unwrap())
        } else {
            self.buf.push(Some(value));
            ArenaHandle(NonZeroU32::new(self.buf.len().try_into().unwrap()).unwrap())
        }
    }

    pub fn remove(&mut self, handle: ArenaHandle) -> Option<T> {
        self.buf.get_mut(handle.as_index()).and_then(|v| v.take())
    }

    pub fn len(&self) -> usize {
        // FIXME O(n) lmao
        self.values().count()
    }

    pub fn values(&self) -> impl Iterator<Item = &T> + '_ {
        self.buf.iter().flat_map(|x| x.as_ref())
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> + '_ {
        self.buf.iter_mut().flat_map(|x| x.as_mut())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: f32,
}

pub struct ChunkedVec<T, const CHUNK_SIZE: usize = 64> {
    buf: Vec<Box<[mem::MaybeUninit<T>; CHUNK_SIZE]>>,
    len: usize,
}

impl<T, const CHUNK_SIZE: usize> ChunkedVec<T, CHUNK_SIZE> {
    pub fn push(&mut self) {
        todo!();
    }
}

impl<T, const CHUNK_SIZE: usize> Default for ChunkedVec<T, CHUNK_SIZE> {
    fn default() -> Self {
        Self {
            buf: Default::default(),
            len: 0,
        }
    }
}
