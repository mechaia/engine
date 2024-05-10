#![feature(ptr_metadata, layout_for_ptr, wrapping_next_power_of_two)]

pub mod bit;
pub mod math;
pub mod soa;
pub mod sync;

pub use num_complex::Complex32 as Complex;
pub use rand;

use core::{
    mem,
    num::NonZeroU32,
    ops::{Index, IndexMut},
};
use glam::{Quat, Vec3};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TransformScale {
    pub translation: Vec3,
    pub scale: f32,
    pub rotation: Quat,
}

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

    fn from_index(index: usize) -> Self {
        ArenaHandle(NonZeroU32::new(u32::try_from(index + 1).unwrap()).unwrap())
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
            ArenaHandle::from_index(i)
        } else {
            self.buf.push(Some(value));
            ArenaHandle::from_index(self.buf.len() - 1)
        }
    }

    pub fn remove(&mut self, handle: ArenaHandle) -> Option<T> {
        self.buf.get_mut(handle.as_index()).and_then(|v| v.take())
    }

    pub fn len(&self) -> usize {
        // FIXME O(n) lmao
        self.values().count()
    }

    pub fn iter(&self) -> impl Iterator<Item = (ArenaHandle, &T)> + '_ {
        self.buf
            .iter()
            .enumerate()
            .flat_map(|(i, x)| x.as_ref().map(move |x| (ArenaHandle::from_index(i), x)))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (ArenaHandle, &mut T)> + '_ {
        self.buf
            .iter_mut()
            .enumerate()
            .flat_map(|(i, x)| x.as_mut().map(move |x| (ArenaHandle::from_index(i), x)))
    }

    pub fn keys(&self) -> impl Iterator<Item = ArenaHandle> + '_ {
        self.iter().map(|(i, _)| i)
    }

    pub fn values(&self) -> impl Iterator<Item = &T> + '_ {
        self.iter().map(|(_, x)| x)
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> + '_ {
        self.iter_mut().map(|(_, x)| x)
    }
}

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
    };

    pub fn new(translation: Vec3, rotation: Quat) -> Self {
        Self {
            translation,
            rotation,
        }
    }

    pub fn apply_to_translation(&self, translation: Vec3) -> Vec3 {
        self.translation + (self.rotation * translation)
    }

    pub fn apply_to_translation_inv(&self, translation: Vec3) -> Vec3 {
        self.rotation.inverse() * (translation - self.translation)
    }

    pub fn apply_to_direction(&self, direction: Vec3) -> Vec3 {
        self.rotation * direction
    }

    pub fn apply_to_direction_inv(&self, direction: Vec3) -> Vec3 {
        self.rotation.inverse() * direction
    }

    pub fn apply_to_rotation(&self, rotation: Quat) -> Quat {
        self.rotation * rotation
    }

    pub fn apply_to_rotation_inv(&self, rotation: Quat) -> Quat {
        self.rotation.inverse() * rotation
    }

    pub fn apply_to_transform(&self, transform: &Self) -> Self {
        Self {
            translation: self.apply_to_translation(transform.translation),
            rotation: self.apply_to_rotation(transform.rotation),
        }
    }

    pub fn apply_to_transform_inv(&self, transform: &Self) -> Self {
        Self {
            translation: self.apply_to_translation_inv(transform.translation),
            rotation: self.apply_to_rotation_inv(transform.rotation),
        }
    }

    pub fn interpolate(&self, to: &Self, s: f32) -> Self {
        Self {
            translation: self.translation.lerp(to.translation, s),
            rotation: self.rotation.slerp(to.rotation, s),
        }
    }

    pub fn inverse(&self) -> Self {
        let rotation = self.rotation.inverse();
        let translation = rotation * -self.translation;
        Self {
            rotation,
            translation,
        }
    }

    pub fn lerp(&self, to: &Self, t: f32) -> Self {
        Self {
            rotation: self.rotation.lerp(to.rotation, t),
            translation: self.translation.lerp(to.translation, t),
        }
    }

    pub fn from_rotation_x(angle: f32) -> Self {
        Self::IDENTITY.with_rotation(Quat::from_rotation_x(angle))
    }

    pub fn from_rotation_y(angle: f32) -> Self {
        Self::IDENTITY.with_rotation(Quat::from_rotation_y(angle))
    }

    pub fn from_rotation_z(angle: f32) -> Self {
        Self::IDENTITY.with_rotation(Quat::from_rotation_z(angle))
    }

    pub fn with_translation(&self, translation: Vec3) -> Self {
        Self {
            translation,
            ..*self
        }
    }

    pub fn with_rotation(&self, rotation: Quat) -> Self {
        Self { rotation, ..*self }
    }

    pub fn with_scale(&self, scale: f32) -> TransformScale {
        TransformScale::from(*self).with_scale(scale)
    }
}

impl TransformScale {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        scale: 1.0,
        rotation: Quat::IDENTITY,
    };

    pub fn with_translation(&self, translation: Vec3) -> Self {
        Self {
            translation,
            ..*self
        }
    }

    pub fn with_rotation(&self, rotation: Quat) -> Self {
        Self { rotation, ..*self }
    }

    pub fn with_scale(&self, scale: f32) -> Self {
        Self { scale, ..*self }
    }

    pub fn as_transform(&self) -> Transform {
        Transform {
            translation: self.translation,
            rotation: self.rotation,
        }
    }
}

impl From<Transform> for TransformScale {
    fn from(value: Transform) -> Self {
        Self {
            translation: value.translation,
            rotation: value.rotation,
            scale: 1.0,
        }
    }
}

pub struct ChunkedVec<T, const CHUNK_SIZE: usize = 64> {
    buf: Vec<Box<[mem::MaybeUninit<T>; CHUNK_SIZE]>>,
    len: usize,
}

impl<T, const CHUNK_SIZE: usize> ChunkedVec<T, CHUNK_SIZE> {
    pub fn push(&mut self, value: T) {
        let (i, k) = (self.len / CHUNK_SIZE, self.len % CHUNK_SIZE);
        if k % CHUNK_SIZE == 0 {
            self.new_chunk();
        }
        self.buf[i][k].write(value);
        self.len += 1;
        todo!();
    }

    fn new_chunk(&mut self) {
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
