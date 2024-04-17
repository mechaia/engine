#![feature(inline_const)]

pub mod query;
pub mod tree_map;

use glam::IVec3;
use std::collections::HashMap;

pub struct Map<T> {
    map: HashMap<IVec3, T>,
}

impl<T> Map<T> {
    pub fn new() -> Self {
        Self {
            map: Default::default(),
        }
    }

    pub fn insert(&mut self, key: IVec3, value: T) {
        self.map.insert(key, value);
    }

    pub fn remove(&mut self, key: IVec3) -> Option<T> {
        self.map.remove(&key)
    }

    pub fn get(&self, key: IVec3) -> Option<&T> {
        self.map.get(&key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (IVec3, &T)> {
        self.map.iter().map(|(k, v)| (*k, v))
    }

    pub fn keys(&self) -> impl Iterator<Item = IVec3> + '_ {
        self.map.keys().copied()
    }

    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.map.values()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
}
