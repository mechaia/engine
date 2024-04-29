pub mod camera;
pub mod material;
pub mod mesh;
pub mod texture;

use core::ops::Deref;
use std::sync::Arc;

pub struct Shared<T>(Arc<T>);

impl<T> Shared<T> {
    pub fn new(value: T) -> Self {
        Self(Arc::new(value))
    }
}

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> Deref for Shared<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

unsafe impl<T: crate::DropWith> crate::DropWith for Shared<T> {
    unsafe fn drop_with(self, dev: &crate::Dev) {
        let Ok(v) = Arc::try_unwrap(self.0) else {
            return;
        };
        unsafe { v.drop_with(dev) };
    }
}
