use core::{
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

pub struct BitSlice<'a> {
    ptr: NonNull<u8>,
    len: usize,
    _marker: PhantomData<&'a u8>,
}

impl<'a> BitSlice<'a> {
    pub fn get(&self, index: usize) -> Option<bool> {
        if index >= self.len {
            return None;
        }
        unsafe { Some(self.get_unchecked(index)) }
    }

    pub fn set(&self, index: usize, value: bool) {
        assert!(index < self.len, "out of bounds");
        unsafe { self.set_unchecked(index, value) }
    }

    unsafe fn get_unchecked(&self, index: usize) -> bool {
        (*self.ptr.as_ptr().add(index / 8) >> (index % 8)) & 1 != 0
    }

    unsafe fn set_unchecked(&self, index: usize, value: bool) {
        let p = &mut *self.ptr.as_ptr().add(index / 8);
        let k = index % 8;
        *p &= !(1 << k);
        *p |= u8::from(value) << k;
    }
}

pub struct BitBox {
    data: BitSlice<'static>,
}

impl BitBox {
    pub fn filled(len: usize, value: bool) -> Self {
        let val = u8::from(value) * u8::MAX;
        let buf: Box<[u8]> = (0..(len + 7) / 8).map(|_| val).collect();
        let ptr = Box::into_raw(buf);
        unsafe {
            Self {
                data: BitSlice {
                    ptr: NonNull::new_unchecked((&mut *ptr).as_mut_ptr()),
                    len,
                    _marker: PhantomData,
                },
            }
        }
    }
}

impl Drop for BitBox {
    fn drop(&mut self) {
        let _ =
            unsafe { Vec::from_raw_parts(self.data.ptr.as_ptr(), self.data.len, self.data.len) };
    }
}

impl Deref for BitBox {
    type Target = BitSlice<'static>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for BitBox {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

pub struct BitVec {
    data: BitSlice<'static>,
    capacity: usize,
}

impl BitVec {
    pub fn with_capacity(capacity: usize) -> Self {
        Self::from_vec(Vec::with_capacity((capacity + 7) / 8), 0)
    }

    pub fn filled(len: usize, value: bool) -> Self {
        let val = u8::from(value) * u8::MAX;
        Self::from_vec((0..(len + 7) / 8).map(|_| val).collect(), len)
    }

    pub fn push(&mut self, value: bool) {
        if self.len >= self.capacity {
            self.grow(self.len + 1);
        }
        unsafe { self.set_unchecked(self.len, value) }
        self.len += 1;
    }

    fn grow(&mut self, min_cap: usize) {
        todo!();
    }

    fn from_vec(mut vec: Vec<u8>, real_len: usize) -> Self {
        //let (ptr, len, capacity) = Vec::into_raw_parts(vec);
        let (ptr, len, capacity) = (vec.as_mut_ptr(), vec.len(), vec.capacity());
        mem::forget(vec);
        unsafe {
            Self {
                data: BitSlice {
                    ptr: NonNull::new_unchecked(ptr),
                    len: real_len,
                    _marker: PhantomData,
                },
                capacity: capacity * 8,
            }
        }
    }
}

impl Drop for BitVec {
    fn drop(&mut self) {
        let _ =
            unsafe { Vec::from_raw_parts(self.data.ptr.as_ptr(), self.data.len, self.capacity) };
    }
}

impl Deref for BitVec {
    type Target = BitSlice<'static>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for BitVec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl FromIterator<bool> for BitVec {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let mut s = Self::with_capacity(iter.size_hint().0);
        iter.for_each(|v| s.push(v));
        s
    }
}
