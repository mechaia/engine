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

pub struct BitBox {
    data: BitSlice<'static>,
}

pub struct BitVec {
    data: BitSlice<'static>,
    capacity: usize,
}

impl<'a> BitSlice<'a> {
    pub fn get(&self, index: usize) -> Option<bool> {
        if index >= self.len {
            return None;
        }
        unsafe { Some(self.get_unchecked(index)) }
    }

    pub fn set(&mut self, index: usize, value: bool) {
        assert!(index < self.len, "out of bounds");
        unsafe { self.set_unchecked(index, value) }
    }

    pub fn replace(&mut self, index: usize, value: bool) -> bool {
        assert!(index < self.len, "out of bounds");
        unsafe { self.replace_unchecked(index, value) }
    }

    pub fn fill(&mut self, value: bool) {
        unsafe {
            self.ptr
                .as_ptr()
                .write_bytes(u8::MAX * u8::from(value), self.u8_len())
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    fn u8_len(&self) -> usize {
        (self.len + 7) / 8
    }

    unsafe fn get_unchecked(&self, index: usize) -> bool {
        (*self.ptr.as_ptr().add(index / 8) >> (index % 8)) & 1 != 0
    }

    unsafe fn set_unchecked(&mut self, index: usize, value: bool) {
        self.replace_unchecked(index, value);
    }

    unsafe fn replace_unchecked(&mut self, index: usize, value: bool) -> bool {
        let p = &mut *self.ptr.as_ptr().add(index / 8);
        let k = index % 8;

        let v = *p;
        let mask = 1 << k;

        *p &= !mask;
        *p |= u8::from(value) << k;

        v & mask != 0
    }
}

impl Default for BitSlice<'static> {
    fn default() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            _marker: PhantomData,
        };
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            _marker: PhantomData,
        }
    }
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
            unsafe {
                slice_mut_u8(self.ptr, self.capacity / 8)[self.len / 8].write(u8::from(value))
            };
        } else {
            let l = self.len;
            unsafe { self.set_unchecked(l, value) }
        }
        self.len += 1;
    }

    pub fn iter(&self) -> impl Iterator<Item = bool> + '_ {
        (0..self.len).map(|i| unsafe { self.get_unchecked(i) })
    }

    fn grow(&mut self, min_cap: usize) {
        self.as_vec(|v| {
            v.reserve(min_cap - v.capacity());
        });
    }

    fn from_vec(mut vec: Vec<u8>, real_len: usize) -> Self {
        debug_assert_eq!(vec.len(), (real_len + 7) / 8);
        let (ptr, capacity) = (vec.as_mut_ptr(), vec.capacity());
        mem::forget(vec);
        Self {
            data: BitSlice {
                ptr: unsafe { NonNull::new_unchecked(ptr) },
                len: real_len,
                _marker: PhantomData,
            },
            capacity: capacity * 8,
        }
    }

    fn into_vec(self) -> Vec<u8> {
        let v = unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), self.len, self.capacity) };
        mem::forget(self);
        v
    }

    fn as_vec(&mut self, f: impl FnOnce(&mut Vec<u8>)) {
        let mut vec = mem::take(self).into_vec();
        f(&mut vec);
        *self = Self::from_vec(vec, self.len)
    }
}

impl Drop for BitVec {
    fn drop(&mut self) {
        mem::take(self).into_vec();
    }
}

impl Default for BitVec {
    fn default() -> Self {
        Self::from_vec(Vec::default(), 0)
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

unsafe fn slice_mut_u8<'a>(ptr: NonNull<u8>, len: usize) -> &'a mut [mem::MaybeUninit<u8>] {
    core::slice::from_raw_parts_mut(ptr.cast().as_ptr(), len / 8)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bitvec_push() {
        let mut v = BitVec::default();
        assert_eq!(v.len, 0);
        assert_eq!(v.capacity, 0);
        assert_eq!(v.get(0), None);
        v.push(true);
        assert_eq!(v.len, 1);
        assert!(v.capacity >= 8);
        assert_eq!(v.get(0), Some(true));
        assert_eq!(v.get(1), None);
        let cap = v.capacity;
        v.push(true);
        assert_eq!(v.len, 2);
        assert_eq!(v.capacity, cap);
        assert_eq!(v.get(0), Some(true));
        assert_eq!(v.get(1), Some(true));
        assert_eq!(v.get(2), None);
        v.push(false);
        assert_eq!(v.len, 3);
        assert_eq!(v.capacity, cap);
        assert_eq!(v.get(0), Some(true));
        assert_eq!(v.get(1), Some(true));
        assert_eq!(v.get(2), Some(false));
        assert_eq!(v.get(3), None);
        v.push(false);
        assert_eq!(v.len, 4);
        assert_eq!(v.capacity, cap);
        assert_eq!(v.get(0), Some(true));
        assert_eq!(v.get(1), Some(true));
        assert_eq!(v.get(2), Some(false));
        assert_eq!(v.get(3), Some(false));
        assert_eq!(v.get(4), None);
        v.push(false);
        assert_eq!(v.len, 5);
        assert_eq!(v.capacity, cap);
        assert_eq!(v.get(0), Some(true));
        assert_eq!(v.get(1), Some(true));
        assert_eq!(v.get(2), Some(false));
        assert_eq!(v.get(3), Some(false));
        assert_eq!(v.get(4), Some(false));
        assert_eq!(v.get(5), None);
        v.push(true);
        assert_eq!(v.len, 6);
        assert_eq!(v.capacity, cap);
        assert_eq!(v.get(0), Some(true));
        assert_eq!(v.get(1), Some(true));
        assert_eq!(v.get(2), Some(false));
        assert_eq!(v.get(3), Some(false));
        assert_eq!(v.get(4), Some(false));
        assert_eq!(v.get(5), Some(true));
        assert_eq!(v.get(6), None);
        v.push(false);
        assert_eq!(v.len, 7);
        assert_eq!(v.capacity, cap);
        assert_eq!(v.get(0), Some(true));
        assert_eq!(v.get(1), Some(true));
        assert_eq!(v.get(2), Some(false));
        assert_eq!(v.get(3), Some(false));
        assert_eq!(v.get(4), Some(false));
        assert_eq!(v.get(5), Some(true));
        assert_eq!(v.get(6), Some(false));
        assert_eq!(v.get(7), None);
        v.push(false);
        assert_eq!(v.len, 8);
        assert_eq!(v.capacity, cap);
        assert_eq!(v.get(0), Some(true));
        assert_eq!(v.get(1), Some(true));
        assert_eq!(v.get(2), Some(false));
        assert_eq!(v.get(3), Some(false));
        assert_eq!(v.get(4), Some(false));
        assert_eq!(v.get(5), Some(true));
        assert_eq!(v.get(6), Some(false));
        assert_eq!(v.get(7), Some(false));
        assert_eq!(v.get(8), None);
        v.push(false);
        assert_eq!(v.len, 9);
        assert!(v.capacity >= 16);
        assert_eq!(v.get(0), Some(true));
        assert_eq!(v.get(1), Some(true));
        assert_eq!(v.get(2), Some(false));
        assert_eq!(v.get(3), Some(false));
        assert_eq!(v.get(4), Some(false));
        assert_eq!(v.get(5), Some(true));
        assert_eq!(v.get(6), Some(false));
        assert_eq!(v.get(7), Some(false));
        assert_eq!(v.get(8), Some(false));
        assert_eq!(v.get(9), None);
    }
}
