use core::{marker::PhantomData, ptr::NonNull, slice};
use std::alloc::Layout;

pub struct Slice<'a, R> {
    len: usize,
    raw: R,
    _marker: PhantomData<&'a R>,
}

pub struct Vec<R> {
    len: usize,
    capacity: usize,
    raw: R,
}

// rust is bugging out with move |i| ...
pub struct IterMut<'a, R> {
    cur: usize,
    len: usize,
    raw: &'a mut R,
}

impl<'a, R: Raw> Slice<'a, R> {
    pub fn as_slices(&'a self) -> R::ElementSlices<'a> {
        unsafe { self.raw.as_slices(self.len) }
    }

    pub fn as_slices_mut(&'a mut self) -> R::ElementMutSlices<'a> {
        unsafe { self.raw.as_slices_mut(self.len) }
    }

    pub fn from_slices(value: R::ElementSlices<'a>) -> Self {
        let (raw, len) = R::from_slices(value);
        let _marker = PhantomData;
        Self { len, raw, _marker }
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<R: Raw> Vec<R> {
    pub fn new() -> Self {
        Self {
            len: 0,
            capacity: 0,
            raw: R::DANGLING,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut slf = Self::new();
        slf.grow(capacity);
        slf
    }

    pub fn as_slice(&self) -> Slice<'_, R> {
        Slice {
            len: self.len,
            raw: self.raw,
            _marker: PhantomData,
        }
    }

    pub fn as_slices(&self) -> R::ElementSlices<'_> {
        unsafe { self.raw.as_slices(self.len) }
    }

    pub fn as_slices_mut(&mut self) -> R::ElementMutSlices<'_> {
        unsafe { self.raw.as_slices_mut(self.len) }
    }

    pub fn iter(&self) -> impl Iterator<Item = R::ElementRef<'_>> {
        (0..self.len).map(move |i| unsafe { self.raw.get_unchecked(i) })
    }

    pub fn iter_mut<'a>(&'a mut self) -> IterMut<'a, R> {
        IterMut {
            cur: 0,
            len: self.len,
            raw: &mut self.raw,
        }
    }

    pub fn push(&mut self, value: R::Element) {
        if self.capacity <= self.len {
            self.grow(self.len + 1);
        }
        unsafe {
            self.raw.set_unchecked(self.len, value);
            self.len += 1;
        }
    }

    pub fn pop(&mut self) -> Option<R::Element> {
        if self.is_empty() {
            return None;
        }
        unsafe {
            self.len -= 1;
            Some(self.raw.take_unchecked(self.len))
        }
    }

    pub fn get(&self, index: usize) -> Option<R::ElementRef<'_>> {
        if index < self.len() {
            unsafe { Some(self.raw.get_unchecked(index)) }
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<R::ElementMut<'_>> {
        if index < self.len() {
            unsafe { Some(self.raw.get_unchecked_mut(index)) }
        } else {
            None
        }
    }

    #[track_caller]
    pub fn set(&mut self, index: usize, value: R::Element) {
        assert!(index < self.len(), "out of bounds");
        unsafe {
            drop(self.raw.take_unchecked(index));
            self.raw.set_unchecked(index, value);
        }
    }

    pub fn clear(&mut self) {
        while !self.is_empty() {
            self.pop();
        }
    }

    fn grow(&mut self, mincap: usize) {
        let newcap = self.capacity.checked_mul(2).unwrap_or(usize::MAX);
        let newcap = mincap.max(newcap);
        unsafe { self.raw.grow(self.capacity, newcap) };
        self.capacity = newcap;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<R: Raw> Default for Vec<R> {
    fn default() -> Self {
        Self {
            len: 0,
            capacity: 0,
            raw: R::DANGLING,
        }
    }
}

impl<R: Raw> FromIterator<R::Element> for Vec<R> {
    fn from_iter<T: IntoIterator<Item = R::Element>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let mut slf = Self::with_capacity(iter.size_hint().0);
        iter.for_each(|v| slf.push(v));
        slf
    }
}

impl<'a, R: Raw> Iterator for IterMut<'a, R> {
    type Item = R::ElementMut<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur >= self.len {
            return None;
        }
        let v = unsafe { self.raw.get_unchecked_mut(self.cur) };
        // FIXME I cannot figure out why Rust is complaining about &mut self lifetime here
        // working around with transmute should be safe, but we should address the root issue
        let v = unsafe { core::mem::transmute(v) };
        self.cur += 1;
        Some(v)
    }
}

mod sealed {
    pub trait Sealed {}
}

#[doc(hidden)]
pub trait Raw: Copy + sealed::Sealed {
    const DANGLING: Self;
    type Element;
    type ElementRef<'a>
    where
        Self: 'a;
    type ElementMut<'a>
    where
        Self: 'a;
    type ElementSlices<'a>
    where
        Self: 'a;
    type ElementMutSlices<'a>
    where
        Self: 'a;

    unsafe fn as_slices(&self, len: usize) -> Self::ElementSlices<'_>;
    unsafe fn as_slices_mut(&mut self, len: usize) -> Self::ElementMutSlices<'_>;

    unsafe fn get_unchecked(&self, index: usize) -> Self::ElementRef<'_>;
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> Self::ElementMut<'_>;
    unsafe fn take_unchecked(&mut self, index: usize) -> Self::Element;
    unsafe fn set_unchecked(&mut self, index: usize, value: Self::Element);

    unsafe fn grow(&mut self, old_size: usize, new_size: usize);

    fn from_slices(slices: Self::ElementSlices<'_>) -> (Self, usize);
}

// TODO replace with variadic tuples, if we ever get them...
macro_rules! raw {
    ($name:ident $vec:ident $slice:ident $first:ident $($field:ident $type:ident)*) => {
        pub type $vec<$($type,)*> = Vec<$name<$($type,)*>>;

        pub type $slice<'a, $($type,)*> = Slice<'a, $name<$($type,)*>>;

        #[doc(hidden)]
        pub struct $name<$($type,)*> {
            $($field: NonNull<$type>,)*
        }

        impl<$($type,)*> Clone for $name<$($type,)*> {
            fn clone(&self) -> Self {
                Self { $($field: self.$field,)* }
            }
        }

        impl<$($type,)*> Copy for $name<$($type,)*> {}

        impl<$($type,)*> sealed::Sealed for $name<$($type,)*> {}

        impl<$($type,)*> Raw for $name<$($type,)*> {
            const DANGLING: Self = Self { $($field: NonNull::dangling(),)* };
            type Element = ($($type,)*);
            type ElementRef<'a> = ($(&'a $type,)*)
            where
                $($type: 'a,)*
                Self: 'a;
            type ElementMut<'a> = ($(&'a mut $type,)*)
            where
                $($type: 'a,)*
                Self: 'a;
            type ElementSlices<'a> = ($(&'a [$type],)*)
            where
                $($type: 'a,)*
                Self: 'a;
            type ElementMutSlices<'a> = ($(&'a mut [$type],)*)
            where
                $($type: 'a,)*
                Self: 'a;

            unsafe fn as_slices(&self, len: usize) -> Self::ElementSlices<'_> {
                ($(slice::from_raw_parts(self.$field.as_ptr(), len),)*)
            }

            unsafe fn as_slices_mut(&mut self, len: usize) -> Self::ElementMutSlices<'_> {
                ($(slice::from_raw_parts_mut(self.$field.as_ptr(), len),)*)
            }

            unsafe fn get_unchecked(&self, index: usize) -> Self::ElementRef<'_> {
                ($(&*self.$field.as_ptr().add(index),)*)
            }

            unsafe fn get_unchecked_mut(&mut self, index: usize) -> Self::ElementMut<'_> {
                ($(&mut *self.$field.as_ptr().add(index),)*)
            }

            unsafe fn take_unchecked(&mut self, index: usize) -> Self::Element {
                ($(self.$field.as_ptr().add(index).read(),)*)
            }

            unsafe fn set_unchecked(&mut self, index: usize, value: Self::Element) {
                let ($($field,)*) = value;
                $(self.$field.as_ptr().add(index).write($field);)*
            }

            unsafe fn grow(&mut self, old_size: usize, new_size: usize) {
                $(self.$field = grow::<$type>(self.$field, old_size, new_size);)*
            }

            fn from_slices(($($field,)*): Self::ElementSlices<'_>) -> (Self, usize) {
                let len = $first.len();
                // assert_eq! generated a fair bit of bloat, so do it manually
                $(if $field.len() != len { std::panic::panic_any("SoA length mismatch") })*
                unsafe {
                    (Self { $($field: NonNull::new_unchecked($field.as_ptr() as *mut _),)* }, len)
                }
            }
        }
    };
}

raw!(Raw2 Vec2 Slice2 a a A b B);
raw!(Raw3 Vec3 Slice3 a a A b B c C);
raw!(Raw4 Vec4 Slice4 a a A b B c C d D);
raw!(Raw5 Vec5 Slice5 a a A b B c C d D e E);
raw!(Raw6 Vec6 Slice6 a a A b B c C d D e E f F);
raw!(Raw7 Vec7 Slice7 a a A b B c C d D e E f F g G);
raw!(Raw8 Vec8 Slice8 a a A b B c C d D e E f F g G h H);

unsafe fn grow<T>(ptr: NonNull<T>, old_size: usize, new_size: usize) -> NonNull<T> {
    let old_layout = Layout::array::<T>(old_size).expect("layout");
    let new_layout = Layout::array::<T>(new_size).expect("layout");
    let ptr = if old_size > 0 {
        std::alloc::realloc(ptr.as_ptr().cast(), old_layout, new_layout.size())
    } else {
        std::alloc::alloc(new_layout)
    };
    NonNull::new(ptr).expect("realloc").cast()
}
