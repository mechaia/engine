use {
    self::sealed::Side,
    super::{FastAtomic, FastNonAtomic},
    core::{
        alloc::Layout,
        mem, ops,
        ptr::{self, NonNull},
        sync::atomic,
    },
};

mod sealed {
    pub trait Side {
        const FLAG_LIVE_BIT: super::FastNonAtomic;
        const SELF: Self;
    }
}

pub struct Left;
pub struct Right;

pub struct Half<S: sealed::Side, T: ?Sized> {
    shared: NonNull<HalfData<T>>,
    _side: S,
}

pub(super) struct HalfData<T: ?Sized> {
    flags: FastAtomic,
    data: T,
}

impl sealed::Side for Left {
    const FLAG_LIVE_BIT: FastNonAtomic = 1 << 0;
    const SELF: Self = Self;
}

impl sealed::Side for Right {
    const FLAG_LIVE_BIT: FastNonAtomic = 1 << 1;
    const SELF: Self = Self;
}

impl<S: sealed::Side, T> Half<S, mem::MaybeUninit<T>> {
    pub fn assume_init(self) -> Half<S, T> {
        Half {
            shared: self.shared.cast(),
            _side: S::SELF,
        }
    }
}

impl<S: sealed::Side, T: ?Sized> Drop for Half<S, T> {
    fn drop(&mut self) {
        unsafe {
            let mask = !S::FLAG_LIVE_BIT;
            let prev = self
                .shared
                .as_ref()
                .flags
                .fetch_and(mask, atomic::Ordering::Relaxed);
            if prev & mask == 0 {
                do_drop(self.shared)
            }
        }
    }
}

unsafe impl<S: sealed::Side, T: ?Sized> Send for Half<S, T> {}

impl<S: sealed::Side, T: ?Sized> ops::Deref for Half<S, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &self.shared.as_ref().data }
    }
}

#[cold]
unsafe fn do_drop<T: ?Sized>(shared: NonNull<HalfData<T>>) {
    ptr::drop_in_place(shared.as_ptr());
    let (ptr, metadata) = shared.as_ptr().to_raw_parts();
    let layout = layout::<HalfData<T>>(metadata);
    std::alloc::dealloc(ptr.cast(), layout);
}

pub fn half<T>(value: T) -> (Half<Left, T>, Half<Right, T>) {
    half_with(move || value)
}

pub fn half_default<T: Default>() -> (Half<Left, T>, Half<Right, T>) {
    half_with(T::default)
}

pub fn half_with<T, F: FnOnce() -> T>(f: F) -> (Half<Left, T>, Half<Right, T>) {
    unsafe { half_init_with((), |p: *mut T| p.write(f())) }
}

pub fn half_uninit<T>() -> (
    Half<Left, mem::MaybeUninit<T>>,
    Half<Right, mem::MaybeUninit<T>>,
) {
    todo!()
}

/// # Safety
///
/// Metadata must be valid.
///
/// Function must write valid data.
pub(super) unsafe fn half_init_with<T, F>(
    metadata: <HalfData<T> as ptr::Pointee>::Metadata,
    f: F,
) -> (Half<Left, T>, Half<Right, T>)
where
    T: ?Sized,
    HalfData<T>: ptr::Pointee,
    F: FnOnce(*mut T),
{
    unsafe {
        let layout = layout::<HalfData<T>>(metadata);

        let ptr = std::alloc::alloc(layout);
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let ptr: *mut HalfData<T> = ptr::from_raw_parts_mut(ptr.cast(), metadata);

        ptr::addr_of_mut!((*ptr).flags).write(super::FastAtomic::new(
            Left::FLAG_LIVE_BIT | Right::FLAG_LIVE_BIT,
        ));
        f(ptr::addr_of_mut!((*ptr).data));

        let shared = NonNull::new_unchecked(ptr);
        (
            Half {
                shared,
                _side: Left,
            },
            Half {
                shared,
                _side: Right,
            },
        )
    }
}

unsafe fn layout<T: ?Sized + ptr::Pointee>(metadata: T::Metadata) -> Layout {
    let stub_ptr: *const T = ptr::from_raw_parts(ptr::null(), metadata);
    Layout::for_value_raw(stub_ptr)
}
