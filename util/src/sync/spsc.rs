use super::half::{self, Half, Left, Right};
use core::{
    mem::MaybeUninit,
    ptr,
    sync::atomic::{AtomicUsize, Ordering},
};
use std::{cell::UnsafeCell, num::NonZeroUsize};

pub struct FixedSender<T> {
    half: Half<Left, Data<T>>,
}

pub struct FixedReceiver<T> {
    half: Half<Right, Data<T>>,
}

pub struct Iter<'a, T> {
    data: &'a Data<T>,
    index: usize,
    end: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            return None;
        }
        let v = unsafe { self.data.take_unchecked_masked(self.index) };
        self.index = self.index.wrapping_add(1);
        Some(v)
    }
}

// TODO figure out how to do layout properly for stuff like this.
// For now, stick with repr(C), which is what Arc does
#[repr(C)]
struct Data<T: Sized> {
    meta: Meta,
    data: [UnsafeCell<MaybeUninit<T>>],
}

struct Meta {
    send: AtomicUsize,
    recv: AtomicUsize,
    // size, as visible to the user.
    size: NonZeroUsize,
}

impl<T: Send + 'static> FixedSender<T> {
    pub fn size(&self) -> NonZeroUsize {
        self.half.meta.size
    }

    pub fn free(&self) -> usize {
        self.size().get() - self.filled()
    }

    pub fn filled(&self) -> usize {
        let send = self.half.meta.send.load(Ordering::Relaxed);
        // do recv last so we get the very latest value
        let recv = self.half.meta.recv.load(Ordering::Relaxed);
        send.wrapping_sub(recv)
    }

    /// Send a single value.
    ///
    /// Returns the value if queue is full.
    pub fn send(&mut self, value: T) -> Option<T> {
        let mut it = [value].into_iter();
        self.send_iter(&mut it);
        it.next()
    }

    /// Send an iterator of values.
    ///
    /// Returns the amount of elements sent.
    pub fn send_iter<I: IntoIterator<Item = T>>(&mut self, values: I) -> usize {
        let mut values = values.into_iter().take(self.free());
        let start @ mut index = self.half.meta.send.load(Ordering::Relaxed);

        let end = self.half.meta.recv.load(Ordering::Relaxed);
        let end = end + self.half.meta.size.get();

        while index != end {
            let Some(v) = values.next() else { break };
            unsafe {
                self.half.data[index & self.half.mask()]
                    .get()
                    .write(MaybeUninit::new(v));
            }
            index = index.wrapping_add(1);
        }

        self.half.meta.send.store(index, Ordering::Release);
        index.wrapping_sub(start)
    }
}

impl<T: Send + 'static> FixedReceiver<T> {
    pub fn size(&self) -> NonZeroUsize {
        self.half.meta.size
    }

    pub fn free(&self) -> usize {
        self.size().get() - self.filled()
    }

    pub fn filled(&self) -> usize {
        let recv = self.half.meta.recv.load(Ordering::Relaxed);
        // do send last so we get the very latest value
        let send = self.half.meta.send.load(Ordering::Relaxed);
        send.wrapping_sub(recv)
    }

    /// Receive a single value.
    ///
    /// Returns `None` if no values free.
    pub fn recv(&mut self) -> Option<T> {
        let index = self.half.meta.recv.load(Ordering::Relaxed);
        let end = self.half.meta.send.load(Ordering::Acquire);
        if index != end {
            let v = unsafe { self.half.take_unchecked_masked(index) };
            self.half
                .meta
                .recv
                .store(index.wrapping_add(1), Ordering::Relaxed);
            Some(v)
        } else {
            None
        }
    }

    // FIXME this is unsound
    // If we update the recv index immediately, send() can corrupt the values
    // If we update the recv index after, we risk corruption if mem::forget() is used.
    // Any runtime workarounds will probably add too much overhead compared to just updating
    // the counter.
    // Still, a way to avoid updating the counter til the would be nice to allow the compiler
    // (or us) to insert memcpy where possible.
    //
    // Add other methods like recv_into() to work around this.
    /*
    /// Receive multiple values, up to `amount`.
    pub fn recv_iter(&mut self, amount: usize) -> Iter<'_, T> {
        let index = self.half.meta.recv.load(Ordering::Relaxed);
        let end = self.half.meta.send.load(Ordering::Acquire);

        let count = end.wrapping_sub(index);
        let mincount = amount.min(count);

        let end = index.wrapping_add(mincount);
        self.half.meta.recv.store(end, Ordering::Relaxed);

        Iter { data: &self.half, index, end }
    }
    */

    /// Receive and put multiple values into a buffer.
    ///
    /// Returns the amount of values written.
    pub fn recv_into(&mut self, buf: &mut [T]) -> usize {
        let mut index = self.half.meta.recv.load(Ordering::Relaxed);
        let end = self.half.meta.send.load(Ordering::Acquire);

        let count = end.wrapping_sub(index);
        let mincount = buf.len().min(count);
        let end = index.wrapping_add(mincount);

        for v in buf[..mincount].iter_mut() {
            *v = unsafe { self.half.take_unchecked_masked(index) };
            index = index.wrapping_add(1);
        }

        self.half.meta.recv.store(end, Ordering::Relaxed);

        mincount
    }
}

impl<T> Data<T> {
    fn mask(&self) -> usize {
        self.data.len().wrapping_sub(1)
    }

    fn get_masked(&self, index: usize) -> *mut MaybeUninit<T> {
        self.data[index & self.mask()].get()
    }

    unsafe fn take_unchecked_masked(&self, index: usize) -> T {
        unsafe { (*self.get_masked(index)).assume_init_read() }
    }
}

impl<T> Drop for Data<T> {
    fn drop(&mut self) {
        let mut index = *self.meta.recv.get_mut();
        let end = *self.meta.send.get_mut();
        while index != end {
            unsafe { ptr::drop_in_place(self.get_masked(index)) };
            index = index.wrapping_add(1);
        }
    }
}

/// Create a channel with an exact fixed size.
pub fn fixed<T>(size: NonZeroUsize) -> (FixedSender<T>, FixedReceiver<T>) {
    // TODO figure out if there's a (good, efficient) way to avoid rounding to a power of 2.
    let size_p2 = size.get().wrapping_next_power_of_two();
    unsafe {
        let (left, right) = half::half_init_with::<Data<T>, _>(size_p2, |p| {
            p.cast::<Meta>().write(Meta {
                send: AtomicUsize::new(0),
                recv: AtomicUsize::new(0),
                size,
            })
        });
        (FixedSender { half: left }, FixedReceiver { half: right })
    }
}

#[cfg(test)]
mod test {
    struct DropCheck(u32);

    impl Drop for DropCheck {
        fn drop(&mut self) {
            assert_ne!(self.0, 0, "dropped twice!");
            self.0 = 0;
        }
    }

    #[test]
    fn fixed_3() {
        let (mut send, mut recv) = super::fixed::<DropCheck>(3.try_into().unwrap());
        assert_eq!(send.filled(), 0);

        assert!(send.send(DropCheck(42)).is_none());
        assert_eq!(send.filled(), 1);
        assert_eq!(recv.filled(), 1);

        assert_eq!(recv.recv().unwrap().0, 42);
        assert_eq!(send.filled(), 0);
        assert_eq!(recv.filled(), 0);

        assert!(send.send(DropCheck(1337)).is_none());
        assert_eq!(send.filled(), 1);
        assert_eq!(recv.filled(), 1);

        assert!(send.send(DropCheck(0xdeadbeef)).is_none());
        assert_eq!(send.filled(), 2);
        assert_eq!(recv.filled(), 2);

        assert!(send.send(DropCheck(0xcafebabe)).is_none());
        assert_eq!(send.filled(), 3);
        assert_eq!(recv.filled(), 3);

        assert!(!send.send(DropCheck(0xfa1afe1)).is_none());
        assert_eq!(send.filled(), 3);
        assert_eq!(recv.filled(), 3);

        assert_eq!(recv.recv().unwrap().0, 1337);
        assert_eq!(send.filled(), 2);
        assert_eq!(recv.filled(), 2);

        assert_eq!(recv.recv().unwrap().0, 0xdeadbeef);
        assert_eq!(send.filled(), 1);
        assert_eq!(recv.filled(), 1);

        assert_eq!(recv.recv().unwrap().0, 0xcafebabe);
        assert_eq!(send.filled(), 0);
        assert_eq!(recv.filled(), 0);

        assert!(recv.recv().is_none());
        assert_eq!(send.filled(), 0);
        assert_eq!(recv.filled(), 0);

        // one more to "wrap around"
        assert!(send.send(DropCheck(777)).is_none());
        assert_eq!(send.filled(), 1);
        assert_eq!(recv.filled(), 1);

        assert_eq!(recv.recv().unwrap().0, 777);
        assert_eq!(send.filled(), 0);
        assert_eq!(recv.filled(), 0);

        assert!(recv.recv().is_none());
        assert_eq!(send.filled(), 0);
        assert_eq!(recv.filled(), 0);
    }

    #[test]
    fn fixed_3_thread() {
        let (mut send, mut recv) = super::fixed::<DropCheck>(3.try_into().unwrap());

        let thr = std::thread::spawn(move || {
            assert!(send.send(DropCheck(42)).is_none());
            // bad for performance but who gives a shit
            while send.filled() > 0 {
                std::thread::yield_now();
            }
            assert!(send.send(DropCheck(1337)).is_none());
            assert!(send.send(DropCheck(0xdeadbeef)).is_none());
            assert!(send.send(DropCheck(0xcafebabe)).is_none());
            // this one may fail, so spin a bit
            let mut v = DropCheck(0xfa1afe1);
            loop {
                let Some(vv) = send.send(v) else { break };
                v = vv;
                std::thread::yield_now();
            }
            while send.filled() > 0 {
                std::thread::yield_now();
            }
            assert!(send.send(DropCheck(777)).is_none());
        });

        let mut r = || loop {
            if let Some(v) = recv.recv() {
                break v;
            }
            std::thread::yield_now();
        };

        assert_eq!(r().0, 42);
        assert_eq!(r().0, 1337);
        assert_eq!(r().0, 0xdeadbeef);
        assert_eq!(r().0, 0xcafebabe);
        assert_eq!(r().0, 0xfa1afe1);
        assert_eq!(r().0, 777);
        assert!(recv.recv().is_none());

        thr.join().unwrap();
    }

    #[test]
    fn empty_recv_into() {
        let (_, mut recv) = super::fixed(3.try_into().unwrap());
        assert_eq!(recv.free(), 3);
        assert_eq!(recv.filled(), 0);
        let n = recv.recv_into(&mut [0.0; 3]);
        assert_eq!(n, 0);
    }
}
