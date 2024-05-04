pub mod envelope;
pub mod filter;
pub mod transform;
pub mod wave;

use crate::Source;
use glam::Vec3;
use util::{
    math::fixed::{U0d32, U32d32},
    soa::Slice2,
};

pub struct Periodic<F> {
    f: F,
    t: U0d32,
}

pub struct Increasing<F> {
    f: F,
    t: U32d32,
}

pub struct Adjust<S, F, G> {
    source: S,
    time: F,
    sample: G,
}

pub struct Combine<A, B, F> {
    a: A,
    b: B,
    f: F,
}

pub struct LimitDuration<S> {
    source: S,
    time: U32d32,
    max_duration: U32d32,
}

pub struct NegExp<S> {
    source: S,
    time: U32d32,
    scale: U32d32,
    scale_factor: U32d32,
}


// make generic arg type explicit here since I'm sick of type errors
impl<F: Fn(U0d32) -> f32> Periodic<F> {
    pub const fn new(f: F) -> Self {
        Self { f, t: U0d32::ZERO }
    }
}

// ditto
impl<S, F: Fn(U32d32) -> U32d32, G: Fn(f32) -> f32> Adjust<S, F, G> {
    pub const fn new(source: S, time: F, sample: G) -> Self {
        Self {
            source,
            time,
            sample,
        }
    }
}

// ditto
impl<A, B, F: Fn(f32, f32) -> f32> Combine<A, B, F> {
    pub fn new(a: A, b: B, f: F) -> Self {
        Self { a, b, f }
    }
}

impl<F: Fn(U0d32) -> f32> Source for Periodic<F> {
    fn current(&self, channel: usize) -> f32 {
        (self.f)(self.t)
    }

    fn next_sample(&mut self, dt: U32d32) -> bool {
        self.t = self.t.wrapping_add(dt.frac());
        true
    }
}

impl<A: Source, B: Source, F: Fn(f32, f32) -> f32> Source for Combine<A, B, F> {
    fn current(&self, channel: usize) -> f32 {
        (self.f)(self.a.current(channel), self.b.current(channel))
    }

    fn next_sample(&mut self, dt: U32d32) -> bool {
        let x = self.a.next_sample(dt);
        let y = self.b.next_sample(dt);
        x & y
    }
}

impl<S: Source, F: Fn(U32d32) -> U32d32, G: Fn(f32) -> f32> Source for Adjust<S, F, G> {
    fn current(&self, channel: usize) -> f32 {
        (self.sample)(self.source.current(channel))
    }

    fn next_sample(&mut self, dt: U32d32) -> bool {
        self.source.next_sample((self.time)(dt))
    }
}

impl<S: Source> Source for LimitDuration<S> {
    fn current(&self, channel: usize) -> f32 {
        self.source.current(channel)
    }

    fn next_sample(&mut self, dt: U32d32) -> bool {
        self.time += dt;
        self.time <= self.max_duration && self.source.next_sample(dt)
    }
}

pub fn lerp(x: f32, y: f32, t: f32) -> f32 {
    x + ((y - x) * t)
}

pub fn neg_exp(t: U32d32) -> f32 {
    (-t.to_f32()).exp()
}

pub fn limit_duration<S>(source: S, max_duration: U32d32) -> LimitDuration<S> {
    LimitDuration {
        source,
        time: U32d32::ZERO,
        max_duration,
    }
}
