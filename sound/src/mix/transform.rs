use std::ops;
use rand::{Rng, RngCore};
use util::math::fixed::U32d32;
use crate::Source;

pub struct Damp<S> {
    value: f32,
    damp: f32,
    source: S,
}

pub struct RandomTimeScale<R, S> {
    rng: R,
    source: S,
    range: ops::RangeInclusive<U32d32>,
}

impl<R, S> RandomTimeScale<R, S> {
    pub fn new(rng: R, source: S, range: ops::RangeInclusive<U32d32>) -> Self {
        Self { rng, source, range }
    }
}

impl<S> Damp<S> {
    pub fn new(source: S, damp: f32) -> Self {
        Self {
            value: 0.0,
            damp,
            source,
        }
    }
}

impl<R: RngCore, S: Source> Source for RandomTimeScale<R, S> {
    fn current(&self, channel: usize) -> f32 {
        self.source.current(0)
    }

    fn next_sample(&mut self, dt: U32d32) -> bool {
        self.source.next_sample(dt * self.rng.gen_range(self.range.clone()))
    }
}

impl<S: Source> Source for Damp<S> {
    fn current(&self, channel: usize) -> f32 {
        self.value
    }

    fn next_sample(&mut self, dt: U32d32) -> bool {
        if !self.source.next_sample(dt) {
            return false;
        }
        //self.value = lerp(self.source.current(0), self.current(0), dbg!(self.damp.powf(dt.to_f32())));
        self.value = super::lerp(self.source.current(0), self.current(0), self.damp);
        true
    }
}

pub fn linear_dropoff(sample: f32, distance: f32) -> f32 {
    sample / distance
}

pub fn square_dropoff(sample: f32, distance: f32) -> f32 {
    sample / (distance * distance)
}

//pub fn flatten<S>(s: <S
