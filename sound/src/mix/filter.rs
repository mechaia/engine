use crate::Source;
use util::math::fixed::U32d32;

pub struct FiniteImpulseResponse<const N: usize> {
    weights: [f32; N],
    delta: U32d32,
}

pub struct RingBuffer<const N: usize> {
    values: [[f32; 2]; N],
    index: usize,
}

pub struct FiniteImpulseResponseSource<const N: usize, S> {
    fir: FiniteImpulseResponse<N>,
    samples: RingBuffer<N>,
    source: S,
    t: U32d32,
}

impl<const N: usize> FiniteImpulseResponse<N> {
    pub fn new(weights: [f32; N], delta: U32d32) -> Self {
        let sum = weights.iter().sum::<f32>();
        let weights = weights.map(|x| x / sum);
        Self { weights, delta }
    }

    pub fn apply(&self, samples: &RingBuffer<N>, channel: usize) -> f32 {
        self.weights
            .iter()
            .zip(samples.iter())
            .map(|(&w, s)| w * s[channel])
            .sum()
    }
}

impl<const N: usize> RingBuffer<N> {
    pub fn new_filled(value: f32) -> Self {
        Self {
            values: [[value; 2]; N],
            index: 0,
        }
    }

    pub fn push(&mut self, value: [f32; 2]) {
        self.index += 1;
        if self.index >= self.values.len() {
            self.index = 0;
        }
        self.values[self.index] = value;
    }

    pub fn iter(&self) -> impl Iterator<Item = [f32; 2]> + '_ {
        (self.index..self.values.len())
            .chain(0..self.index)
            .map(|i| self.values[i])
    }
}

impl<const N: usize, S: Source> FiniteImpulseResponseSource<N, S> {
    pub fn new(source: S, weights: [f32; N], delta: U32d32) -> Self {
        Self {
            fir: FiniteImpulseResponse::new(weights, delta),
            samples: Default::default(),
            source,
            t: U32d32::ZERO,
        }
    }
}

impl<const N: usize, S: Source> Source for FiniteImpulseResponseSource<N, S> {
    fn current(&self, channel: usize) -> f32 {
        self.fir.apply(&self.samples, channel)
    }

    fn next_sample(&mut self, dt: U32d32) -> bool {
        if !self.source.next_sample(dt) {
            return false;
        }
        self.t += dt;
        while self.t >= self.fir.delta {
            self.samples
                .push([self.source.current(0), self.source.current(1)]);
            self.t -= dt;
        }
        true
    }
}

impl<const N: usize> Default for RingBuffer<N> {
    fn default() -> Self {
        Self {
            values: [[0.0; 2]; N],
            index: 0,
        }
    }
}
