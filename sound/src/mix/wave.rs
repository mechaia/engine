use rand::{rngs::ThreadRng, Rng, RngCore};
use std::f32::consts::TAU;
use util::math::fixed::{U0d32, U32d32};

pub struct FullRandom<R = ThreadRng> {
    rng: R,
    current: f32,
}

pub struct RandomStep<R = ThreadRng> {
    rng: R,
    current: f32,
    velocity: f32,
    max_velocity: f32,
    acceleration: f32,
}

pub struct RandomStep2<R = ThreadRng> {
    rng: R,
    current: f32,
    velocity: f32,
}

impl<R: RngCore> FullRandom<R> {
    pub fn new(mut rng: R) -> Self {
        Self {
            current: rng.gen_range(-1.0..=1.0),
            rng,
        }
    }

    pub fn next_sample(&mut self, dt: U32d32) -> f32 {
        self.rng.gen_range(-1.0..=1.0)
    }
}

impl<R: RngCore> RandomStep<R> {
    pub fn new(rng: R, acceleration: f32, max_velocity: f32) -> Self {
        Self {
            rng,
            current: 0.0,
            velocity: 0.0,
            max_velocity,
            acceleration,
        }
    }

    pub fn set_acceleration(&mut self, acceleration: f32) {
        self.acceleration = acceleration;
    }

    pub fn next_sample(&mut self, dt: U32d32) -> f32 {
        let accel = dt.to_f32() * self.acceleration;
        let towards = self.rng.gen_range(-1.0..=1.0) / 2.0;
        if towards < self.current {
            self.velocity -= accel;
        } else {
            self.velocity += accel;
        }
        self.velocity = self.velocity.clamp(-self.max_velocity, self.max_velocity);
        self.current += self.velocity;
        //self.current = self.current.clamp(-1.0, 1.0);
        self.current
    }
}

impl<R: RngCore> RandomStep2<R> {
    pub fn new(mut rng: R, velocity: f32) -> Self {
        Self {
            rng,
            current: 0.0,
            velocity,
        }
    }

    pub fn next_sample(&mut self, dt: U32d32) -> f32 {
        let target = self.rng.gen_range(-1.0..=1.0);
        self.current = super::lerp(self.current, target, dt.to_f32() * self.velocity);
        self.current
    }
}

#[inline(always)]
fn map_01_to_n11(y: f32) -> f32 {
    (2.0 * y) - 1.0
}

pub fn sawtooth(t: U0d32) -> f32 {
    map_01_to_n11(t.to_f32())
}

pub fn square(t: U0d32) -> f32 {
    map_01_to_n11(f32::from(t < U0d32::HALF))
}

pub fn sine(t: U0d32) -> f32 {
    (t.to_f32() * TAU).sin()
}

pub fn triangle(t: U0d32) -> f32 {
    // at step (2) the wave will start at -1
    // add 1/4 to correct for that
    let t = t.wrapping_add(U0d32::FRAC_1_4 * 3).to_f32();

    // (1)     /                   \   /
    //        /                     \ /
    //       +           =>          +
    //      /
    //     /
    let y = map_01_to_n11(t).abs();
    // (2) \   /                     .
    //      \ /                     / \
    //       +           =>        / + \
    map_01_to_n11(1.0 - y)
}

/*
pub fn inv_power(t: U0d32, base: f32, factor: f32) -> f32 {
    base.powf(t.to_f32() * factor)
}

pub fn inv_exp(t: U0d32, factor: f32) -> f32 {
    (t.to_f32() * factor).exp()
}

// https://physics.stackexchange.com/a/73791
//
// Sound[{Play[
//    Re[E^(-((10000 I)/(4 10^-6 I + 60 t)))/Sqrt[10^-6 - 15 I t]], {t, 0, 15}]}]
//
// 10000 here is distance^2
pub fn emilio_pisanty(t: U0d32, dur: f32, distance: f32) -> f32 {
    use util::Complex;

    let t = t.to_f32() * 15.0;

    let i = Complex::i();

    let den = (1e-6 - dur * i * t).sqrt();
    let num = {
        let den = 4.0 * (i * 1e-6 + dur * t);
        let num = -distance.powi(2) * i;
        num / den
    }
    .exp();

    (num / den).re
}
*/
