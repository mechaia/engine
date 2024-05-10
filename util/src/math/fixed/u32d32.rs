use {
    super::U0d32,
    core::{fmt, ops},
    rand::{
        distributions::{uniform::{SampleBorrow, SampleUniform, UniformInt, UniformSampler}, Distribution, Standard},
        Rng,
    },
};

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct U32d32(pub u64);

pub struct U32d32Sampler {
    sampler: UniformInt<u64>,
}

impl U32d32 {
    pub const ZERO: Self = Self(0);
    pub const HALF: Self = Self(1u64 << 31);
    pub const ONE: Self = Self(1u64 << 32);
    pub const MIN: Self = Self(0);
    pub const MAX: Self = Self(u64::MAX);
    pub const EPSILON: Self = Self(1);

    pub fn from_f32(n: f32) -> Self {
        Self((n * ((1u64 << 32) as f32)) as u64)
    }

    pub fn from_f64(n: f64) -> Self {
        Self((n * ((1u64 << 32) as f64)) as u64)
    }

    pub fn to_f32(self) -> f32 {
        (self.0 as f32) / ((1u64 << 32) as f32)
    }

    pub fn to_f64(self) -> f64 {
        (self.0 as f64) / ((1u64 << 32) as f64)
    }

    pub fn int(self) -> u32 {
        (self.0 >> 32) as u32
    }

    pub fn frac(self) -> U0d32 {
        U0d32(self.0 as u32)
    }

    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        self.0.checked_sub(rhs.0).map(Self)
    }

    pub fn wrapping_sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }

    pub fn wrapping_mul(self, rhs: Self) -> Self {
        Self((u128::from(self.0).wrapping_mul(u128::from(rhs.0)) >> 32) as u64)
    }

    pub fn wrapping_mul_u0d32(self, rhs: U0d32) -> Self {
        self.wrapping_mul(rhs.into())
    }

    pub fn lerp(self, to: Self, t: Self) -> Self {
        (self * (Self::ONE - t)) + (to * t)
    }
}

impl ops::Add<Self> for U32d32 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl ops::Sub<Self> for U32d32 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl ops::Mul<u32> for U32d32 {
    type Output = Self;

    fn mul(self, rhs: u32) -> Self::Output {
        Self(self.0 * u64::from(rhs))
    }
}

impl ops::Mul<U32d32> for u32 {
    type Output = U32d32;

    fn mul(self, rhs: U32d32) -> Self::Output {
        rhs * self
    }
}

impl ops::Mul<Self> for U32d32 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(((u128::from(self.0) * u128::from(rhs.0)) >> 32) as u64)
    }
}

impl ops::Mul<U0d32> for U32d32 {
    type Output = Self;

    fn mul(self, rhs: U0d32) -> Self::Output {
        self * Self::from(rhs)
    }
}

impl ops::Div<Self> for U32d32 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self((u128::from(self.0 << 32) / u128::from(rhs.0)) as u64)
    }
}

impl ops::Div<U0d32> for U32d32 {
    type Output = Self;

    fn div(self, rhs: U0d32) -> Self::Output {
        self / Self::from(rhs)
    }
}

impl ops::Div<u32> for U32d32 {
    type Output = Self;

    fn div(self, rhs: u32) -> Self::Output {
        Self(self.0 / u64::from(rhs))
    }
}

impl ops::AddAssign<Self> for U32d32 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl ops::SubAssign<Self> for U32d32 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl ops::MulAssign<U0d32> for U32d32 {
    fn mul_assign(&mut self, rhs: U0d32) {
        *self = *self * rhs;
    }
}

impl fmt::Debug for U32d32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_f64().fmt(f)
    }
}

impl fmt::Display for U32d32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_f64().fmt(f)
    }
}

impl SampleUniform for U32d32 {
    type Sampler = U32d32Sampler;
}

impl UniformSampler for U32d32Sampler {
    type X = U32d32;

    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Self {
            sampler: UniformInt::new(low.borrow().0, high.borrow().0),
        }
    }

    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Self {
            sampler: UniformInt::new(low.borrow().0, high.borrow().0),
        }
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        U32d32(self.sampler.sample(rng))
    }

    fn sample_single<R: Rng + ?Sized, B1, B2>(low: B1, high: B2, rng: &mut R) -> Self::X
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        U32d32(UniformInt::<u64>::sample_single(
            low.borrow().0,
            high.borrow().0,
            rng,
        ))
    }

    fn sample_single_inclusive<R: Rng + ?Sized, B1, B2>(low: B1, high: B2, rng: &mut R) -> Self::X
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        U32d32(UniformInt::<u64>::sample_single_inclusive(
            low.borrow().0,
            high.borrow().0,
            rng,
        ))
    }
}

impl Distribution<U0d32> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> U0d32 {
        U0d32(rng.gen())
    }
}
