use {
    super::U32d32,
    core::{fmt, num::Wrapping, ops},
    rand::{
        distributions::uniform::{SampleBorrow, SampleUniform, UniformInt, UniformSampler},
        Rng,
    },
};

/// Range [0;1[
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct U0d32(pub u32);

pub struct U0d32Sampler {
    sampler: UniformInt<u32>,
}

impl U0d32 {
    pub const ZERO: Self = Self(0);
    pub const HALF: Self = Self(1u32 << 31);
    pub const FRAC_1_4: Self = Self(1u32 << 30);
    pub const MIN: Self = Self(0);
    pub const MAX: Self = Self(u32::MAX);
    pub const EPSILON: Self = Self(1);

    pub fn from_f32(n: f32) -> Self {
        Self((n * ((1u64 << 32) as f32)) as u32)
    }

    pub fn to_f32(self) -> f32 {
        (self.0 as f32) / ((1u64 << 32) as f32)
    }

    pub fn to_f64(self) -> f64 {
        (self.0 as f64) / ((1u64 << 32) as f64)
    }

    pub fn powi(mut self, mut n: u32) -> Self {
        let mut x = U32d32::ONE;
        while n != 0 {
            if n & 1 != 0 {
                x *= self;
            }
            n >>= 1;
            self *= self;
        }
        x.frac()
    }

    pub fn checked_sub(self, rhs: U0d32) -> Option<Self> {
        self.0.checked_sub(rhs.0).map(Self)
    }

    pub fn wrapping_add(self, rhs: U0d32) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }

    pub fn wrapping_sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }

    pub fn wrapping_mul(self, rhs: U0d32) -> Self {
        Self((u64::from(self.0).wrapping_mul(u64::from(rhs.0)) >> 32) as u32)
    }

    pub fn wrapping_mul_u32(self, rhs: u32) -> Self {
        Self(self.0.wrapping_mul(rhs))
    }

    pub fn saturating_sub(self, rhs: U0d32) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }
}

impl From<u32> for U32d32 {
    fn from(value: u32) -> Self {
        Self(u64::from(value) << 32)
    }
}

impl From<U0d32> for U32d32 {
    fn from(value: U0d32) -> Self {
        Self(u64::from(value.0))
    }
}

impl ops::Add<Self> for U0d32 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl ops::Sub<Self> for U0d32 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl ops::Mul<u32> for U0d32 {
    type Output = Self;

    fn mul(self, rhs: u32) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl ops::Mul<U0d32> for u32 {
    type Output = U0d32;

    fn mul(self, rhs: U0d32) -> Self::Output {
        rhs * self
    }
}

impl ops::Mul<Self> for U0d32 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(((u64::from(self.0) * u64::from(rhs.0)) >> 32) as u32)
    }
}

impl ops::Div<u32> for U0d32 {
    type Output = Self;

    fn div(self, rhs: u32) -> Self::Output {
        Self(self.0 / rhs)
    }
}

impl ops::AddAssign<Self> for U0d32 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl ops::SubAssign<Self> for U0d32 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl ops::MulAssign<Self> for U0d32 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl ops::Div<U32d32> for U0d32 {
    type Output = Self;

    fn div(self, rhs: U32d32) -> Self::Output {
        Self(((u64::from(self.0) << 32) / rhs.0) as u32)
    }
}

impl ops::Add<U0d32> for Wrapping<U0d32> {
    type Output = Self;

    fn add(self, rhs: U0d32) -> Self::Output {
        Self(self.0.wrapping_add(rhs))
    }
}

impl ops::AddAssign<U0d32> for Wrapping<U0d32> {
    fn add_assign(&mut self, rhs: U0d32) {
        *self = *self + rhs;
    }
}

impl fmt::Debug for U0d32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        U32d32::from(*self).fmt(f)
    }
}

impl fmt::Display for U0d32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        U32d32::from(*self).fmt(f)
    }
}

impl SampleUniform for U0d32 {
    type Sampler = U0d32Sampler;
}

impl UniformSampler for U0d32Sampler {
    type X = U0d32;

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
        U0d32(self.sampler.sample(rng))
    }

    fn sample_single<R: Rng + ?Sized, B1, B2>(low: B1, high: B2, rng: &mut R) -> Self::X
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        U0d32(UniformInt::<u32>::sample_single(
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
        U0d32(UniformInt::<u32>::sample_single_inclusive(
            low.borrow().0,
            high.borrow().0,
            rng,
        ))
    }
}
