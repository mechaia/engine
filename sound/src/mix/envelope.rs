use {
    crate::Source,
    util::{
        math::fixed::{U0d32, U32d32},
        soa::{Slice2, Vec2, Vec3},
    },
};

struct Envelope<S> {
    start_time_scale: U32d32,
    start_amplitude_scale: f32,
    points: Vec3<U0d32, U32d32, f32>,
    source: S,
    t: U0d32,
}

pub struct Floor<S>(Envelope<S>);
pub struct Linear<S>(Envelope<S>);

impl<S> Envelope<S> {
    fn new(
        start_time_scale: U32d32,
        start_amplitude_scale: f32,
        points: Vec3<U0d32, U32d32, f32>,
        source: S,
    ) -> Self {
        Self {
            start_time_scale,
            start_amplitude_scale,
            points,
            source,
            t: U0d32::ZERO,
        }
    }

    fn index_plus_one(&self) -> usize {
        for (i, (&pt, _, __)) in self.points.iter().enumerate().rev() {
            if pt <= self.t {
                return i + 1;
            }
        }
        0
    }

    fn get(&self, index: usize) -> (Option<U0d32>, U32d32, f32) {
        self.points.get(index).map_or(
            (None, self.start_time_scale, self.start_amplitude_scale),
            |(t, st, sa)| (Some(*t), *st, *sa),
        )
    }
}

impl<S: Source> Envelope<S> {
    fn next_sample(&mut self, dt: U32d32, scale: U32d32) -> bool {
        self.t = self.t.wrapping_add(dt.frac());
        self.source.next_sample(dt * scale)
    }
}

impl<S: Source> Floor<S> {
    pub fn new(
        start_time_scale: U32d32,
        start_amplitude_scale: f32,
        points: Vec3<U0d32, U32d32, f32>,
        source: S,
    ) -> Self {
        Self(Envelope::new(
            start_time_scale,
            start_amplitude_scale,
            points,
            source,
        ))
    }
}

impl<S: Source> Linear<S> {
    pub fn new(
        start_time_scale: U32d32,
        start_amplitude_scale: f32,
        points: Vec3<U0d32, U32d32, f32>,
        source: S,
    ) -> Self {
        Self(Envelope::new(
            start_time_scale,
            start_amplitude_scale,
            points,
            source,
        ))
    }
}

impl<S: Source> Source for Floor<S> {
    fn current(&self, channel: usize) -> f32 {
        let i = self.0.index_plus_one().wrapping_sub(1);
        let (_, _, sa) = self.0.get(i);
        self.0.source.current(channel) * sa
    }

    fn next_sample(&mut self, dt: U32d32) -> bool {
        let i = self.0.index_plus_one().wrapping_sub(1);
        let (_, st, _) = self.0.get(i);
        self.0.next_sample(dt, st)
    }
}

impl<S: Source> Source for Linear<S> {
    fn current(&self, channel: usize) -> f32 {
        let i = self.0.index_plus_one();
        let (ta, _, saa) = self.0.get(i.wrapping_sub(1));
        let (tb, _, sab) = self.0.get(i);

        let ta = ta.unwrap_or(U0d32::ZERO);
        let tb = tb.map_or(U32d32::ONE, U32d32::from);
        let t = (self.0.t - ta) / (tb - U32d32::from(ta));

        let s = super::lerp(saa, sab, t.to_f32());

        self.0.source.current(channel) * s
    }

    fn next_sample(&mut self, dt: U32d32) -> bool {
        let i = self.0.index_plus_one();
        let (ta, sta, _) = self.0.get(i.wrapping_sub(1));
        let (tb, stb, _) = self.0.get(i);

        let ta = ta.unwrap_or(U0d32::ZERO);
        let tb = tb.map_or(U32d32::ONE, U32d32::from);
        let t = (self.0.t - ta) / (tb - U32d32::from(ta));

        let s = sta.lerp(stb, t.into());
        self.0.next_sample(dt, s)
    }
}
