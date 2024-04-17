use glam::{BVec3, IVec3, Vec3};

// Algorithm:
// - use IVec3 for position to avoid precision issues
// - Vec3 are in range [0.0; 1.0[
// - for each step:
//   - determine distance to edges, vec3(1) - fpos
//   - determine step factors, (vec3(1) - fpos) / fstep
//   - step with take minimum factor
//   - floor() -> ivec3()
//   - set signs
//   - add to position
//   - remap fpos to [0.0; 1.0[
//
// TODO consider replacing fp math with integer math
// to avoid conversions between int <-> fp
pub struct Ray {
    position: IVec3,
    fpos: Vec3,
    fstep: Vec3,
    sign: IVec3,
}

impl Ray {
    pub fn new(start: Vec3, direction: Vec3) -> Self {
        let signum = direction.signum();

        Self {
            position: start.floor().as_ivec3(),
            fpos: (start * signum).rem_euclid(Vec3::ONE),
            fstep: direction.abs().normalize(),
            sign: signum.as_ivec3(),
        }
    }

    pub fn current(&self) -> IVec3 {
        self.position
    }

    fn step(&mut self) -> BVec3 {
        let step_factor = (Vec3::ONE - self.fpos) / self.fstep;
        self.fpos += self.fstep * step_factor.min_element();
        // don't actually use floor(), but set based on max element
        // to work around precision issues
        let step = Vec3::splat(self.fpos.max_element()).cmpeq(self.fpos);
        // ensure we never go straight diagonal, as that can lead to some unintuitive interactions
        // (e.g. laser shot damaging blocks behind armor)
        let f = |i| BVec3::new(i == 0, i == 1, i == 2);
        let (x, y, z) = (f(0), f(1), f(2));
        let step = [x, x, y, x, z, x, y, x][step.bitmask() as usize];
        self.fpos = self.fpos - Vec3::from(step);
        self.position += IVec3::from(step) * self.sign;
        step
    }
}

impl Iterator for Ray {
    type Item = IVec3;

    fn next(&mut self) -> Option<Self::Item> {
        let pos = self.current();
        self.step();
        Some(pos)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn origin_along_x() {
        let mut ray = Ray::new(Vec3::ZERO, Vec3::X);
        assert_eq!(ray.next(), Some(IVec3::X * 0));
        assert_eq!(ray.next(), Some(IVec3::X * 1));
        assert_eq!(ray.next(), Some(IVec3::X * 2));
        assert_eq!(ray.next(), Some(IVec3::X * 3));
    }

    #[test]
    fn origin_along_xy() {
        let mut ray = Ray::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 0.0));
        assert_eq!(ray.next(), Some(IVec3::new(0, 0, 0)));
        assert_eq!(ray.next(), Some(IVec3::new(1, 0, 0)));
        assert_eq!(ray.next(), Some(IVec3::new(1, 1, 0)));
        assert_eq!(ray.next(), Some(IVec3::new(2, 1, 0)));
        assert_eq!(ray.next(), Some(IVec3::new(2, 2, 0)));
        assert_eq!(ray.next(), Some(IVec3::new(3, 2, 0)));
        assert_eq!(ray.next(), Some(IVec3::new(3, 3, 0)));
    }

    #[test]
    fn origin_along_xyz() {
        let mut ray = Ray::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(ray.next(), Some(IVec3::new(0, 0, 0)));
        assert_eq!(ray.next(), Some(IVec3::new(1, 0, 0)));
        assert_eq!(ray.next(), Some(IVec3::new(1, 1, 0)));
        assert_eq!(ray.next(), Some(IVec3::new(1, 1, 1)));
        assert_eq!(ray.next(), Some(IVec3::new(2, 1, 1)));
        assert_eq!(ray.next(), Some(IVec3::new(2, 2, 1)));
        assert_eq!(ray.next(), Some(IVec3::new(2, 2, 2)));
        assert_eq!(ray.next(), Some(IVec3::new(3, 2, 2)));
        assert_eq!(ray.next(), Some(IVec3::new(3, 3, 2)));
        assert_eq!(ray.next(), Some(IVec3::new(3, 3, 3)));
    }

    #[test]
    fn origin_along_negx() {
        let mut ray = Ray::new(Vec3::ZERO, -Vec3::X);
        assert_eq!(ray.next(), Some(IVec3::X * -0));
        assert_eq!(ray.next(), Some(IVec3::X * -1));
        assert_eq!(ray.next(), Some(IVec3::X * -2));
        assert_eq!(ray.next(), Some(IVec3::X * -3));
    }

    #[test]
    fn origin_along_xnegy() {
        let mut ray = Ray::new(Vec3::ZERO, Vec3::new(1.0, -1.0, 0.0));
        assert_eq!(ray.next(), Some(IVec3::new(0, -0, 0)));
        assert_eq!(ray.next(), Some(IVec3::new(1, -0, 0)));
        assert_eq!(ray.next(), Some(IVec3::new(1, -1, 0)));
        assert_eq!(ray.next(), Some(IVec3::new(2, -1, 0)));
        assert_eq!(ray.next(), Some(IVec3::new(2, -2, 0)));
        assert_eq!(ray.next(), Some(IVec3::new(3, -2, 0)));
        assert_eq!(ray.next(), Some(IVec3::new(3, -3, 0)));
    }

    #[test]
    fn one_along_2x1y() {
        let mut ray = Ray::new(Vec3::ONE, Vec3::new(2.0, 1.0, 0.0));
        assert_eq!(ray.next(), Some(IVec3::new(1, 1, 1)));
        assert_eq!(ray.next(), Some(IVec3::new(2, 1, 1)));
        assert_eq!(ray.next(), Some(IVec3::new(3, 1, 1)));
        assert_eq!(ray.next(), Some(IVec3::new(3, 2, 1)));
        assert_eq!(ray.next(), Some(IVec3::new(4, 2, 1)));
        assert_eq!(ray.next(), Some(IVec3::new(5, 2, 1)));
        assert_eq!(ray.next(), Some(IVec3::new(5, 3, 1)));
    }

    #[test]
    fn fppos_along_2x1y() {
        let mut ray = Ray::new(Vec3::ONE, Vec3::new(2.1, 1.0, 0.0));
        assert_eq!(ray.next(), Some(IVec3::new(1, 1, 1)));
        assert_eq!(ray.next(), Some(IVec3::new(2, 1, 1)));
        assert_eq!(ray.next(), Some(IVec3::new(3, 1, 1)));
        assert_eq!(ray.next(), Some(IVec3::new(3, 2, 1)));
        assert_eq!(ray.next(), Some(IVec3::new(4, 2, 1)));
        assert_eq!(ray.next(), Some(IVec3::new(5, 2, 1)));
        assert_eq!(ray.next(), Some(IVec3::new(5, 3, 1)));
    }
}
