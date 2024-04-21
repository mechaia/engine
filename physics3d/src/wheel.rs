use crate::{CastRayNormalResult, Physics, RigidBodyHandle, Transform};
use glam::{Quat, Vec3};

/// Collection of wheels on a single body.
///
/// Wheels must be grouped and processed together to keep physics as accurate as reasonably possible.
pub struct VehicleBody {
    wheels: util::Arena<WheelInstance>,
}

#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct WheelHandle(util::ArenaHandle);

struct WheelInstance {
    transform: Transform,
    radius: f32,
    static_friction: f32,
    dynamic_friction: f32,
    suspension: WheelSuspensionInstance,
    rotation: f32,
    angle: f32,
    torque: f32,
    brake: f32,
}

struct WheelSuspensionInstance {
    // TODO read up on suspension damping
    //
    // Might be useful to have a separate suspension helper structure
    cur_length: f32,
    max_length: f32,
    force_per_distance: f32,
    damp_per_velocity: f32,
}

struct WheelNow {
    pos: Vec3,
    dir: Vec3,
    ray: Option<CastRayNormalResult>,
    len: f32,
}

impl VehicleBody {
    pub fn new() -> Self {
        Self {
            wheels: Default::default(),
        }
    }

    pub fn add_wheel(&mut self, wheel: Wheel) -> WheelHandle {
        let h = self.wheels.insert(WheelInstance {
            transform: wheel.transform,
            radius: wheel.radius,
            static_friction: wheel.static_friction,
            dynamic_friction: wheel.dynamic_friction,
            suspension: WheelSuspensionInstance {
                cur_length: 0.0,
                max_length: wheel.suspension.max_length,
                force_per_distance: wheel.suspension.force_per_distance,
                damp_per_velocity: wheel.suspension.damp_per_velocity,
            },
            rotation: 0.0,
            angle: 0.0,
            torque: 0.0,
            brake: 0.0,
        });
        WheelHandle(h)
    }

    pub fn remove_wheel(&mut self, wheel: WheelHandle) {
        self.wheels.remove(wheel.0).expect("no wheel with handle");
    }

    pub fn set_wheel_angle(&mut self, wheel: WheelHandle, angle: f32) {
        self.wheels[wheel.0].angle = angle;
    }

    pub fn set_wheel_torque(&mut self, wheel: WheelHandle, torque: f32) {
        self.wheels[wheel.0].torque = torque;
    }

    pub fn set_wheel_brake(&mut self, wheel: WheelHandle, brake: f32) {
        self.wheels[wheel.0].brake = brake;
    }

    /// Returns transform for axle and tire.
    pub fn wheel_local_transform(&self, wheel: WheelHandle) -> (Transform, Transform) {
        let w = &self.wheels[wheel.0];
        let dir = w.transform.rotation * Vec3::NEG_Z;
        let axle = Transform {
            translation: dir * w.suspension.cur_length,
            rotation: Quat::from_rotation_z(w.angle),
        };
        let axle = w.transform.apply_to_transform(&axle);
        let tire = Transform {
            translation: axle.translation,
            rotation: axle.rotation * Quat::from_rotation_x(w.rotation),
        };
        (axle, tire)
    }

    pub fn apply(&mut self, physics: &mut Physics, body: RigidBodyHandle) {
        let now = self.collect_rays(physics, body);
        self.apply_suspension(physics, body, &now);
        self.apply_friction(physics, body, &now);

        for (w, n) in self.wheels.values_mut().zip(now) {
            if let Some(res) = n.ray {
                w.suspension.cur_length = (res.distance - w.radius).max(0.0);
            } else {
                w.suspension.cur_length = w.suspension.max_length;
            }
        }
    }

    fn collect_rays(&mut self, physics: &Physics, body: RigidBodyHandle) -> Vec<WheelNow> {
        let mut v = Vec::with_capacity(self.wheels.len());
        for w in self.wheels.values() {
            let trf = physics.rigid_body_transform(body);
            let start = trf.apply_to_translation(w.transform.translation);
            let dir = trf.apply_to_direction(w.transform.rotation * Vec3::NEG_Z);

            let ray = physics.cast_ray_normal(
                start,
                dir * (w.radius + w.suspension.max_length),
                Some(body),
            );
            v.push(WheelNow {
                pos: start,
                dir,
                ray,
                len: w.radius + w.suspension.cur_length,
            });
        }
        v
    }

    fn apply_suspension(&mut self, physics: &mut Physics, body: RigidBodyHandle, now: &[WheelNow]) {
        let dt = physics.time_delta();

        for (w, n) in self.wheels.values().zip(now) {
            let Some(res) = n.ray else { continue };
            if res.distance <= n.len {
                let point = n.pos + n.dir * res.distance;

                // spring
                let force = (w.suspension.max_length - w.suspension.cur_length)
                    * w.suspension.force_per_distance;

                // damp
                let velocity_at_point = physics.velocity_at_world_point(body, point);
                let velocity_along_suspension = -n.dir.dot(velocity_at_point);
                let damp = w.suspension.damp_per_velocity * velocity_along_suspension;
                let force = force - damp;

                physics.apply_impulse_at(body, point, -(force * dt) * n.dir);
            }
        }
    }

    fn apply_friction(&mut self, physics: &mut Physics, body: RigidBodyHandle, now: &[WheelNow]) {
        let trf = physics.rigid_body_transform(body);
        let dt = physics.time_delta();

        let mass = physics.rigid_body_mass(body);
        // FIXME funky shit going on here
        //let mass_per_wheel = mass / now.iter().filter(|r| r.ray.is_some()).count() as f32;
        let mass_per_wheel = mass / self.wheels.len() as f32;

        for (w, n) in self.wheels.values_mut().zip(now) {
            let Some(res) = n.ray else { continue };
            if res.distance <= n.len {
                let point = n.pos + n.dir * res.distance;

                let trf_l = Transform {
                    translation: w.transform.translation,
                    rotation: w.transform.rotation * Quat::from_rotation_z(w.angle),
                };
                let trf = trf.apply_to_transform(&trf_l);

                let vehicle_along = trf.apply_to_direction(Vec3::Y);
                let vehicle_across = trf.apply_to_direction(Vec3::X);

                // FIXME I think this is correct, but I need pen and paper to work this out properly
                // FIXME probable cause of NaN errors
                let vehicle_across_surface = vehicle_along.cross(res.normal).normalize();

                let velocity_at_point = physics.velocity_at_world_point(body, point);

                let velocity_along_wheel = vehicle_along.dot(velocity_at_point);
                let velocity_across_wheel = vehicle_across.dot(velocity_at_point);

                let velocity_across_wheel_surface = vehicle_across_surface.dot(velocity_at_point);

                // FIXME account for mass when applying friction
                // Also requires accounting for friction applied by other wheels
                let force = w.torque * w.radius;
                let mut impulse = w.static_friction
                    * mass_per_wheel
                    * Vec3::new(
                        //velocity_across_wheel,
                        velocity_across_wheel_surface,
                        velocity_along_wheel * w.brake,
                        0.0,
                    );
                impulse.y += force * dt;

                physics.apply_impulse_at(body, point, trf.apply_to_direction(-impulse));

                // FIXME account for gravity
                // and other forces
                let velocity_at_point = physics.velocity_at_world_point(body, point);
                let velocity_along_wheel = vehicle_along.dot(velocity_at_point);
                let angular_velocity = velocity_along_wheel / w.radius;
                if angular_velocity.is_nan() {
                    dbg!(vehicle_along, velocity_at_point, w.radius, point);
                }
                w.rotation -= angular_velocity * dt;
            }
        }
    }
}

/// Singular wheel on a spring
pub struct Wheel {
    pub transform: Transform,
    pub radius: f32,
    pub static_friction: f32,
    pub dynamic_friction: f32,
    pub suspension: WheelSuspension,
}

pub struct WheelSuspension {
    pub max_length: f32,
    pub force_per_distance: f32,
    pub damp_per_velocity: f32,
}

/// Two wheels on an axle
pub struct AxleWheels {}

/// Tank tread
pub struct TankTread {}
