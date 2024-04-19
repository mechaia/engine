pub mod wheel;

use glam::{Quat, Vec3};
use rapier3d::{
    dynamics::{
        self, CCDSolver, ImpulseJointSet, IntegrationParameters, IslandManager, MultibodyJointSet,
        RigidBody, RigidBodyBuilder, RigidBodySet,
    },
    geometry::{
        self, BroadPhase, Collider, ColliderBuilder, ColliderSet, NarrowPhase, Ray, SharedShape,
    },
    math::{Isometry, Point, Translation, Vector},
    na::{Quaternion, Unit},
    pipeline::{PhysicsPipeline, QueryFilter, QueryFilterFlags, QueryPipeline},
};
use std::num::NonZeroU32;
use std::ops::Index;

pub struct Physics {
    physics_pipeline: PhysicsPipeline,
    gravity: Vec3,
    integration_parameters: IntegrationParameters,
    island_manager: IslandManager,
    broad_phase: BroadPhase,
    narrow_phase: NarrowPhase,
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,
    physics_hooks: (),
    event_handler: (),
    shapes: util::Arena<SharedShape>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RigidBodyHandle(dynamics::RigidBodyHandle);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColliderHandle(geometry::ColliderHandle);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SensorHandle(geometry::ColliderHandle);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShapeHandle(util::ArenaHandle);

#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub rotation: Quat,
    pub translation: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct ColliderProperties {
    pub local_transform: Transform,
    pub friction: f32,
    pub bounciness: f32,
    pub user_data: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct SensorProperties {
    pub local_transform: Transform,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SetRigidBodyProperties {
    pub world_transform: Option<Transform>,
    pub linear_velocity: Option<Vec3>,
    pub angular_velocity: Option<Vec3>,
    pub enable_ccd: Option<bool>,
}

#[derive(Clone, Copy, Debug)]
pub struct CastRayResult {
    pub distance: f32,
    pub user_data: u32,
}

impl Physics {
    pub fn new() -> Self {
        Self {
            gravity: Vec3::NEG_Z * 9.81,
            integration_parameters: IntegrationParameters {
                dt: 1.0 / 60.0,
                ..Default::default()
            },
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            physics_hooks: (),
            event_handler: (),
            shapes: Default::default(),
        }
    }

    pub fn set_time_delta(&mut self, dt: f32) {
        self.integration_parameters.dt = dt;
    }

    pub fn time_delta(&self) -> f32 {
        self.integration_parameters.dt
    }

    pub fn set_gravity(&mut self, acceleration: Vec3) {
        self.gravity = acceleration;
    }

    pub fn step(&mut self) {
        self.physics_pipeline.step(
            &glam_to_vector(self.gravity),
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &self.physics_hooks,
            &self.event_handler,
        );
        self.query_pipeline
            .update(&self.rigid_body_set, &self.collider_set);
    }

    fn insert_rigid_body(&mut self, body: RigidBody) -> RigidBodyHandle {
        RigidBodyHandle(self.rigid_body_set.insert(body))
    }

    pub fn make_fixed_rigid_body(&mut self) -> RigidBodyHandle {
        self.insert_rigid_body(RigidBodyBuilder::fixed().build())
    }

    pub fn make_dynamic_rigid_body(&mut self) -> RigidBodyHandle {
        self.insert_rigid_body(RigidBodyBuilder::dynamic().build())
    }

    pub fn make_kinematic_position_rigid_body(&mut self) -> RigidBodyHandle {
        self.insert_rigid_body(RigidBodyBuilder::kinematic_position_based().build())
    }

    pub fn make_kinematic_velocity_rigid_body(&mut self) -> RigidBodyHandle {
        self.insert_rigid_body(RigidBodyBuilder::kinematic_velocity_based().build())
    }

    fn insert_shape(&mut self, shape: SharedShape) -> ShapeHandle {
        ShapeHandle(self.shapes.insert(shape))
    }

    pub fn make_cuboid_shape(&mut self, half_extents: Vec3) -> ShapeHandle {
        self.insert_shape(SharedShape::cuboid(
            half_extents.x,
            half_extents.y,
            half_extents.z,
        ))
    }

    pub fn make_sphere_shape(&mut self, radius: f32) -> ShapeHandle {
        self.insert_shape(SharedShape::ball(radius))
    }

    pub fn make_halfspace_shape(&mut self, normal: Vec3) -> ShapeHandle {
        self.insert_shape(SharedShape::halfspace(glam_to_unit_vector(normal)))
    }

    pub fn make_compound_shape(&mut self, shapes: &[(Transform, ShapeHandle)]) -> ShapeHandle {
        assert!(!shapes.is_empty());
        let mut nshapes = Vec::with_capacity(shapes.len());
        for (tr, sh) in shapes {
            nshapes.push((transform_to_isometry(*tr), self.shapes[sh.0].clone()));
        }
        self.insert_shape(SharedShape::compound(nshapes))
    }

    pub fn add_collider(
        &mut self,
        body: RigidBodyHandle,
        shape: ShapeHandle,
        properties: &ColliderProperties,
    ) -> ColliderHandle {
        let shape = self.shapes[shape.0].clone();
        let collider = ColliderBuilder::new(shape)
            .sensor(false)
            .position(transform_to_isometry(properties.local_transform))
            .friction(properties.friction)
            .restitution(properties.bounciness);
        let h = self
            .collider_set
            .insert_with_parent(collider, body.0, &mut self.rigid_body_set);
        ColliderHandle(h)
    }

    pub fn add_sensor(
        &mut self,
        body: RigidBodyHandle,
        shape: ShapeHandle,
        properties: &SensorProperties,
    ) -> SensorHandle {
        let shape = self.shapes[shape.0].clone();
        let collider = ColliderBuilder::new(shape)
            .sensor(true)
            .position(transform_to_isometry(properties.local_transform));
        let h = self
            .collider_set
            .insert_with_parent(collider, body.0, &mut self.rigid_body_set);
        SensorHandle(h)
    }

    pub fn remove_collider(&mut self, body: RigidBodyHandle, collider: ColliderHandle) {
        self.collider_set.remove(
            collider.0,
            &mut self.island_manager,
            &mut self.rigid_body_set,
            true,
        );
    }

    pub fn rigid_body_mass(&self, body: RigidBodyHandle) -> f32 {
        self.rigid_body_set[body.0].mass()
    }

    pub fn set_rigid_body_properties(
        &mut self,
        body: RigidBodyHandle,
        properties: &SetRigidBodyProperties,
    ) {
        if let Some(v) = properties.world_transform {
            self.rigid_body_set[body.0].set_position(transform_to_isometry(v), true);
        }
        if let Some(v) = properties.linear_velocity {
            self.rigid_body_set[body.0].set_linvel(glam_to_vector(v), true);
        }
        if let Some(v) = properties.angular_velocity {
            self.rigid_body_set[body.0].set_angvel(glam_to_vector(v), true);
        }
        if let Some(v) = properties.enable_ccd {
            self.rigid_body_set[body.0].enable_ccd(v);
        }
    }

    // FIXME we should discourage direct queries
    pub fn rigid_body_transform(&self, body: RigidBodyHandle) -> Transform {
        isometry_to_transform(*self.rigid_body_set[body.0].position())
    }

    pub fn apply_impulse(&mut self, body: RigidBodyHandle, impulse: Vec3) {
        self.rigid_body_set[body.0].apply_impulse(glam_to_vector(impulse), true);
    }

    pub fn apply_torque_impulse(&mut self, body: RigidBodyHandle, torque_impulse: Vec3) {
        self.rigid_body_set[body.0].apply_torque_impulse(glam_to_vector(torque_impulse), true);
    }

    pub fn apply_impulse_at(&mut self, body: RigidBodyHandle, point: Vec3, impulse: Vec3) {
        self.rigid_body_set[body.0].apply_impulse_at_point(
            glam_to_vector(impulse),
            glam_to_point(point),
            true,
        );
    }

    pub fn add_force(&mut self, body: RigidBodyHandle, force: Vec3) {
        self.rigid_body_set[body.0].add_force(glam_to_vector(force), true);
    }

    pub fn add_torque_force(&mut self, body: RigidBodyHandle, torque_force: Vec3) {
        self.rigid_body_set[body.0].add_torque(glam_to_vector(torque_force), true);
    }

    pub fn add_force_at(&mut self, body: RigidBodyHandle, point: Vec3, force: Vec3) {
        self.rigid_body_set[body.0].add_force_at_point(
            glam_to_vector(force),
            glam_to_point(point),
            true,
        );
    }

    pub fn cast_ray(
        &self,
        start: Vec3,
        length: Vec3,
        ignore_body: Option<RigidBodyHandle>,
    ) -> Option<CastRayResult> {
        let max_toi = length.length();
        let ray = Ray {
            origin: glam_to_point(start),
            // from docs:
            //   max_toi: the maximum time-of-impact that can be reported by this cast.
            //   This effectively limits the length of the ray to ray.dir.norm() * max_toi. Use Real::MAX for an unbounded ray.
            // safe to assume we don't need to normalize the direction
            dir: glam_to_vector(length.normalize()),
        };
        let filter = QueryFilter {
            flags: QueryFilterFlags::empty(),
            groups: None,
            exclude_collider: None,
            exclude_rigid_body: ignore_body.map(|h| h.0),
            predicate: None,
        };
        self.query_pipeline
            .cast_ray(
                &self.rigid_body_set,
                &self.collider_set,
                &ray,
                max_toi,
                true,
                filter,
            )
            .map(|(collider, distance)| CastRayResult {
                distance,
                user_data: self.collider_set[collider].user_data as u32,
            })
    }

    pub fn velocity_at_world_point(&self, body: RigidBodyHandle, point: Vec3) -> Vec3 {
        vector_to_glam(self.rigid_body_set[body.0].velocity_at_point(&glam_to_point(point)))
    }
}

fn glam_to_point(v: Vec3) -> Point<f32> {
    Point::<f32>::new(v.x, v.y, v.z)
}

fn glam_to_translation(v: Vec3) -> Translation<f32> {
    Translation::<f32>::new(v.x, v.y, v.z)
}

fn glam_to_vector(v: Vec3) -> Vector<f32> {
    Vector::<f32>::new(v.x, v.y, v.z)
}

fn glam_to_unit_vector(v: Vec3) -> Unit<Vector<f32>> {
    Unit::new_unchecked(Vector::<f32>::new(v.x, v.y, v.z))
}

fn glam_to_rotation(q: Quat) -> Unit<Quaternion<f32>> {
    Unit::new_unchecked(Quaternion::new(q.x, q.y, q.z, q.w))
}

fn transform_to_isometry(tr: Transform) -> Isometry<f32> {
    Isometry {
        rotation: glam_to_rotation(tr.rotation),
        translation: glam_to_translation(tr.translation),
    }
}

fn vector_to_glam(v: Vector<f32>) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

fn translation_to_glam(v: Translation<f32>) -> Vec3 {
    vector_to_glam(v.vector)
}

fn rotation_to_glam(v: Unit<Quaternion<f32>>) -> Quat {
    Quat::from_xyzw(v.i, v.j, v.k, v.w)
}

fn isometry_to_transform(iso: Isometry<f32>) -> Transform {
    Transform {
        translation: translation_to_glam(iso.translation),
        rotation: rotation_to_glam(iso.rotation),
    }
}

impl Transform {
    pub fn apply_to_translation(&self, translation: Vec3) -> Vec3 {
        self.translation + (self.rotation * translation)
    }

    pub fn apply_to_direction(&self, direction: Vec3) -> Vec3 {
        self.rotation * direction
    }

    pub fn apply_to_direction_inv(&self, direction: Vec3) -> Vec3 {
        self.rotation.inverse() * direction
    }

    pub fn apply_to_transform(&self, transform: &Self) -> Self {
        Self {
            translation: self.translation + self.rotation * transform.translation,
            rotation: self.rotation * transform.rotation,
        }
    }
}

impl From<Transform> for util::Transform {
    fn from(value: Transform) -> Self {
        Self {
            translation: value.translation,
            rotation: value.rotation,
            scale: 1.0,
        }
    }
}
