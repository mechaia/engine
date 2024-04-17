use glam::{Quat, Vec2, Vec3, Vec3A};

pub mod gltf;

#[derive(Default)]
pub struct Collection {
    pub meshes: Vec<Mesh>,
    pub armatures: Vec<Armature>,
    pub models: Vec<Model>,
}

pub struct Mesh {
    pub indices: Vec<u32>,
    /// translation, normal, uv, joints, weights
    pub vertices: util::soa::Vec5<Vec3, Vec3, Vec2, [u16; 4], [f32; 4]>,
}

pub struct Armature {
    /// Tree structure of bones with transforms relative to parent
    ///
    /// - transform relative to parent
    /// - child count
    /// - output index: Mapping of bone to "joint", for use by vertices
    bones: util::soa::Vec3<Transform, u16, u16>,
    /// "Inverse" transform, from parent bone space to model space
    bone_parent_to_model: Box<[Transform]>,
    output_count: u16,
}

impl Armature {
    fn new(bones: util::soa::Vec3<Transform, u16, u16>) -> Self {
        Self {
            bone_parent_to_model: [].into(),
            output_count: bones
                .iter()
                .filter(|e| *e.1 != u16::MAX)
                .count()
                .try_into()
                .unwrap(),
            bones,
        }
        .calc_inv_transforms()
    }

    fn calc_inv_transforms(mut self) -> Self {
        debug_assert!(self.bone_parent_to_model.is_empty());
        let transforms = (0..self.bones.len())
            .map(|_| Transform::default())
            .collect::<Vec<_>>();

        let mut out = (0..self.output_count)
            .map(|_| Transform::default())
            .collect::<Box<_>>();

        let mut index = 0;
        while index < self.bones.len() {
            self.apply_rec(&transforms[index], &transforms, &mut index, &mut out);
        }

        for o in out.iter_mut() {
            *o = o.inverse();
        }

        self.bone_parent_to_model = out;
        self
    }

    pub fn apply(&self, transforms: &[Transform]) -> Box<[Transform]> {
        assert_eq!(transforms.len(), self.bones.len());

        let mut out = (0..self.output_count)
            .map(|_| Transform::default())
            .collect::<Box<_>>();

        let mut index = 0;
        while index < self.bones.len() {
            self.apply_rec(&transforms[index], transforms, &mut index, &mut out);
        }

        for (i, o) in out.iter_mut().enumerate() {
            *o = self.apply_direct(i, o);
        }

        out.into()
    }

    fn apply_rec(
        &self,
        cur: &Transform,
        transforms: &[Transform],
        index: &mut usize,
        out: &mut [Transform],
    ) {
        let (trf, &child_count, &output) = self.bones.get(*index).unwrap();

        if output != u16::MAX {
            out[usize::from(output)] = self.apply_direct(usize::from(output), cur);
        }
        let end = *index + usize::from(child_count);
        while *index < end {
            let cur = cur.apply_as_child(&transforms[*index]);
            self.apply_rec(&cur, transforms, index, out);
        }
    }

    pub fn apply_direct(&self, index: usize, transform: &Transform) -> Transform {
        transform.apply_as_child(&self.bone_parent_to_model[index])
    }
}

#[derive(Debug)]
pub struct Model {
    pub mesh_index: usize,
    pub armature_index: usize,
    pub name: Option<Box<str>>,
}

#[repr(align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct Transform {
    pub rotation: Quat,
    pub translation: Vec3A,
}

impl Transform {
    fn apply_as_child(&self, child: &Self) -> Self {
        let translation = (self.rotation * child.translation) + self.translation;
        let rotation = self.rotation * child.rotation;
        Self {
            rotation,
            translation,
        }
    }

    fn inverse(&self) -> Self {
        Self {
            rotation: self.rotation.inverse(),
            translation: -self.translation,
        }
    }
}
