use glam::{Quat, Vec2, Vec3, Vec3A};

pub mod gltf;

#[derive(Default)]
pub struct Collection {
    pub meshes: Vec<Mesh>,
    pub armatures: Vec<Armature>,
    pub models: Vec<Model>,
    /// List of scenes, all starting from a single root node.
    pub scenes: Vec<Node>,
}

pub struct Model {
    pub mesh_index: usize,
    pub armature_index: usize,
}

#[repr(align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct Transform {
    pub rotation: Quat,
    pub translation: Vec3A,
}

pub struct Mesh {
    pub indices: Vec<u32>,
    /// translation, normal, uv, joints, weights
    pub vertices: util::soa::Vec5<Vec3, Vec3, Vec2, [u16; 4], [f32; 4]>,
    /// How many transforms to push to render correctly.
    pub transform_count: u16,
}

pub struct Armature {
    /// Tree structure of bones with transforms relative to parent
    ///
    /// - transform relative to parent
    /// - child count
    /// - output index: Mapping of bone to "joint", for use by vertices
    bones: util::soa::Vec3<Transform, u16, u16>,
    /// "Inverse" transform, from model space to local bone space
    model_to_local_bone: Box<[Transform]>,
    output_count: u16,
}

pub enum Node {
    /// Parent of one or more nodes.
    Parent {
        children: Box<[(Transform, Node)]>,
        properties: Properties,
    },
    /// Leaf node, with index to model.
    Leaf {
        model: usize,
        properties: Properties,
    },
}

impl Node {
    pub fn properties(&self) -> &Properties {
        match self {
            Self::Parent { properties, .. } => properties,
            Self::Leaf { properties, .. } => properties,
        }
    }

    pub fn descendants(&self) -> impl Iterator<Item = &Node> + '_ {
        struct Iter<'a> {
            stack: Vec<&'a [(Transform, Node)]>,
        }
        impl<'a> Iterator for Iter<'a> {
            type Item = &'a Node;

            fn next(&mut self) -> Option<Self::Item> {
                let (n, l) = loop {
                    let l = self.stack.pop()?;
                    let Some((n, l)) = l.split_first() else {
                        continue;
                    };
                    break (n, l);
                };
                self.stack.push(l);
                match &n.1 {
                    Node::Parent { children, .. } => self.stack.push(&**children),
                    Node::Leaf { .. } => {}
                }
                Some(&n.1)
            }
        }
        let stack = match self {
            Self::Parent { children, .. } => Vec::from([&**children]),
            Self::Leaf { .. } => Vec::new(),
        };
        Iter { stack }
    }
}

#[derive(Default)]
pub struct Properties {
    pub name: Option<Box<str>>,
    pub custom: Box<[(Box<str>, CustomValue)]>,
}

pub enum CustomValue {
    Str(Box<str>),
}

impl Armature {
    /// Transform, child count, output
    fn new(bones: util::soa::Vec3<Transform, u16, u16>) -> Self {
        Self {
            model_to_local_bone: [].into(),
            output_count: bones
                .iter()
                .filter(|e| *e.2 != u16::MAX)
                .count()
                .try_into()
                .unwrap(),
            bones,
        }
        .calc_inv_transforms()
    }

    fn calc_inv_transforms(mut self) -> Self {
        debug_assert!(self.model_to_local_bone.is_empty());

        self.model_to_local_bone = (0..self.output_count)
            .map(|_| Transform::default())
            .collect::<Box<_>>();

        let mut index = 0;
        while index < self.bones.len() {
            self.calc_inv_transforms_rec(&Transform::IDENTITY, &mut index);
        }

        dbg!(&self.model_to_local_bone);

        #[cfg(debug_assertions)]
        {
            let trfs = (0..self.output_count)
                .map(|_| Transform::IDENTITY)
                .collect::<Vec<_>>();
            let out = self.apply(&Transform::IDENTITY, &trfs, true);
            for t in out.into_vec() {
                assert!(t.translation.x.abs() < 1e-6);
                assert!(t.translation.y.abs() < 1e-6);
                assert!(t.translation.z.abs() < 1e-6);
                assert!(t.rotation.x.abs() < 1e-6);
                assert!(t.rotation.y.abs() < 1e-6);
                assert!(t.rotation.z.abs() < 1e-6);
                assert!((t.rotation.w - 1.0).abs() < 1e-6);
            }
        }

        self
    }

    fn calc_inv_transforms_rec(&mut self, parent_trf: &Transform, index: &mut usize) {
        let (trf, &child_count, &output) = self.bones.get(*index).unwrap();
        *index += 1;

        let trf = parent_trf.apply_to_transform(trf);

        if output != u16::MAX {
            self.model_to_local_bone[usize::from(output)] = trf.inverse();
            //self.model_to_local_bone[usize::from(output)] = trf;
        }

        let end = *index + usize::from(child_count);
        while *index < end {
            self.calc_inv_transforms_rec(&trf, index);
        }
    }

    pub fn apply(&self, base_transform: &Transform, transforms: &[Transform], inverse: bool) -> Box<[Transform]> {
        assert_eq!(transforms.len(), self.bones.len());

        let mut out = (0..self.output_count)
            .map(|_| Transform::default())
            .collect::<Box<_>>();

        let mut index = 0;
        while index < self.bones.len() {
            self.apply_rec(base_transform, &mut index, transforms, &mut out, inverse);
        }

        out.into()
    }

    fn apply_rec(
        &self,
        parent_trf: &Transform,
        index: &mut usize,
        transforms: &[Transform],
        out: &mut [Transform],
        inverse: bool,
    ) {
        let (trf, &child_count, &output) = self.bones.get(*index).unwrap();
        let trf = parent_trf.apply_to_transform(&trf);
        let trf = trf.apply_to_transform(&transforms[usize::from(*index)]);
        *index += 1;

        if output != u16::MAX {
            out[usize::from(output)] = if inverse {
                self.apply_direct(usize::from(output), &trf)
            } else {
                trf
            };
        }

        let end = *index + usize::from(child_count);
        while *index < end {
            self.apply_rec(&trf, index, transforms, out, inverse);
        }
    }

    pub fn apply_direct(&self, index: usize, transform: &Transform) -> Transform {
        transform.apply_to_transform(&self.model_to_local_bone[index])
    }
}

impl Transform {
    pub const IDENTITY: Self = Self {
        rotation: Quat::IDENTITY,
        translation: Vec3A::ZERO,
    };

    fn apply_to_transform(&self, child: &Self) -> Self {
        let translation = self.translation + (self.rotation * child.translation);
        let rotation = self.rotation * child.rotation;
        Self {
            rotation,
            translation,
        }
    }

    fn inverse(&self) -> Self {
        let rotation = self.rotation.inverse();
        let translation = rotation * -self.translation;
        Self {
            rotation,
            translation,
        }
    }

    fn is_identity(&self) -> bool {
        self.translation == Vec3A::ONE && self.rotation == Quat::IDENTITY
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn transform_invert_transform() {
        let trf = Transform {
            translation: Vec3A::new(5.0, 4.0, 3.0),
            rotation: Quat::from_axis_angle(Vec3::new(1.0, 3.0, 2.0).normalize(), 1.0),
        };
        let trf2 = trf.inverse().apply_to_transform(&trf);
        assert_eq!(trf2.translation, Vec3A::ZERO);
        assert_eq!(trf2.rotation, Quat::IDENTITY);
    }

    #[test]
    fn transform_invert_translate() {
        let trf = Transform {
            translation: Vec3A::new(5.0, 4.0, 3.0),
            rotation: Quat::from_axis_angle(Vec3::new(1.0, 3.0, 2.0).normalize(), 1.0),
        };
        let trf_inv = trf.inverse();

        let p = Vec3A::new(1.0, 2.0, 3.0);
        let q = trf.translation + (trf.rotation * p);
        let pp = trf_inv.translation + (trf_inv.rotation * q);

        assert!((p.x - pp.x).abs() < 1e-6);
        assert!((p.y - pp.y).abs() < 1e-6);
        assert!((p.z - pp.z).abs() < 1e-6);
    }
}
