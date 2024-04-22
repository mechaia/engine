use glam::{Quat, Vec2, Vec3, Vec3A};

use crate::Transform;
use core::iter;
use std::collections::HashMap;

struct Builder<'a> {
    gltf: &'a ::gltf::Gltf,
    bin: &'a [u8],
    nodes: Vec<Node>,
    collection: crate::Collection,
    mesh_map: HashMap<usize, usize>,
    armature_map: HashMap<usize, usize>,
}

#[derive(Debug)]
struct Node {
    transform: Transform,
    parent: usize,
}

pub fn from_glb_slice(data: &[u8]) -> crate::Collection {
    let glb = ::gltf::Glb::from_slice(data).unwrap();
    let gltf = ::gltf::Gltf::from_slice(&glb.json).unwrap();
    let bin = glb.bin.as_deref().unwrap();

    let mut builder = Builder {
        gltf: &gltf,
        bin,
        nodes: load_nodes(&gltf),
        collection: Default::default(),
        mesh_map: Default::default(),
        armature_map: Default::default(),
    };

    for scene in gltf.scenes() {
        let nodes = scene
            .nodes()
            .map(|n| parse_node_tree(&mut builder, n))
            .collect::<Vec<_>>();
        assert!(!nodes.is_empty(), "scene with no nodes");
        let node = if nodes.len() == 1 && nodes[0].0.is_identity() {
            nodes.into_iter().next().unwrap().1
        } else {
            crate::Node::Parent {
                children: nodes.into(),
                properties: Default::default(),
            }
        };
        builder.collection.scenes.push(node);
    }

    builder.collection
}

fn parse_node_tree(builder: &mut Builder<'_>, node: ::gltf::Node<'_>) -> (Transform, crate::Node) {
    let model = builder.make_model(&node);
    let mut children = node
        .children()
        .map(|node| parse_node_tree(builder, node))
        .collect::<Vec<_>>();

    let mut properties = crate::Properties::default();
    properties.name = node.name().map(|s| s.into());

    let snode = if children.is_empty() {
        crate::Node::Leaf {
            model: model.unwrap_or(usize::MAX),
            properties,
        }
    } else {
        if let Some(model) = model {
            children.push((
                Transform::IDENTITY,
                crate::Node::Leaf {
                    model,
                    properties: Default::default(),
                },
            ));
        }
        crate::Node::Parent {
            children: children.into(),
            properties,
        }
    };
    (to_transform(node.transform()), snode)
}

/// Load all nodes so we have proper fucking backlinks
fn load_nodes(gltf: &::gltf::Gltf) -> Vec<Node> {
    let mut nodes = Vec::new();

    // fucking retarded garbage fucking format what the fucking fuck
    for node in gltf.nodes() {
        let (translation, rotation, scale) = node.transform().decomposed();
        //assert_eq!(scale, [1.0; 3], "non-identity scale");
        nodes.push(Node {
            transform: Transform {
                translation: Vec3A::from_array(translation),
                rotation: Quat::from_array(rotation),
            },
            parent: usize::MAX,
        });
    }

    for node in gltf.nodes() {
        for child in node.children() {
            assert_eq!(
                nodes[child.index()].parent,
                usize::MAX,
                "child with multiple parents"
            );
            nodes[child.index()].parent = node.index();
        }
    }

    nodes
}

impl Builder<'_> {
    fn make_model(&mut self, node: &::gltf::Node<'_>) -> Option<usize> {
        let mesh_index = node.mesh().map_or(usize::MAX, |m| self.load_mesh(m));
        let armature_index = node.skin().map_or(usize::MAX, |m| self.load_armature(m));

        if mesh_index == usize::MAX && armature_index == usize::MAX {
            return None;
        }

        let i = self.collection.models.len();
        self.collection.models.push(crate::Model {
            mesh_index,
            armature_index,
        });
        Some(i)
    }

    fn load_mesh(&mut self, mesh: ::gltf::Mesh<'_>) -> usize {
        if let Some(i) = self.mesh_map.get(&mesh.index()) {
            return *i;
        }

        assert_eq!(mesh.primitives().count(), 1, "todo: multi-primitive mesh");
        let p = mesh.primitives().next().unwrap();
        let r = p.reader(source_bin(self.bin));

        let indices = r.read_indices().unwrap().into_u32().collect();

        let vertices = r
            .read_positions()
            .unwrap()
            .map(Vec3::from_array)
            .zip(
                r.read_normals()
                    .unwrap()
                    .map(Vec3::from_array)
                    .chain(iter::repeat(Vec3::Z)),
            )
            .zip(
                r.read_tex_coords(0)
                    .into_iter()
                    .flat_map(|v| v.into_f32())
                    .map(Vec2::from_array)
                    .chain(iter::repeat(Vec2::ZERO)),
            )
            .zip(
                r.read_joints(0)
                    .into_iter()
                    .flat_map(|v| v.into_u16())
                    .chain(iter::repeat([0; 4])),
            )
            .zip(
                r.read_weights(0)
                    .into_iter()
                    .flat_map(|v| v.into_f32())
                    .chain(iter::repeat([1.0, 0.0, 0.0, 0.0])),
            )
            .map(|((((a, b), c), d), e)| (a, b, c, d, e))
            .collect::<util::soa::Vec5<_, _, _, _, _>>();

        let transform_count = *vertices
            .as_slices()
            .3
            .iter()
            .flat_map(|v| v)
            .max()
            .unwrap_or(&0)
            + 1;

        let m = crate::Mesh {
            indices,
            vertices,
            transform_count,
        };

        let len = m.vertices.len();
        assert!(m
            .indices
            .iter()
            .all(|i| usize::try_from(*i).unwrap() <= len));

        let i = self.collection.meshes.len();
        self.collection.meshes.push(m);
        self.mesh_map.insert(mesh.index(), i);
        i
    }

    fn load_armature(&mut self, skin: ::gltf::Skin<'_>) -> usize {
        if let Some(i) = self.armature_map.get(&skin.index()) {
            return *i;
        }

        let r = skin.reader(source_bin(self.bin));
        dbg!(r.read_inverse_bind_matrices().unwrap().map(|m| glam::Mat4::from_cols_array_2d(&m)).map(|m| m.to_scale_rotation_translation())
            .collect::<Vec<_>>());

        let armature = build_armature(self.gltf, &skin, &self.nodes);

        let mut outputs = (0..armature.bones.len())
            .map(|_| u16::MAX)
            .collect::<Vec<_>>();

        for (k, &i) in armature.joint_index_to_bone.iter().enumerate() {
            outputs[i] = k.try_into().unwrap();
        }

        let bones = armature
            .bones
            .into_iter()
            .zip(armature.descendant_count)
            .zip(outputs)
            .map(|((a, b), c)| (a, b, c))
            .collect();

        let armature = crate::Armature::new(bones);

        let i = self.collection.armatures.len();
        self.collection.armatures.push(armature);
        self.armature_map.insert(skin.index(), i);
        i
    }
}

// FIXME all of this is garbage
// How fucking difficult can it be to extract an armature?

#[derive(Debug)]
struct Armature {
    bones: Vec<Transform>,
    descendant_count: Vec<u16>,
    joint_index_to_bone: Box<[usize]>,
    joints: Vec<usize>,
}

// GLTF IS FUCKING RETARDED
fn build_armature(gltf: &::gltf::Gltf, armature: &::gltf::Skin<'_>, nodes: &[Node]) -> Armature {
    let joints = armature.joints().map(|n| n.index()).collect::<Vec<_>>();

    // collect ancestors
    let chains = joints
        .iter()
        .map(|i| collect_chains(nodes, *i))
        .collect::<Vec<_>>();

    // skip common prefix
    let mut prefix = 0;
    while chains
        .iter()
        .all(|v| chains[0].get(prefix) == v.get(prefix))
    {
        prefix += 1;
    }
    let prefix = prefix
        .checked_sub(1)
        .expect("no common root, what the fuck?");

    // build armature from common root
    // map joints to bones while at it
    let mut armature = Armature {
        bones: Vec::new(),
        descendant_count: Vec::new(),
        joint_index_to_bone: armature.joints().map(|_| usize::MAX).collect(),
        joints,
    };

    let root = gltf.nodes().skip(chains[0][prefix]).next().unwrap();
    collect_bones(&mut armature, nodes, root);

    armature
}

fn collect_chains(nodes: &[Node], mut start: usize) -> Vec<usize> {
    let mut chains = Vec::new();
    while nodes[start].parent != usize::MAX {
        chains.push(start);
        start = nodes[start].parent;
    }
    chains.push(start);
    chains.reverse();
    chains
}

fn collect_bones(armature: &mut Armature, nodes: &[Node], node: ::gltf::Node<'_>) {
    if !armature.joints.contains(&node.index()) {
        return;
    }

    let start = armature.descendant_count.len();
    armature.descendant_count.push(u16::MAX);
    armature.bones.push(to_transform(node.transform()));
    for n in node.children() {
        collect_bones(armature, nodes, n);
    }
    let end = armature.descendant_count.len();
    armature.descendant_count[start] = u16::try_from(end - start - 1).unwrap();

    if let Some(i) = armature.joints.iter().position(|n| *n == node.index()) {
        armature.joint_index_to_bone[i] = start;
    }
}

fn repeat<T: Default>(count: usize) -> Vec<T> {
    iter::repeat_with(T::default).take(count).collect()
}

fn repeat2<T: Clone>(count: usize, value: T) -> Vec<T> {
    iter::repeat_with(move || value.clone())
        .take(count)
        .collect()
}

fn source_bin<'a>(bin: &'a [u8]) -> impl Fn(::gltf::Buffer<'_>) -> Option<&'a [u8]> + Clone + 'a {
    move |buffer| {
        assert!(matches!(buffer.source(), ::gltf::buffer::Source::Bin));
        Some(bin)
    }
}

fn to_transform(transform: ::gltf::scene::Transform) -> Transform {
    let (translation, rotation, scale) = transform.decomposed();
    assert!(scale.iter().all(|v| (v - 1.0).abs() < 1e-6), "non-identity scale {scale:?}");
    Transform {
        rotation: Quat::from_array(rotation),
        translation: Vec3A::from_array(translation),
    }
}
