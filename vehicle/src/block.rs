use super::{BlockId, Map};

type MeshId = u32;

#[derive(Clone, Copy, Default)]
pub struct Connections(util::BitMap8);

impl Connections {
    pub fn px(&self) -> bool {
        self.0.get(0)
    }

    pub fn nx(&self) -> bool {
        self.0.get(1)
    }

    pub fn py(&self) -> bool {
        self.0.get(2)
    }

    pub fn ny(&self) -> bool {
        self.0.get(3)
    }

    pub fn pz(&self) -> bool {
        self.0.get(4)
    }

    pub fn nz(&self) -> bool {
        self.0.get(5)
    }

    pub fn set_px(&mut self, value: bool) {
        self.0.set(0, value)
    }

    pub fn set_nx(&mut self, value: bool) {
        self.0.set(1, value)
    }

    pub fn set_py(&mut self, value: bool) {
        self.0.set(2, value)
    }

    pub fn set_ny(&mut self, value: bool) {
        self.0.set(3, value)
    }

    pub fn set_pz(&mut self, value: bool) {
        self.0.set(4, value)
    }

    pub fn set_nz(&mut self, value: bool) {
        self.0.set(5, value)
    }
}

pub struct Block {
    connections: Connections,
    mesh_id: MeshId,
}

pub struct BlockDB {
    connections: Vec<Connections>,
    mesh_id: Vec<MeshId>,
    name_to_id: Map<Box<str>, BlockId>,
}

impl Default for BlockDB {
    fn default() -> Self {
        Self {
            connections: Vec::from([Connections::default()]),
            mesh_id: Default::default(),
            name_to_id: Default::default(),
        }
    }
}

impl BlockDB {
    pub fn add(&mut self, name: String, block: Block) {
        let id = self.connections.len().try_into().unwrap();
        let id = BlockId::new(id).unwrap();
        self.connections.push(block.connections);
        self.mesh_id.push(block.mesh_id);
        self.name_to_id.insert(name.into(), id);
    }

    pub(crate) fn connections(&self, id: u16) -> Connections {
        self.connections[usize::from(id)]
    }
}
