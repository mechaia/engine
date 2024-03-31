#[derive(Default)]
pub struct ConnectionMap(u8);

impl ConnectionMap {
    fn bit(&self, index: u8) -> bool {
        (self.0 >> index) & 1 != 0
    }

    pub fn nx(&self) -> bool {
        self.bit(0)
    }

    pub fn px(&self) -> bool {
        self.bit(1)
    }

    pub fn ny(&self) -> bool {
        self.bit(2)
    }

    pub fn py(&self) -> bool {
        self.bit(3)
    }

    pub fn nz(&self) -> bool {
        self.bit(4)
    }

    pub fn pz(&self) -> bool {
        self.bit(5)
    }
}
