#[derive(Clone, Copy, Debug, Default)]
pub struct BitMap8(u8);

impl BitMap8 {
    pub fn get(&self, index: u8) -> bool {
        debug_assert!(index < 8);
        (self.0 >> index) & 1 != 0
    }

    pub fn set(&mut self, index: u8, value: bool) {
        debug_assert!(index < 8);
        self.0 &= !(1 << index);
        self.0 |= u8::from(value) << index;
    }
}
