pub mod bdf;

use glam::{U16Vec2, UVec2};
use std::collections::HashMap;

/// A texture atlas of a font.
///
/// Maps characters to UV rects.
pub struct FontMap {
    dim: U16Vec2,
    buf: Box<[u8]>,
    map: HashMap<char, Glyph>,
    line_height: u16,
}

#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub start: U16Vec2,
    pub end: U16Vec2,
}

#[derive(Clone, Copy, Debug)]
pub struct Glyph {
    pub size: U16Vec2,
    pub offset: U16Vec2,
    pub texture: Rect,
}

impl FontMap {
    pub fn dimensions(&self) -> UVec2 {
        self.dim.into()
    }

    pub fn line_height(&self) -> u16 {
        self.line_height
    }

    pub fn copy_into(&self, b: &mut [u8]) {
        assert_eq!(
            b.len(),
            self.buf.len(),
            "dimension mismatch. Correct format?"
        );
        b.copy_from_slice(&self.buf);
    }

    pub fn copy_into_rgba(&self, b: &mut [u8]) {
        assert_eq!(
            b.len(),
            self.buf.len() * 4,
            "dimension mismatch. Correct format?"
        );
        for (d, s) in b.chunks_mut(4).zip(&*self.buf) {
            d.fill(*s);
        }
    }

    pub fn get(&self, chr: char) -> Option<Glyph> {
        self.map.get(&chr).copied()
    }
}
