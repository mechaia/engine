//! Utilities for immediate rendering
//!
//! To ensure nice graphics, layout is done in pixel-space.
//! Computed layouts are clamped to integers to ensure consistent pixel padding on all borders.

use glam::{IVec2, U16Vec2, UVec2};

use crate::Gui;

pub mod container;
pub mod text;

/// Immediate GUI layout helper.
///
/// Keeps track of layout.
pub struct Layouter {
    stack: Vec<Rect<U16Vec2>>,
}

#[derive(Clone, Copy, Debug)]
pub struct Rect<O> {
    pub offset: O,
    pub size: U16Vec2,
}

#[derive(Clone, Copy, Debug)]
pub struct Margin<T> {
    pub left: T,
    pub right: T,
    pub top: T,
    pub bottom: T,
}

#[derive(Clone, Copy, Debug)]
pub enum Value {
    Pixel(u16),
    Mm(f32),
    Norm(f32),
}

impl Layouter {
    pub fn new(gui: &Gui) -> Self {
        let sh = gui.shared.lock().unwrap();
        let vp = sh.viewport;
        let mut stack = Vec::new();
        stack.push(Rect {
            offset: U16Vec2::ZERO,
            size: vp,
        });
        drop(sh);
        Self { stack }
    }

    pub fn push_margin(&mut self, margin: Margin<Value>) {
        let rect = self.stack.last().expect("root layout");

        let value_to_px = |value: Value, len: u16| {
            match value {
                Value::Pixel(v) => v,
                // FIXME account for dpi
                Value::Mm(v) => v as u16,
                Value::Norm(v) => (v * f32::from(len)) as u16,
            }
        };

        let left = value_to_px(margin.left, rect.size.x);
        let right = value_to_px(margin.right, rect.size.x);
        let top = value_to_px(margin.top, rect.size.y);
        let bottom = value_to_px(margin.bottom, rect.size.y);

        let new_rect = Rect {
            offset: rect.offset + U16Vec2::new(left, top),
            size: rect.size - U16Vec2::new(left + right, top + bottom),
        };

        self.stack.push(new_rect);
    }

    /// Get current rect
    pub fn current(&self) -> Rect<U16Vec2> {
        *self.stack.last().expect("root layout")
    }

    /// Pop a sublayout.
    ///
    /// # Panics
    ///
    /// No layouts pushed.
    pub fn pop(&mut self) {
        self.stack.pop().expect("no layouts");
    }
}

impl Rect<U16Vec2> {
    /// Determine offset to put rectangle in center of this rectangle.
    ///
    /// Can be negative if `rect_size` is larger than this rectangle.
    pub fn center_rect_offset(&self, rect_size: U16Vec2) -> IVec2 {
        let s = IVec2::from(self.size) - IVec2::from(rect_size);
        IVec2::from(self.offset) + (s / 2)
    }

    /// Clip a rectangle to fit.
    ///
    /// Returns second rectangle with offsets.
    pub fn clip_rect(&self, rect: &Rect<IVec2>) -> (Option<Self>, Margin<i32>) {
        let s_start = IVec2::from(self.offset);
        let s_end = s_start + IVec2::from(self.size);

        let r_start = IVec2::from(rect.offset);
        let r_end = r_start + IVec2::from(rect.size);

        let d_start = (s_start - r_start).max(IVec2::ZERO);
        let d_end = (s_end - r_end).min(IVec2::ZERO);

        let n_start = r_start + d_start;
        let n_end = r_end + d_end;

        let n = U16Vec2::try_from(n_end.saturating_sub(n_start))
            .ok()
            .map(|size| Rect {
                offset: U16Vec2::try_from(n_start).unwrap(),
                size,
            });
        let d = Margin {
            left: d_start.x,
            top: d_start.y,
            right: d_end.x,
            bottom: d_end.y,
        };
        (n, d)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// should be perfectly centered
    #[test]
    fn rect_even_center_rect_offset_even() {
        assert_eq!(
            Rect {
                offset: U16Vec2::ZERO,
                size: U16Vec2::new(4, 6),
            }
            .center_rect_offset((2, 2).into()),
            IVec2::new(1, 2)
        );
    }

    /// should be perfectly centered
    #[test]
    fn rect_odd_center_rect_offset_odd() {
        assert_eq!(
            Rect {
                offset: U16Vec2::ZERO,
                size: U16Vec2::new(5, 7),
            }
            .center_rect_offset((1, 1).into()),
            IVec2::new(2, 3)
        );
    }
}
