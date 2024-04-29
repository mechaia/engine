//! Utilities for immediate rendering
//!
//! To ensure nice graphics, layout is done in pixel-space.
//! Computed layouts are clamped to integers to ensure consistent pixel padding on all borders.

use glam::{IVec2, U16Vec2};

pub mod container;
pub mod text;

#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub offset: IVec2,
    pub size: U16Vec2,
}

#[derive(Clone, Copy, Debug)]
pub struct Margins<T> {
    pub left: T,
    pub right: T,
    pub top: T,
    pub bottom: T,
}

#[derive(Clone, Copy, Debug)]
pub enum Value {
    Pixel(i32),
    Norm(f32),
}

impl Rect {
    pub fn from_offset_size(offset: IVec2, size: U16Vec2) -> Self {
        Self { offset, size }
    }

    pub fn from_start_end(start: IVec2, end: IVec2) -> Option<Self> {
        Some(Self {
            offset: start,
            size: U16Vec2::try_from(end - start).ok()?,
        })
    }

    pub fn from_corners(a: IVec2, b: IVec2) -> Option<Self> {
        Self::from_start_end(a.min(b), a.max(b))
    }

    pub fn start(&self) -> IVec2 {
        self.offset
    }

    pub fn end(&self) -> IVec2 {
        self.offset + IVec2::from(self.size)
    }

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
    pub fn clip_rect(&self, rect: &Self) -> (Option<Self>, Margins<i32>) {
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
                offset: n_start,
                size,
            });
        let d = Margins {
            left: d_start.x,
            top: d_start.y,
            right: d_end.x,
            bottom: d_end.y,
        };
        (n, d)
    }
}

impl<T: Clone> Margins<T> {
    pub fn splat(value: T) -> Self {
        Self {
            left: value.clone(),
            right: value.clone(),
            top: value.clone(),
            bottom: value,
        }
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
                offset: IVec2::ZERO,
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
                offset: IVec2::ZERO,
                size: U16Vec2::new(5, 7),
            }
            .center_rect_offset((1, 1).into()),
            IVec2::new(2, 3)
        );
    }
}
