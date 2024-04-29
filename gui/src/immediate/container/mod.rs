use super::{Margins, Rect, Value};
use glam::{IVec2, U16Vec2};

/// Calculate new rectangle with margins
///
/// Returns `None` if the size would be negative.
pub fn with_margins(outer: Rect, margins: Margins<Value>) -> Option<Rect> {
    let value_to_px = |value: Value, len: u16| match value {
        Value::Pixel(v) => v,
        Value::Norm(v) => (v * f32::from(len)) as i32,
    };

    let left = value_to_px(margins.left, outer.size.x);
    let right = value_to_px(margins.right, outer.size.x);
    let top = value_to_px(margins.top, outer.size.y);
    let bottom = value_to_px(margins.bottom, outer.size.y);

    let offset = outer.offset + IVec2::new(left, top);
    let size = IVec2::from(outer.size) - IVec2::new(left + right, top + bottom);

    let size = U16Vec2::try_from(size).ok()?;

    Some(Rect { offset, size })
}
