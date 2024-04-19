use super::Rect;
use crate::{font::FontMap, Draw, Instance};
use glam::{IVec2, U16Vec2, Vec4};

/// Automatically clips out-of-bounds text.
pub fn draw(
    draw: &mut Draw<'_>,
    mut pos: IVec2,
    clip: Rect<U16Vec2>,
    text: &str,
    texture: u32,
    color: Vec4,
    font_map: &FontMap,
) {
    // TODO newlines, wrapping ...
    // We need a layouter for glyphs

    let mut i = 0;
    for chr in text.chars() {
        let r = font_map.get(chr).unwrap();

        let rect = Rect {
            offset: pos,
            size: r.size,
        };
        let (clipped, delta) = clip.clip_rect(&rect);

        if let Some(clipped) = clipped {
            /*
            if pos.x >= i32::from(r.size.x) {
                break;
            }

            let position = pos.max(IVec2::ZERO);
            let clip = pos - position;
            let size = (IVec2::from(size) - clip).max(IVec2::ZERO);
            */

            let uv_start = IVec2::from(r.texture.start) + IVec2::new(delta.left, delta.top);
            let uv_end = IVec2::from(r.texture.end) + IVec2::new(delta.right, delta.bottom);

            let inst = Instance {
                position: clipped.offset,
                size: clipped.size,
                rotation: 0.0,
                uv_start: uv_start.as_vec2() / font_map.dimensions().as_vec2(),
                uv_end: uv_end.as_vec2() / font_map.dimensions().as_vec2(),
                texture,
                color,
            };
            draw.push(&inst);
        }

        pos.x += i32::from(r.size.x);
    }
}

pub fn size(text: &str, font_map: &FontMap) -> U16Vec2 {
    let mut pos @ mut size = U16Vec2::ZERO;
    for chr in text.chars() {
        let r = font_map.get(chr).unwrap();
        size = size.max(pos + r.size);
        pos.x += r.size.x;
    }
    size
}
