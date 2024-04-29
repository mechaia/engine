use super::Rect;
use crate::{font::FontMap, Draw, Instance};
use glam::{IVec2, U16Vec2, Vec4};

pub struct Text<'a, 'b> {
    draw: &'a mut Draw<'b>,
    clip: Rect,
    pos: IVec2,
    line_start: i32,
    texture: u32,
    color: Vec4,
    font_map: &'a FontMap,
    wrap_mode: WrapMode,
}

pub enum WrapMode {
    Clip,
    WordWrap { line_width: u16 },
    WordBreak { line_width: u16 },
}

impl<'a, 'b> Text<'a, 'b> {
    pub fn new(
        draw: &'a mut Draw<'b>,
        font_texture: u32,
        font_map: &'a FontMap,
        start: IVec2,
        clip: Rect,
    ) -> Self {
        let clip = {
            let d = clip.start().min(IVec2::ZERO);
            Rect::from_start_end(clip.start() + d, clip.end()).unwrap()
        };

        Self {
            draw,
            texture: font_texture,
            font_map,
            color: Vec4::ONE,
            wrap_mode: WrapMode::Clip,
            pos: start,
            line_start: start.x,
            clip,
        }
    }

    pub fn set_color(&mut self, color: Vec4) {
        self.color = color;
    }

    pub fn push(&mut self, chr: char) {
        if chr == '\n' {
            self.pos.x = self.line_start;
            self.pos.y += i32::from(self.font_map.line_height());
            return;
        }

        let r = self.font_map.get(chr).unwrap();

        let rect = Rect {
            offset: self.pos,
            size: r.size,
        };
        let (clipped, delta) = self.clip.clip_rect(&rect);

        if let Some(clipped) = clipped {
            let uv_start = IVec2::from(r.texture.start) + IVec2::new(delta.left, delta.top);
            let uv_end = IVec2::from(r.texture.end) + IVec2::new(delta.right, delta.bottom);

            let inst = Instance {
                position: clipped.offset.try_into().unwrap(),
                size: clipped.size,
                rotation: 0.0,
                uv_start: uv_start.as_vec2() / self.font_map.dimensions().as_vec2(),
                uv_end: uv_end.as_vec2() / self.font_map.dimensions().as_vec2(),
                texture: self.texture,
                color: self.color,
            };
            self.draw.push(&inst);
        }

        match &self.wrap_mode {
            WrapMode::Clip => self.pos.x += i32::from(r.size.x),
            WrapMode::WordWrap { line_width } => {}
            WrapMode::WordBreak { line_width } => {}
        }
    }
}

impl Extend<char> for Text<'_, '_> {
    fn extend<T: IntoIterator<Item = char>>(&mut self, iter: T) {
        for c in iter {
            self.push(c);
        }
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
