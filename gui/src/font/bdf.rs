// "reverse-engineered" by reading the spleen-*.bdf files and taking guesses

use glam::{I16Vec2, U16Vec2, UVec2, Vec2Swizzles};

use super::{FontMap, Glyph, Rect};
use core::{fmt, str::FromStr};
use std::{borrow::Cow, collections::HashMap};

pub const SPLEEN_32X64: &str = include_str!("../../font/spleen/spleen-32x64.bdf");
pub const SPLEEN_16X32: &str = include_str!("../../font/spleen/spleen-16x32.bdf");
pub const SPLEEN_8X16: &str = include_str!("../../font/spleen/spleen-8x16.bdf");

struct BoundingBox {
    dim: U16Vec2,
    offset: I16Vec2,
}

struct TexBuf {
    dim: U16Vec2,
    buf: Box<[u8]>,
}

struct MonoPacker {
    tiles: U16Vec2,
    tile_dim: U16Vec2,
    cur_tile: U16Vec2,
    remaining: u32,
    map: FontMap,
}

impl MonoPacker {
    fn from_dim_count(dim: U16Vec2, count: u32, line_height: u16) -> Self {
        // try to keep it square, nothing too complicated
        let mut tiles = U16Vec2::ZERO;
        while UVec2::from(tiles).element_product() < count {
            if tiles.x * dim.x < tiles.y * dim.y {
                tiles.x += 1;
            } else {
                tiles.y += 1;
            }
        }
        Self {
            tiles,
            tile_dim: dim,
            cur_tile: U16Vec2::ZERO,
            remaining: count,
            map: FontMap {
                dim: tiles * dim,
                buf: (0..UVec2::from(tiles * dim).element_product())
                    .map(|_| 0)
                    .collect(),
                map: Default::default(),
                line_height,
            },
        }
    }

    fn push(&mut self, chr: char, data: &[u8]) {
        assert_eq!(data.len(), self.tile_dim.element_product() as usize);
        assert!(self.remaining > 0);

        // do memcpy row by row
        let rowlen = usize::from(self.tile_dim.x);
        let mut it = data.chunks_exact(rowlen);
        for y in 0..self.tile_dim.y {
            let t = (self.cur_tile * self.tile_dim) + U16Vec2::new(0, y);
            let di = usize::from(t.y) * usize::from(self.map.dim.x) + usize::from(t.x);
            let s = it.next().expect("char row");
            self.map.buf[di..di + rowlen].copy_from_slice(s);
        }

        let start = self.cur_tile * self.tile_dim;
        let end = start + self.tile_dim;
        let glyph = Glyph {
            size: self.tile_dim,
            offset: U16Vec2::ZERO, // FIXME
            texture: Rect { start, end },
        };
        let prev = self.map.map.insert(chr, glyph);
        assert!(prev.is_none(), "duplicate character {chr}");

        self.remaining -= 1;
        self.cur_tile.x += 1;
        if self.cur_tile.x >= self.tiles.x {
            self.cur_tile.y += 1;
            self.cur_tile.x = 0;
        }
    }
}

pub fn parse_from_str(s: &str) -> FontMap {
    parse_from_lines(&mut s.lines().map(|s| s.to_string()))
}

pub fn parse_from_lines(iter: &mut dyn Iterator<Item = String>) -> FontMap {
    let iter = &mut wrap(iter);

    let startfont = iter.next().expect("startfont");
    assert_eq!(startfont, "STARTFONT 2.1");

    let mut bbx: Option<BoundingBox> = None;
    let mut packer = None;
    let mut line_height = None;

    loop {
        let line = iter.next().expect("line");
        let words = &mut line.split_whitespace();
        let cmd = words.next().expect("command");
        match cmd {
            "CHARS" => {
                let char_count = next_as::<u32>(words);
                next_none(words);
                let bbx = bbx.take().unwrap();
                let line_height = line_height.expect("no size defined");
                let p = MonoPacker::from_dim_count(bbx.dim, char_count, line_height);
                let p = packer.insert(p);
                parse_chars(iter, p);
            }
            "FONT" => {}
            "FONTBOUNDINGBOX" => {
                bbx = Some(BoundingBox {
                    dim: U16Vec2 {
                        x: next_as(words),
                        y: next_as(words),
                    },
                    offset: I16Vec2 {
                        x: next_as(words),
                        y: next_as(words),
                    },
                });
                next_none(words);
            }
            "SIZE" => {
                line_height = Some(next_as(words));
                let _ = next_as::<u8>(words);
                let _ = next_as::<u8>(words);
                next_none(words);
            }
            "STARTPROPERTIES" => {
                let line_count = next_as::<u8>(words);
                next_none(words);
                parse_start_properties(iter, line_count);
            }
            "ENDFONT" => {
                next_none(words);
                break;
            }
            _ => panic!("unknown command {cmd}"),
        }
    }

    assert!(iter.next().is_none(), "expected EOF");
    packer.expect("no chars parsed").map
}

fn parse_start_properties(lines: &mut dyn Iterator<Item = String>, line_count: u8) {
    // just skip, idk
    for _ in 0..line_count {
        lines.next().expect("startproperties line");
    }
    let end = lines.next().expect("endproperties");
    assert_eq!(end, "ENDPROPERTIES");
}

fn parse_chars(lines: &mut dyn Iterator<Item = String>, packer: &mut MonoPacker) {
    let mut buf = Vec::with_capacity(packer.tile_dim.element_product().into());

    for _ in 0..packer.remaining {
        let line = lines.next().expect("startchar");
        let (cmd, rest) = line.split_once(' ').expect("startchar");
        assert_eq!(cmd, "STARTCHAR");

        let mut encoding = None;
        let mut bbx = None;

        'chr: loop {
            let line = lines.next().expect("line");
            let words = &mut line.split_whitespace();
            let cmd = words.next().expect("command");

            match cmd {
                "ENCODING" => {
                    let enc = next_as::<u32>(words);
                    encoding = Some(char::try_from(enc).expect("invalid encoding"));
                    next_none(words);
                }
                "SWIDTH" | "DWIDTH" => {
                    let _ = next_as::<u16>(words);
                    let _ = next_as::<u16>(words);
                    next_none(words);
                }
                "BBX" => {
                    bbx = Some(BoundingBox {
                        dim: U16Vec2 {
                            x: next_as(words),
                            y: next_as(words),
                        },
                        offset: I16Vec2 {
                            x: next_as(words),
                            y: next_as(words),
                        },
                    });
                }
                "BITMAP" => break 'chr,
                _ => panic!("unknown command {cmd}"),
            }
        }

        let encoding = encoding.expect("no encoding specified");
        let bbx = bbx.expect("no bbx specified");

        for y in 0..bbx.dim.y {
            let line = lines.next().expect("line");
            assert_eq!(line.len() as u16, (bbx.dim.x + 3) / 4);
            let f = |i| match line.as_bytes()[i] {
                c @ b'0'..=b'9' => c - b'0',
                c @ b'A'..=b'F' => c - b'A' + 10,
                c => panic!("expected hex character, got {}", c as char),
            };
            for xi in 0..bbx.dim.x / 4 {
                let b = f(usize::from(xi));
                for x in (0..4).rev() {
                    buf.push(((b >> x) & 1) * u8::MAX);
                }
            }
            if bbx.dim.x % 4 != 0 {
                let b = f(line.len() - 1);
                for x in (4 - (bbx.dim.x % 4)..4).rev() {
                    buf.push(((b >> x) & 1) * u8::MAX);
                }
            }
        }

        packer.push(encoding, &buf);
        buf.clear();

        let line = lines.next().expect("startchar");
        assert_eq!(line, "ENDCHAR");
    }
}

/// Wrapper with some utils like skipping comments
fn wrap(iter: &mut dyn Iterator<Item = String>) -> impl Iterator<Item = String> + '_ {
    iter.filter(|s| !s.starts_with("COMMENT "))
}

fn next_as<T>(iter: &mut dyn Iterator<Item = &str>) -> T
where
    T: FromStr + fmt::Debug,
    T::Err: fmt::Debug,
{
    iter.next()
        .expect("something to parse")
        .parse::<T>()
        .expect("failed to parse value")
}

fn next_none(iter: &mut dyn Iterator<Item = &str>) {
    let v = iter.next();
    assert!(v.is_none(), "expected EOL");
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn spleen_5x8() {
        let s = include_str!("../../font/spleen/spleen-5x8.bdf");
        parse_from_lines(&mut s.lines().map(|s| s.to_string()));
    }
}
