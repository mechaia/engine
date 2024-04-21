use std::{collections::HashMap, ops::RangeInclusive};

pub use window::SmolStr;

pub type Key = window::InputKey;

/// Mapping of inputs
#[derive(Clone, Debug, Default)]
pub struct InputMap {
    label_to_input: HashMap<Box<str>, Vec<(Key, Remap)>>,
}

/// Remap input to a different range
#[derive(Clone, Debug)]
pub struct Remap {
    range: RangeInclusive<f32>,
    scale: f32,
    offset: f32,
}

impl InputMap {
    pub fn from_list(list: &[(&str, Key, Remap)]) -> Self {
        let mut slf = Self::default();
        for (lbl, input, remap) in list {
            slf.add(lbl.to_string().into(), input.clone(), remap.clone());
        }
        slf
    }

    pub fn add(&mut self, label: Box<str>, input: Key, remap: Remap) {
        self.label_to_input
            .entry(label)
            .or_default()
            .push((input, remap));
    }

    pub fn get(&self, label: &str) -> &[(Key, Remap)] {
        self.label_to_input.get(label).map_or(&[], |v| &**v)
    }
}

impl Remap {
    /// Remap one range to another
    pub fn from_ranges(range: RangeInclusive<f32>, to: RangeInclusive<f32>) -> Self {
        let range_len = range.end() - range.start();
        let to_len = to.end() - to.start();
        // TODO what about divide by zero?
        let scale = to_len / range_len;
        let offset = to.start() - range.start() * scale;
        Self {
            range,
            scale,
            offset,
        }
    }

    /// Remap input.
    ///
    /// Returns `None` if out of range.
    pub fn apply(&self, value: f32) -> Option<f32> {
        self.range
            .contains(&value)
            .then(|| (value * self.scale) + self.offset)
    }
}

/// Filter to make boolean input edge-triggered.
#[derive(Default)]
pub struct TriggerEdge {
    prev: bool,
}

impl TriggerEdge {
    pub fn apply(&mut self, input: f32) -> bool {
        let p = self.prev;
        self.prev = input > 0.0;
        !p & self.prev
    }
}

/// Filter to make boolean input "toggled" on edge trigger
// FIXME what was the damn term? "hold"?
#[derive(Default)]
pub struct TriggerToggle {
    edge: TriggerEdge,
    value: bool,
}

impl TriggerToggle {
    pub fn apply(&mut self, input: f32) -> bool {
        self.value ^= self.edge.apply(input);
        self.value
    }
}

/// Parse text configuration while handling errors gracefully
impl InputMap {
    pub fn parse_from_lines_graceful<I: IntoIterator<Item = S>, S: AsRef<str>>(
        lines: I,
        error: &mut dyn FnMut(usize, &str),
    ) -> Self {
        let mut s = Self::default();

        for (i, line) in lines.into_iter().enumerate() {
            let line = line.as_ref();
            let line = line.split_once('#').map_or(line, |l| l.0);
            let mut it = line.split_whitespace();
            let Some(input) = it.next() else {
                // all whitespace
                continue;
            };
            let mut error = |msg| error(i, &format!("{input}: {msg}"));
            let Some(key) = it.next() else {
                error("no key defined, skipping");
                continue;
            };
            let Some(key) = str_to_key(key) else {
                error(&format!("unrecognized key {key}, skipping"));
                continue;
            };
            let mut next = |def: f32, msg| {
                let mut f = || {
                    error(msg);
                    def
                };
                let Some(v) = it.next() else { return f() };
                v.parse::<f32>().unwrap_or_else(|_| f())
            };
            let range_low = next(0.0, "invalid low range defined, assuming 0");
            let range_high = next(1.0, "invalid high range defined, assuming 1");
            let remap_low = next(0.0, "invalid low remap defined, assuming 0");
            let remap_high = next(1.0, "invalid high remap defined, assuming 1");
            let remap = Remap::from_ranges(range_low..=range_high, remap_low..=remap_high);
            s.add(input.into(), key, remap);
        }

        s
    }
}

fn str_to_key(key: &str) -> Option<Key> {
    use window::InputKey::*;
    Some(if key.starts_with(":") {
        match &*key[1..].to_lowercase() {
            ":" | "!" | "+" | "-" | "%" | "/" | "*" => Unicode(SmolStr::new_inline(&key[1..])),
            "space" => Unicode(SmolStr::new_inline(" ")),
            "enter" => Unicode(SmolStr::new_inline("\n")),
            "tab" => Unicode(SmolStr::new_inline("\t")),
            "alt" => Alt,
            "altgr" => AltGr,
            "lctrl" => LCtrl,
            "rctrl" => RCtrl,
            "lshift" => LShift,
            "rshift" => RShift,
            "lsuper" => LSuper,
            "rsuper" => RSuper,
            "mouserelativex" => MouseRelativeX,
            "mouserelativey" => MouseRelativeY,
            "mousewheely" => MouseWheelY,
            "mousebuttonl" => MouseButtonL,
            "mousebuttonm" => MouseButtonM,
            "mousebuttonr" => MouseButtonR,
            "esc" => Esc,
            "arrowup" => ArrowUp,
            "arrowdown" => ArrowDown,
            "arrowleft" => ArrowLeft,
            "arrowright" => ArrowRight,
            _ => return None,
        }
    } else if "!+-/%*".contains(key.chars().next().unwrap()) {
        return None;
    } else {
        Unicode(SmolStr::new(key))
    })
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn remap() {
        let remap = Remap::from_ranges(-1.0..=1.0, 10.0..=60.0);
        assert_eq!(remap.apply(-2.0), None);
        assert_eq!(remap.apply(2.0), None);
        assert_eq!(remap.apply(-1.0), Some(10.0));
        assert_eq!(remap.apply(0.0), Some(35.0));
        assert_eq!(remap.apply(1.0), Some(60.0));
        assert_eq!(remap.apply(-0.5), Some(22.5));
        assert_eq!(remap.apply(0.5), Some(60.0 - 12.5));
    }

    #[test]
    fn text_configuration() {
        let text = r"
            input.a   A   0 1   0 1
            input.b   B   0 1   -1 1
            input.c   C   -1 1  2 5
            input.d   LCtrl   -0.333 0.7   3.1415 13.37
        ";
        let map = InputMap::parse_from_lines_graceful(text.lines(), &mut |i, e| {
            dbg!(i, e);
        });
        dbg!(&map);
        // FIXME validate you lazy cunt
        //todo!();
    }
}
