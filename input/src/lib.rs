use std::{
    collections::{hash_map::Entry, HashMap, VecDeque},
    ops::RangeInclusive,
};
pub use window::SmolStr;

pub type Key = window::InputKey;

/// Mapping of inputs
#[derive(Clone, Debug, Default)]
pub struct InputMap {
    label_map: HashMap<Box<str>, usize>,
    input_map: HashMap<Key, Vec<usize>>,
    list: Vec<(Box<str>, Vec<(Key, Remap)>)>,
}

/// Remap input to a different range
#[derive(Clone, Debug)]
pub struct Remap {
    range: RangeInclusive<f32>,
    scale: f32,
    offset: f32,
}

/// Track input state and events
#[derive(Default)]
pub struct InputState {
    /// Current input value
    ///
    /// If not present, default to 0.0
    current_key: HashMap<Key, f32>,
    /// Current input value
    ///
    /// If not present, default to 0.0
    current_name: HashMap<Box<str>, f32>,
    /// Input events
    events: Vec<Event>,
}

pub struct Event {
    pub key: KeyOrName,
    pub prev: f32,
    pub cur: f32,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum KeyOrName {
    Key(Key),
    Name(Box<str>),
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
        let value = (input.clone(), remap);
        let i = match self.label_map.entry(label.clone()) {
            Entry::Occupied(v) => {
                self.list[*v.get()].1.push(value);
                *v.get()
            }
            Entry::Vacant(v) => {
                self.list.push((label.clone(), Vec::from([value])));
                *v.insert(self.list.len() - 1)
            }
        };
        self.input_map.entry(input).or_default().push(i);
    }

    pub fn get_by_name(&self, label: &str) -> impl Iterator<Item = (&Key, &Remap)> {
        self.label_map
            .get(label)
            .map_or(&[][..], |v| &self.list[*v].1)
            .iter()
            .map(|(a, b)| (a, b))
    }

    pub fn get_by_key(&self, key: &Key) -> impl Iterator<Item = &str> {
        self.input_map
            .get(key)
            .into_iter()
            .flat_map(|v| v.iter().copied())
            .map(|i| &*self.list[i].0)
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

impl InputState {
    pub fn push_key(&mut self, key: Key, value: f32) {
        let prev = if value == 0.0 {
            self.current_key.remove(&key)
        } else {
            self.current_key.insert(key.clone(), value)
        };
        self.events.push(Event {
            key: KeyOrName::Key(key),
            prev: prev.unwrap_or(0.0),
            cur: value,
        });
    }

    pub fn push_name(&mut self, name: Box<str>, value: f32) {
        let prev = if value == 0.0 {
            self.current_name.remove(&name)
        } else {
            self.current_name.insert(name.clone(), value)
        };
        self.events.push(Event {
            key: KeyOrName::Name(name),
            prev: prev.unwrap_or(0.0),
            cur: value,
        });
    }

    /// Get the current value of an input.
    pub fn get_by_key(&self, key: &Key) -> f32 {
        *self.current_key.get(key).unwrap_or(&0.0)
    }

    /// Get the current value of an input.
    pub fn get_by_name(&self, name: &str) -> f32 {
        *self.current_name.get(name).unwrap_or(&0.0)
    }

    /// Iterate over events pushed after the last clear.
    pub fn events(&self) -> impl Iterator<Item = &Event> {
        self.events.iter()
    }

    /// Clear input events
    pub fn clear_events(&mut self) {
        self.events.clear();
    }
}

impl Event {
    pub fn is_hold(&self) -> bool {
        self.prev == self.cur
    }

    pub fn is_hold_over(&self, threshold: f32) -> bool {
        self.is_hold() && threshold <= self.cur
    }

    pub fn is_hold_under(&self, threshold: f32) -> bool {
        self.is_hold() && threshold > self.cur
    }

    pub fn is_edge_over(&self, threshold: f32) -> bool {
        self.prev < threshold && threshold <= self.cur
    }

    pub fn is_edge_under(&self, threshold: f32) -> bool {
        self.prev >= threshold && threshold > self.cur
    }

    pub fn is_edge_or_hold_over(&self, threshold: f32) -> bool {
        self.is_hold_over(threshold) || self.is_edge_over(threshold)
    }

    pub fn is_edge_or_hold_under(&self, threshold: f32) -> bool {
        self.is_hold_under(threshold) || self.is_edge_under(threshold)
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
        let u = |s| Unicode(SmolStr::new_inline(s));
        match &*key[1..].to_lowercase() {
            ":" | "!" | "+" | "-" | "%" | "/" | "*" => Unicode(SmolStr::new_inline(&key[1..])),
            "space" => u(" "),
            "enter" => u("\n"),
            "tab" => u("\t"),
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
            "ampersand" => u("&"),
            "pipe" => u("|"),
            "hat" => u("^"),
            "exclamation" => u("!"),
            "question" => u("?"),
            "star" => u("*"),
            "percent" => u("%"),
            "plus" => u("+"),
            "minux" => u("-"),
            _ => return None,
        }
    } else if " \n\t!+-/%*&|^?".contains(key.chars().next().unwrap()) {
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
