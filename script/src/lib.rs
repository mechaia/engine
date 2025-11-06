#![feature(slice_split_once, map_try_insert, iterator_try_collect)]
#![deny(unused_must_use, elided_lifetimes_in_paths)]

use core::fmt;
use std::{hash::Hash, ops::Index};

mod executor;
pub mod optimize;
mod program;
pub mod sys;

//pub use util::str::PoolBoxU8 as Str;
pub use executor::{Executable, Instance, Yield};
pub use program::Program;

type Str = Box<str>;
type Map<K, V> = std::collections::HashMap<K, V>;

#[derive(Debug, Default)]
pub struct Collection {
    types: Map<Str, Type>,
    constants: Map<Str, Map<Str, Map<Str, Str>>>,
    registers: Map<Str, Str>,
    /// (Index, Value)
    array_registers: Map<Str, (Str, Str)>,
    functions: Map<Str, Function>,
    file_names: Vec<Str>,
}

/// Key-value map with keys in insertion order.
#[derive(Clone)]
struct LinearMap<K, V> {
    map: util::soa::Vec2<K, V>,
}

#[derive(Debug)]
pub(crate) enum Type {
    Enum(Box<[Str]>),
    Group(Box<[(Str, Str)]>),
    Builtin(BuiltinType),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum BuiltinType {
    Int(u8),
    Fp32,
    ConstantString,
    Opaque { bits: u8 },
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct Error {
    line: usize,
    kind: ErrorKind,
}

#[derive(Debug)]
#[allow(dead_code)]
enum ErrorKind {
    UnterminatedString,
    ExpectedIdentifier,
    DuplicateType(String),
    DuplicateField,
    DuplicateRegister(String),
    DuplicateFunction(String),
    DuplicateConstant(String, String),
    ExpectedWord,
    ExpectedEndOfLine,
    ExpectedLine,
    TypeNotFound(String),
    RegisterNotFound(String),
    FunctionNotFound(String),
    InvalidInstruction(String),
}

#[derive(Debug)]
enum Function {
    Block {
        instructions: Box<[Instruction]>,
        /// `None` if return, `Some` if jump.
        next: Option<Str>,
        file: u32,
        lines: Box<[u32]>,
        last_line: u32,
    },
    Switch {
        register: Str,
        branches: Box<[(Str, Str)]>,
        default: Option<Str>,
        file: u32,
        lines: Box<[u32]>,
        last_line: u32,
    },
    Builtin {
        id: u32,
        /// Input registers
        inputs: Box<[Str]>,
        /// Output registers
        outputs: Box<[Str]>,
    },
}

#[derive(Debug)]
enum Instruction {
    Move {
        to: Str,
        from: Str,
    },
    Set {
        to: Str,
        value: Str,
    },
    ToArray {
        index: Str,
        array: Str,
        register: Str,
    },
    FromArray {
        index: Str,
        array: Str,
        register: Str,
    },
    FromGroup {
        group: Str,
        field: Str,
        register: Str,
    },
    ToGroup {
        group: Str,
        field: Str,
        register: Str,
    },
    Call {
        function: Str,
    },
}

struct Lines<'a> {
    lines: core::str::Lines<'a>,
    line: usize,
}

pub const BUILTIN_WRITE_CONSTANT_STRING: u32 = 0;
pub const BUILTIN_WRITE_CHARACTER: u32 = 1;

impl Collection {
    pub fn parse_text(&mut self, file_name: Option<&str>, text: &str) -> Result<(), Error> {
        let file_name = file_name.unwrap_or("<n/a>");
        self.file_names.push(file_name.into());

        let mut lines = Lines {
            lines: text.lines(),
            line: 0,
        };
        while let Some(line) = lines.next() {
            let line = trim_line(line);
            if line.is_empty() {
                continue;
            }
            let (word, line) = next_word(line).map_err(|kind| Error {
                kind,
                line: lines.line,
            })?;
            match word {
                "%" => self.line_parse_type(line),
                "(" => self.line_parse_group(&mut lines, line),
                "_" => self.line_parse_constant(&mut lines, line),
                "$" => self.line_parse_register(line),
                "@" => self.line_parse_register_array(line),
                "[" => self.line_parse_switch(&mut lines, line),
                ">" => self.line_parse_function(&mut lines, line),
                tk => {
                    return Err(Error {
                        line: lines.line,
                        kind: ErrorKind::InvalidInstruction(tk.into()),
                    })
                }
            }
            .map_err(|kind| Error {
                kind,
                line: lines.line,
            })?;
        }
        Ok(())
    }

    pub fn add_standard(&mut self) -> Result<(), Error> {
        sys::add_defaults(self)?;
        self.parse_text(Some("builtin:std.pil"), include_str!("../std.pil"))
    }

    pub fn add_ieee754(&mut self) -> Result<(), Error> {
        sys::add_ieee754(self)?;
        self.parse_text(Some("builtin:ieee754.pil"), include_str!("../ieee754.pil"))
    }

    pub fn add_opaque(&mut self, name: &str, bits: u8) -> Result<(), Error> {
        self.types
            .try_insert(name.into(), Type::Builtin(BuiltinType::Opaque { bits }))
            .map(|_| ())
            .map_err(|e| Error::new(0, ErrorKind::DuplicateType(e.entry.key().to_string())))
    }

    pub fn add_register(&mut self, name: &str, ty: &str) -> Result<(), Error> {
        self.registers
            .try_insert(name.into(), ty.into())
            .map(|_| ())
            .map_err(|e| Error::new(0, ErrorKind::DuplicateType(e.entry.key().to_string())))
    }

    pub fn add_sys(
        &mut self,
        name: &str,
        id: u32,
        inputs: &[&str],
        outputs: &[&str],
    ) -> Result<(), Error> {
        self.functions
            .try_insert(
                name.into(),
                Function::Builtin {
                    id,
                    inputs: inputs.iter().map(|&s| s.into()).collect(),
                    outputs: outputs.iter().map(|&s| s.into()).collect(),
                },
            )
            .map(|_| ())
            .map_err(|e| Error::new(0, ErrorKind::DuplicateFunction(e.entry.key().to_string())))
    }

    pub fn file_id_to_name(&self, id: u32) -> Option<&str> {
        self.file_names.get(id as usize).map(|s| &**s)
    }

    fn line_parse_type(&mut self, line: &str) -> Result<(), ErrorKind> {
        let (name, mut line) = next_word(line)?;
        let mut variants = Vec::new();
        while let Ok((word, l)) = next_word(line) {
            variants.push(word.into());
            line = l;
        }
        let ty = Type::Enum(variants.into());
        self.types
            .try_insert(name.into(), ty)
            .map_err(|_| ErrorKind::DuplicateType(name.into()))
            .map(|_| ())
    }

    fn line_parse_group(&mut self, lines: &mut Lines<'_>, line: &str) -> Result<(), ErrorKind> {
        let [name] = next_words_eol(line)?;

        let mut fields = Vec::<(Str, Str)>::new();

        loop {
            let line = next_line(lines)?;
            let (word, line) = next_word(line)?;
            match word {
                "&" => {
                    let [name, ty] = next_words_eol(line)?;
                    if fields.iter().any(|(n, _)| n == &name) {
                        return Err(ErrorKind::DuplicateField);
                    }
                    fields.push((name, ty));
                }
                ")" => {
                    next_eol(line)?;
                    break;
                }
                tk => todo!("{tk}"),
            }
        }

        self.types
            .try_insert(name, Type::Group(fields.into()))
            .map(|_| ())
            .map_err(|e| ErrorKind::DuplicateType(e.entry.key().to_string()))
    }

    fn line_parse_constant(&mut self, lines: &mut Lines<'_>, line: &str) -> Result<(), ErrorKind> {
        let (ty, line) = next_word(line)?;
        let (name, line) = next_word(line)?;
        next_eol(line)?;

        let mut values = Map::new();

        loop {
            let line = next_line(lines)?;
            let (word, line) = next_word(line)?;
            match word {
                "+" => {
                    let (field, line) = next_word(line)?;
                    let (value, line) = next_word(line)?;
                    next_eol(line)?;
                    values.try_insert(field.into(), value.into()).unwrap();
                }
                "-" => {
                    next_eol(line)?;
                    break;
                }
                tk => todo!("{tk}"),
            }
        }

        self.constants
            .entry(ty.into())
            .or_default()
            .try_insert(name.into(), values)
            .map(|_| ())
            .map_err(|e| ErrorKind::DuplicateConstant(ty.into(), e.entry.key().to_string()))
    }

    fn line_parse_register(&mut self, line: &str) -> Result<(), ErrorKind> {
        let [name, ty] = next_words_eol(line)?;
        self.registers
            .try_insert(name, ty)
            .map_err(|e| ErrorKind::DuplicateRegister(e.entry.key().to_string()))
            .map(|_| ())
    }

    fn line_parse_register_array(&mut self, line: &str) -> Result<(), ErrorKind> {
        let [name, index, value] = next_words_eol(line)?;
        self.array_registers
            .try_insert(name, (index, value))
            .map_err(|e| ErrorKind::DuplicateRegister(e.entry.key().to_string()))
            .map(|_| ())
    }

    fn line_parse_switch(&mut self, lines: &mut Lines<'_>, line: &str) -> Result<(), ErrorKind> {
        let [name, register] = next_words_eol(line)?;

        // Use a plain list to
        // - have consistent compile output
        // - take in account optimizations done by the user, like putting common case first
        let mut branches = Vec::new();
        let mut s_lines = Vec::new();

        let (default, last_line) = loop {
            let line = lines.next().ok_or(ErrorKind::ExpectedLine)?.trim_end();
            let Ok((word, line)) = next_word(line) else {
                continue;
            };
            match word {
                "?" => {
                    let (value, line) = next_word(line)?;
                    let (function, line) = next_word(line)?;
                    branches.push((value.into(), function.into()));
                    s_lines.push(lines.line as u32);
                    next_eol(line)?;
                }
                "]" => {
                    next_eol(line)?;
                    break (None, u32::MAX);
                }
                "!" => {
                    let (name, line) = next_word(line)?;
                    next_eol(line)?;
                    break (Some(name), lines.line as u32);
                }
                tk => todo!("{tk}"),
            }
        };

        let f = Function::Switch {
            register: register.into(),
            branches: branches.into(),
            default: default.map(|d| d.into()),
            file: self.cur_file(),
            lines: s_lines.into(),
            last_line,
        };

        self.functions
            .try_insert(name.into(), f)
            .map(|_| ())
            .map_err(|e| ErrorKind::DuplicateFunction(e.entry.key().clone().into()))
    }

    fn line_parse_function(&mut self, lines: &mut Lines<'_>, line: &str) -> Result<(), ErrorKind> {
        let (name, line) = next_word(line)?;
        next_eol(line)?;

        let mut instructions = util::soa::Vec2::new();

        let next = loop {
            let line = next_line(lines)?;
            let (word, line) = next_word(line)?;
            let instr = match word {
                "." => {
                    let [to, from] = next_words_eol(line)?;
                    Instruction::Move { to, from }
                }
                "+" => {
                    let [to, value] = next_words_eol(line)?;
                    Instruction::Set { to, value }
                }
                "{" => {
                    let [array, index, register] = next_words_eol(line)?;
                    Instruction::ToArray {
                        index,
                        array,
                        register,
                    }
                }
                "}" => {
                    let [array, index, register] = next_words_eol(line)?;
                    Instruction::FromArray {
                        index,
                        array,
                        register,
                    }
                }
                "|" => {
                    let [function] = next_words_eol(line)?;
                    Instruction::Call { function }
                }
                "<" => {
                    next_eol(line)?;
                    break None;
                }
                "=" => {
                    let [next] = next_words_eol(line)?;
                    break Some(next);
                }
                "(" => {
                    let [group, field, register] = next_words_eol(line)?;
                    Instruction::ToGroup {
                        group,
                        field,
                        register,
                    }
                }
                ")" => {
                    let [group, field, register] = next_words_eol(line)?;
                    Instruction::FromGroup {
                        group,
                        field,
                        register,
                    }
                }
                tk => todo!("{tk}"),
            };
            instructions.push((instr, u32::try_from(lines.line).unwrap()));
        };

        let last_line = lines.line.try_into().unwrap();
        let (instructions, lines) = <(Vec<_>, Vec<_>)>::from(instructions);

        let f = Function::Block {
            instructions: instructions.into(),
            lines: lines.into(),
            file: self.cur_file(),
            next: next.map(|s| s.into()),
            last_line,
        };

        self.functions
            .try_insert(name.into(), f)
            .map(|_| ())
            .map_err(|_| ErrorKind::DuplicateFunction(name.into()))
    }

    fn cur_file(&self) -> u32 {
        u32::try_from(self.file_names.len() - 1).unwrap()
    }
}

impl Error {
    fn new(line: usize, kind: ErrorKind) -> Self {
        Self { line, kind }
    }
}

impl<'a> Iterator for Lines<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        self.line += 1;
        self.lines.next()
    }
}

impl<K, V> LinearMap<K, V> {
    fn iter(&self) -> impl Iterator<Item = (&K, &V)> + '_ {
        self.map.iter()
    }

    fn values(&self) -> impl Iterator<Item = &V> + '_ {
        self.iter().map(|(_, v)| v)
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

impl<K, V> LinearMap<K, V>
where
    K: Eq,
{
    fn get(&self, key: &K) -> Option<&V> {
        self.map.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }

    fn contains(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    fn try_insert(&mut self, key: K, value: V) -> Result<&mut V, ()> {
        if self.contains(&key) {
            return Err(());
        }
        self.map.push((key, value));
        Ok(self.map.get_mut(self.map.len() - 1).unwrap().1)
    }
}

impl<K, V> Index<&K> for LinearMap<K, V>
where
    K: Eq,
{
    type Output = V;

    fn index(&self, index: &K) -> &Self::Output {
        self.get(index).expect("item not found")
    }
}

impl<K, V> Index<K> for LinearMap<K, V>
where
    K: Eq,
{
    type Output = V;

    fn index(&self, index: K) -> &Self::Output {
        &self[&index]
    }
}

impl<K, V> Default for LinearMap<K, V> {
    fn default() -> Self {
        Self {
            map: Default::default(),
        }
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for LinearMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut f = f.debug_map();
        for (k, v) in self.map.iter() {
            f.entry(k, v);
        }
        f.finish()
    }
}

impl<K, V> FromIterator<(K, V)> for LinearMap<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        Self {
            map: iter.into_iter().collect(),
        }
    }
}

fn next_line<'a>(lines: &mut Lines<'a>) -> Result<&'a str, ErrorKind> {
    loop {
        let line = lines.next().ok_or(ErrorKind::ExpectedLine)?;
        let line = trim_line(line);
        if !line.is_empty() {
            return Ok(line);
        }
    }
}

fn trim_line(line: &str) -> &str {
    line.split('#').next().unwrap_or("").trim_end()
}

fn next_word(s: &str) -> Result<(&str, &str), ErrorKind> {
    let s = s.trim_start();

    if s.starts_with("\"") {
        return next_string(s);
    }

    let (s, rest) = s.split_once([' ', '\t']).unwrap_or((s, ""));
    if s.is_empty() {
        return Err(ErrorKind::ExpectedWord);
    }
    Ok((s, rest))
}

fn next_string(s: &str) -> Result<(&str, &str), ErrorKind> {
    let mut i = 1;

    loop {
        match s.as_bytes().get(i).ok_or(ErrorKind::UnterminatedString)? {
            b'"' => {
                let (s, rest) = s.split_at(i + 1);
                break Ok((s, rest));
            }
            b'\\' => todo!(),
            _ => i += 1,
        }
    }
}

fn next_eol(s: &str) -> Result<(), ErrorKind> {
    let s = s.trim_start();
    if !s.is_empty() {
        return Err(ErrorKind::ExpectedEndOfLine);
    }
    Ok(())
}

fn next_words_eol<const N: usize>(mut s: &str) -> Result<[Str; N], ErrorKind> {
    let mut array = [""; N];
    for a in array.iter_mut() {
        (*a, s) = next_word(s)?;
    }
    next_eol(s)?;
    Ok(array.map(|s| s.into()))
}

#[cfg(test)]
mod test {
    #[test]
    fn run() {
        let mut col = super::Collection::default();
        col.add_standard().unwrap();
        let s = include_str!("../examples/array.pil");
        let s = include_str!("../examples/hello_world.pil");
        let s = include_str!("../examples/group.pil");
        let s = include_str!("../examples/union.pil");
        col.parse_text(Some("test.pil"), s).unwrap();
        dbg!(&col);
        let (mut prog, mut debug) =
            super::Program::from_collection(&col, &"start".to_string().into()).unwrap();
        dbg!(&debug, &prog);
        super::optimize::simple(&mut prog, Some(&mut debug));
        //super::optimize::remove_dead(&mut prog, Some(&mut debug));
        dbg!(&debug, &prog);
        let (vm, debug) = super::executor::wordvm::WordVM::from_program(&prog, Some(&debug));
        dbg!(&debug, &vm);
        let mut exec = vm.create_state();
        loop {
            use {super::executor::wordvm::Yield, super::sys::wordvm::External};
            if let Some(debug) = &debug {
                if let Some(m) = debug.instructions.get(exec.pc() as usize) {
                    let f = col.file_id_to_name(m.file).unwrap_or("<n/a>");
                    eprintln!("{:>4} -> {}:{}", exec.pc(), f, m.line);
                }
            }
            match exec
                .step(&vm)
                .inspect_err(|_| {
                    dbg!(&exec);
                })
                .unwrap()
            {
                Yield::Preempt => {}
                Yield::Finish => break,
                Yield::Sys { id } => {
                    match super::sys::wordvm::handle(&vm, &mut exec, id).unwrap() {
                        Some(External::WriteByte(c)) => {
                            dbg!(c as char);
                        }
                        None => {}
                    }
                }
            }
        }
        todo!();
    }

    #[test]
    fn to_bytes() {
        let mut col = super::Collection::default();
        col.add_standard().unwrap();
        let s = include_str!("../examples/array.pil");
        let s = include_str!("../examples/hello_world.pil");
        let s = include_str!("../examples/group.pil");
        let s = include_str!("../examples/union.pil");
        col.parse_text(Some("test.pil"), s).unwrap();
        let (mut prog, mut debug) =
            super::Program::from_collection(&col, &"start".to_string().into()).unwrap();
        super::optimize::simple(&mut prog, Some(&mut debug));
        let b = prog.to_bytes();
        for (i, c) in b.chunks(32).enumerate() {
            let o = i * 32;
            eprint!("{o:04x} ");
            for i in 0..32 {
                if i % 4 == 0 {
                    eprint!(" ");
                }
                if let Some(b) = c.get(i) {
                    eprint!("{b:02x}");
                } else {
                    eprint!("  ");
                }
            }
            eprint!("  ");
            for &b in c {
                let b = if b' ' <= b && b <= b'~' {
                    b as char
                } else {
                    '.'
                };
                eprint!("{b}");
            }
            eprintln!();
        }
        todo!();
    }
}
