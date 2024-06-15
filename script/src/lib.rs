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
    switches: Map<Str, Switch>,
    functions: Map<Str, Function>,
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
pub struct Error {
    line: usize,
    kind: ErrorKind,
}

#[derive(Debug)]
enum ErrorKind {
    UnterminatedString,
    ExpectedIdentifier,
    DuplicateType(String),
    DuplicateField,
    OverlappingRegister(String),
    DuplicateFunction(String),
    ExpectedWord,
    ExpectedEndOfLine,
    ExpectedLine,
    TypeNotFound(String),
    RegisterNotFound(String),
    FunctionNotFound(String),
    InvalidInstruction(String),
}

#[derive(Debug)]
struct Switch {
    register: Str,
    branches: Box<[(Str, Str)]>,
    default: Option<Str>,
}

#[derive(Debug)]
enum Function {
    User {
        instructions: Box<[Instruction]>,
        /// `None` if return, `Some` if jump.
        next: Option<Str>,
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
    pub fn parse_text(&mut self, text: &str) -> Result<(), Error> {
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
        self.parse_text(include_str!("../std.pil"))
    }

    pub fn add_ieee754(&mut self) -> Result<(), Error> {
        sys::add_ieee754(self)?;
        self.parse_text(include_str!("../ieee754.pil"))
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
            .map_err(|_| todo!())
    }

    fn line_parse_register(&mut self, line: &str) -> Result<(), ErrorKind> {
        let [name, ty] = next_words_eol(line)?;
        self.registers
            .try_insert(name, ty)
            .map_err(|e| ErrorKind::OverlappingRegister(e.entry.key().to_string()))
            .map(|_| ())
    }

    fn line_parse_register_array(&mut self, line: &str) -> Result<(), ErrorKind> {
        let [name, index, value] = next_words_eol(line)?;
        self.array_registers
            .try_insert(name, (index, value))
            .map_err(|e| ErrorKind::OverlappingRegister(e.entry.key().to_string()))
            .map(|_| ())
    }

    fn line_parse_switch(&mut self, lines: &mut Lines<'_>, line: &str) -> Result<(), ErrorKind> {
        let [name, register] = next_words_eol(line)?;

        // Use a plain list to
        // - have consistent compile output
        // - take in account optimizations done by the user, like putting common case first
        let mut branches = Vec::new();

        let default = loop {
            let line = lines.next().ok_or(ErrorKind::ExpectedLine)?.trim_end();
            let Ok((word, line)) = next_word(line) else {
                continue;
            };
            match word {
                "?" => {
                    let (value, line) = next_word(line)?;
                    let (function, line) = next_word(line)?;
                    branches.push((value.into(), function.into()));
                    next_eol(line)?;
                }
                "]" => {
                    next_eol(line)?;
                    break None;
                }
                "!" => {
                    let (name, line) = next_word(line)?;
                    next_eol(line)?;
                    break Some(name);
                }
                tk => todo!("{tk}"),
            }
        };

        let f = Switch {
            register: register.into(),
            branches: branches.into(),
            default: default.map(|d| d.into()),
        };

        self.switches
            .try_insert(name.into(), f)
            .map(|_| ())
            .map_err(|e| ErrorKind::DuplicateFunction(e.entry.key().clone().into()))
    }

    fn line_parse_function(&mut self, lines: &mut Lines<'_>, line: &str) -> Result<(), ErrorKind> {
        let (name, line) = next_word(line)?;
        next_eol(line)?;

        let mut instructions = Vec::new();

        let next = loop {
            let line = next_line(lines)?;
            let (word, line) = next_word(line)?;
            match word {
                "." => {
                    let [to, from] = next_words_eol(line)?;
                    instructions.push(Instruction::Move { to, from });
                }
                "+" => {
                    let [to, value] = next_words_eol(line)?;
                    instructions.push(Instruction::Set { to, value });
                }
                "{" => {
                    let [array, index, register] = next_words_eol(line)?;
                    instructions.push(Instruction::ToArray {
                        index,
                        array,
                        register,
                    });
                }
                "}" => {
                    let [array, index, register] = next_words_eol(line)?;
                    instructions.push(Instruction::FromArray {
                        index,
                        array,
                        register,
                    });
                }
                "|" => {
                    let [function] = next_words_eol(line)?;
                    instructions.push(Instruction::Call { function });
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
                    instructions.push(Instruction::ToGroup {
                        group,
                        field,
                        register,
                    });
                }
                ")" => {
                    let [group, field, register] = next_words_eol(line)?;
                    instructions.push(Instruction::FromGroup {
                        group,
                        field,
                        register,
                    });
                }
                tk => todo!("{tk}"),
            }
        };

        let f = Function::User {
            instructions: instructions.into(),
            next: next.map(|s| s.into()),
        };

        self.functions
            .try_insert(name.into(), f)
            .map(|_| ())
            .map_err(|_| ErrorKind::DuplicateFunction(name.into()))
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
            return Err(())
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
        Self { map: Default::default() }
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
        Self { map: iter.into_iter().collect() }
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
    fn stuff() {
        let mut col = super::Collection::default();
        col.add_standard().unwrap();
        let s = include_str!("../examples/hello_world.pil");
        let s = include_str!("../examples/union.pil");
        let s = include_str!("../examples/group.pil");
        let s = include_str!("../examples/array.pil");
        col.parse_text(s).unwrap();
        dbg!(&col);
        let mut prog = super::Program::from_collection(&col, &"start".to_string().into()).unwrap();
        dbg!(&prog);
        super::optimize::simple(&mut prog);
        dbg!(&prog);
        let vm = super::executor::wordvm::WordVM::from_program(&prog);
        dbg!(&vm);
        let mut exec = vm.create_state();
        loop {
            use {super::executor::wordvm::Yield, super::sys::wordvm::External};
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
}
