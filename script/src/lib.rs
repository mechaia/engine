#![feature(slice_split_once, map_try_insert, iterator_try_collect)]
#![deny(unused_must_use, elided_lifetimes_in_paths)]

use std::{borrow::Borrow, collections::BTreeMap, hash::Hash, str::Lines};

mod executor;
pub mod optimize;
mod program;

//pub use util::str::PoolBoxU8 as Str;
pub use program::Program;

type Str = Box<str>;
type Map<K, V> = std::collections::HashMap<K, V>;

#[derive(Debug, Default)]
pub struct Collection {
    types: Map<Str, Type>,
    constants: Map<Str, Map<Str, Map<Str, Str>>>,
    registers: PrefixMap<Str, Str>,
    /// (Index, Value)
    array_registers: PrefixMap<Str, (Str, Str)>,
    switches: Map<Str, Switch>,
    functions: Map<Str, Function>,
}

#[derive(Debug)]
pub(crate) enum Type {
    Enum(Box<[Str]>),
    Group(Box<[(Str, Str)]>),
    Builtin(BuiltinType),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum BuiltinType {
    Integer32,
    Natural32,
    ConstantString,
}

#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
}

#[derive(Debug)]
enum ErrorKind {
    UnterminatedString,
    ExpectedIdentifier,
    DuplicateType,
    DuplicateField,
    OverlappingRegister(String),
    DuplicateFunction,
    ExpectedWord,
    ExpectedEndOfLine,
    ExpectedLine,
    TypeNotFound(String),
    RegisterNotFound(String),
    FunctionNotFound(String),
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
        /// Registers used by this builtin.
        registers: Box<[(Mode, Str)]>,
    },
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    InOut,
    In,
    Out,
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
    Call {
        function: Str,
    },
}

#[derive(Clone, Debug)]
struct PrefixMap<K, V> {
    map: BTreeMap<K, V>,
}

pub const BUILTIN_WRITE_CONSTANT_STRING: u32 = 0;
pub const BUILTIN_WRITE_CHARACTER: u32 = 1;
pub const BUILTIN_WRITE_INTEGER32: u32 = 2;
pub const BUILTIN_WRITE_NATURAL32: u32 = 3;
pub const BUILTIN_WRITE_NEWLINE: u32 = 4;

impl Collection {
    pub fn add_default_builtins(&mut self) -> Result<(), Error> {
        let mut f = |k: &str, v| {
            self.types.try_insert(k.into(), Type::Builtin(v)).unwrap();
        };
        f("ConstantString", BuiltinType::ConstantString);
        f("Integer32", BuiltinType::Integer32);
        f("Natural32", BuiltinType::Natural32);

        let mut f = |k: &str, v: &str| {
            self.registers.try_insert(k.into(), v.into()).unwrap();
        };
        f("write_constant_string_value", "ConstantString");
        f("write_character_value", "Natural32");
        f("write_integer32_value", "Integer32");
        f("write_natural32_value", "Natural32");

        let mut f = |k: &str, id, registers: &[(_, &str)]| {
            self.functions
                .try_insert(
                    k.into(),
                    Function::Builtin {
                        id,
                        registers: registers
                            .iter()
                            .map(|(m, n)| (*m, n.to_string().into_boxed_str()))
                            .collect(),
                    },
                )
                .unwrap();
        };
        use Mode::*;
        f(
            "write_constant_string",
            BUILTIN_WRITE_CONSTANT_STRING,
            &[(In, "write_constant_string_value")],
        );
        f(
            "write_character",
            BUILTIN_WRITE_CHARACTER,
            &[(In, "write_character_value")],
        );
        f(
            "write_integer32",
            BUILTIN_WRITE_INTEGER32,
            &[(In, "write_integer32_value")],
        );
        f(
            "write_natural32",
            BUILTIN_WRITE_NATURAL32,
            &[(In, "write_natural32_value")],
        );
        f("write_newline", BUILTIN_WRITE_NEWLINE, &[]);

        Ok(())
    }

    pub fn parse_text(&mut self, prefix: &str, text: &str) -> Result<(), Error> {
        let mut lines = text.lines();
        while let Some(line) = lines.next() {
            let line = trim_line(line);
            if line.is_empty() {
                continue;
            }
            let (word, line) = next_word(line).map_err(|kind| Error { kind })?;
            match word {
                "%" => self.line_parse_type(line),
                "(" => self.line_parse_group(&mut lines, line),
                "_" => self.line_parse_constant(&mut lines, line),
                "~" => self.line_parse_alias(line),
                "$" => self.line_parse_register(line),
                "@" => self.line_parse_register_array(line),
                "[" => self.line_parse_switch(&mut lines, line),
                ">" => self.line_parse_function(&mut lines, line),
                tk => todo!("{tk}"),
            }
            .map_err(|kind| Error { kind })?;
        }
        Ok(())
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
            .map_err(|_| ErrorKind::DuplicateType)
            .map(|_| ())
    }

    fn line_parse_group(&mut self, lines: &mut Lines<'_>, line: &str) -> Result<(), ErrorKind> {
        let (name, line) = next_word(line)?;
        next_eol(line)?;

        let mut fields = Vec::<(Str, Str)>::new();

        loop {
            let line = next_line(lines)?;
            let (word, line) = next_word(line)?;
            match word {
                "&" => {
                    let (name, line) = next_word(line)?;
                    let (ty, line) = next_word(line)?;
                    if fields.iter().any(|(n, _)| &**n == name) {
                        return Err(ErrorKind::DuplicateField);
                    }
                    fields.push((name.into(), ty.into()));
                    next_eol(line)?;
                }
                ")" => {
                    next_eol(line)?;
                    break;
                }
                tk => todo!("{tk}"),
            }
        }

        self.types
            .try_insert(name.into(), Type::Group(fields.into()))
            .map(|_| ())
            .map_err(|_| todo!())
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

    fn line_parse_alias(&mut self, line: &str) -> Result<(), ErrorKind> {
        todo!()
    }

    fn line_parse_register(&mut self, line: &str) -> Result<(), ErrorKind> {
        let (name, line) = next_word(line)?;
        let (value, line) = next_word(line)?;
        next_eol(line)?;
        self.registers
            .try_insert(name.into(), value.into())
            .map_err(|_| ErrorKind::OverlappingRegister(name.into()))
            .map(|_| ())
    }

    fn line_parse_register_array(&mut self, line: &str) -> Result<(), ErrorKind> {
        let (name, line) = next_word(line)?;
        let (index, line) = next_word(line)?;
        let (value, line) = next_word(line)?;
        next_eol(line)?;
        self.array_registers
            .try_insert(name.into(), (index.into(), value.into()))
            .map_err(|_| ErrorKind::OverlappingRegister(name.into()))
            .map(|_| ())
    }

    fn line_parse_switch(&mut self, lines: &mut Lines<'_>, line: &str) -> Result<(), ErrorKind> {
        let (name, line) = next_word(line)?;
        let (register, line) = next_word(line)?;
        next_eol(line)?;

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

        if default.is_some() {
            let line = lines.next().ok_or(ErrorKind::ExpectedLine)?.trim_end();
            let (word, line) = next_word(line)?;
            if word != "]" {
                todo!()
            }
            next_eol(line)?;
        }

        let f = Switch {
            register: register.into(),
            branches: branches.into(),
            default: default.map(|d| d.into()),
        };

        self.switches
            .try_insert(name.into(), f)
            .map(|_| ())
            .map_err(|_| todo!())
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
                    let (to, line) = next_word(line)?;
                    let (from, line) = next_word(line)?;
                    next_eol(line)?;
                    instructions.push(Instruction::Move {
                        to: to.into(),
                        from: from.into(),
                    });
                }
                "+" => {
                    let (to, line) = next_word(line)?;
                    let (value, line) = next_word(line)?;
                    next_eol(line)?;
                    instructions.push(Instruction::Set {
                        to: to.into(),
                        value: value.into(),
                    });
                }
                "{" => {
                    let (index, line) = next_word(line)?;
                    let (array, line) = next_word(line)?;
                    let (register, line) = next_word(line)?;
                    next_eol(line)?;
                    instructions.push(Instruction::ToArray {
                        index: index.into(),
                        array: array.into(),
                        register: register.into(),
                    });
                }
                "}" => {
                    let (index, line) = next_word(line)?;
                    let (register, line) = next_word(line)?;
                    let (array, line) = next_word(line)?;
                    next_eol(line)?;
                    instructions.push(Instruction::FromArray {
                        index: index.into(),
                        array: array.into(),
                        register: register.into(),
                    });
                }
                "|" => {
                    let (function, line) = next_word(line)?;
                    next_eol(line)?;
                    instructions.push(Instruction::Call {
                        function: function.into(),
                    });
                }
                "<" => {
                    next_eol(line)?;
                    break None;
                }
                "=" => {
                    let (next, line) = next_word(line)?;
                    next_eol(line)?;
                    break Some(next);
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
            .map_err(|_| todo!())
    }
}

impl<K, V> PrefixMap<K, V> {
    fn len(&self) -> usize {
        self.map.len()
    }
}

impl<K, V> PrefixMap<K, V>
where
    K: Ord + Borrow<str>,
{
    fn get<'a>(&self, key: &'a str) -> Option<(&V, &'a str)> {
        // https://users.rust-lang.org/t/is-it-possible-to-range-str-str-on-a-btreeset-string/93546/9
        // utter garbage
        use core::ops::Bound::*;
        let range = (Unbounded, Included(key));
        let (k, v) = self.map.range::<str, _>(range).next_back()?;
        let k = k.borrow();
        if key.len() < k.len() {
            return None;
        }
        let (pre, post) = key.split_at(k.len());
        if pre != k {
            return None;
        }
        Some((v, post))
    }

    fn contains(&self, key: &str) -> bool {
        // A ... AA ... [AB] ... ABC ... AC
        //
        // ^-------------^ conflict (1)
        //
        //               ^-------^ conflict (2)

        // TODO we'll probably need a custom datastructure to handle (1) efficiently

        // (1)
        for i in key.char_indices().map(|x| x.0).chain([key.len()]) {
            if self.map.contains_key(&key[..i]) {
                return true;
            }
        }

        // (2)
        use core::ops::Bound::*;
        let range = (Included(key), Unbounded);
        let Some(x) = self.map.range::<str, _>(range).next() else {
            return false;
        };
        x.0.borrow().starts_with(key)
    }

    fn try_insert(&mut self, key: K, value: V) -> Result<(), ()> {
        if self.contains(key.borrow()) {
            return Err(());
        }
        self.map
            .try_insert(key, value)
            .unwrap_or_else(|_| unreachable!());
        Ok(())
    }

    fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.map.iter()
    }
}

impl<K, V> Default for PrefixMap<K, V> {
    fn default() -> Self {
        Self {
            map: Default::default(),
        }
    }
}

impl<K, V> FromIterator<(K, V)> for PrefixMap<K, V>
where
    K: Ord + Borrow<str>,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut s = Self::default();
        for (k, v) in iter {
            s.try_insert(k, v).unwrap();
        }
        s
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

#[cfg(test)]
mod test {
    #[test]
    fn stuff() {
        let mut col = super::Collection::default();
        col.add_default_builtins().unwrap();
        //let s = include_str!("../examples/hello_world.pil");
        let s = include_str!("../examples/union.pil");
        //let s = include_str!("../examples/group.pil");
        //let s = include_str!("../examples/array.pil");
        col.parse_text("", s).unwrap();
        dbg!(&col);
        let mut prog = super::Program::from_collection(&col, &"start".to_string().into()).unwrap();
        dbg!(&prog);
        super::optimize::simple(&mut prog);
        dbg!(&prog);
        let vm = super::executor::wordvm::WordVM::from_program(&prog);
        dbg!(&vm);
        let mut exec = vm.create_state();
        loop {
            match exec
                .step(&vm)
                .inspect_err(|_| {
                    dbg!(&exec);
                })
                .unwrap()
            {
                super::executor::wordvm::Yield::Preempt => {}
                super::executor::wordvm::Yield::Finish => break,
                super::executor::wordvm::Yield::Sys { id } => {
                    let regs = vm.sys_registers(id);
                    match id {
                        super::BUILTIN_WRITE_CONSTANT_STRING => {
                            let offt = exec.register(regs[0]).unwrap();
                            let len = exec.register(regs[0] + 1).unwrap();
                            let s = &vm.strings_buffer()[offt as usize..][..len as usize];
                            let s = core::str::from_utf8(s).unwrap();
                            dbg!(s);
                        }
                        super::BUILTIN_WRITE_CHARACTER => {
                            let chr = exec.register(regs[0]).unwrap();
                            let chr = char::from_u32(chr).unwrap();
                            dbg!(chr);
                        }
                        super::BUILTIN_WRITE_INTEGER32 => {
                            let n = exec.register(regs[0]).unwrap();
                            dbg!(n as i32);
                        }
                        super::BUILTIN_WRITE_NATURAL32 => {
                            let n = exec.register(regs[0]).unwrap();
                            dbg!(n as u32);
                        }
                        super::BUILTIN_WRITE_NEWLINE => {
                            dbg!("newline");
                        }
                        id => todo!("{id}"),
                    }
                }
            }
        }
        todo!();
    }

    #[test]
    fn prefix_map() {
        let mut map = super::PrefixMap::<&str, u32>::default();

        map.try_insert("a.b", 42).unwrap();
        map.try_insert("a.b", 42).unwrap_err();
        map.try_insert("a.b.a", 42).unwrap_err();
        map.try_insert("a.b.c", 42).unwrap_err();

        let (v, post) = map.get("a.b.x").unwrap();
        assert_eq!(*v, 42);
        assert_eq!(post, ".x");

        let (v, post) = map.get("a.b.a").unwrap();
        assert_eq!(*v, 42);
        assert_eq!(post, ".a");

        assert!(map.get("a.a").is_none());
        assert!(map.get("a.c").is_none());
    }
}
