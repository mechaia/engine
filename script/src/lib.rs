#![feature(slice_split_once, map_try_insert, iterator_try_collect)]
#![deny(unused_must_use, elided_lifetimes_in_paths)]

use std::str::Lines;

mod executor;
mod program;
pub mod optimize;

//pub use util::str::PoolBoxU8 as Str;
pub use program::Program;

type Str = Box<str>;
type Map<K, V> = std::collections::HashMap<K, V>;

#[derive(Debug, Default)]
pub struct Collection {
    types: Map<Str, Type>,
    registers: Map<Str, Register>,
    switches: Map<Str, Switch>,
    functions: Map<Str, Function>,
}

#[derive(Debug)]
pub(crate) enum Type {
    User(Box<[Str]>),
    Builtin(BuiltinType),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum BuiltinType {
    Integer32,
    Natural32,
    ConstantString,
}

#[derive(Debug)]
pub(crate) enum Register {
    User(Str),
    Builtin(BuiltinRegister),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum BuiltinRegister {
    WriteConstantString,
    WriteCharacter,
    WriteInteger32,
    WriteNatural32,
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
    DuplicateRegister,
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
    Builtin(BuiltinFunction),
}

#[derive(Clone, Copy, Debug)]
enum BuiltinFunction {
    WriteConstantString,
    WriteCharacter,
    WriteInteger32,
    WriteNatural32,
    WriteNewline,
}

#[derive(Debug)]
enum Instruction {
    Move { to: Str, from: Str },
    Set { to: Str, value: Str },
    Call { function: Str },
}

impl Collection {
    pub fn add_default_builtins(&mut self) -> Result<(), Error> {
        let mut f = |k: &str, v| {
            self.types.try_insert(k.into(), Type::Builtin(v)).unwrap();
        };
        f("ConstantString", BuiltinType::ConstantString);
        f("Integer32", BuiltinType::Integer32);
        f("Natural32", BuiltinType::Natural32);

        let mut f = |k: &str, v| {
            self.registers
                .try_insert(k.into(), Register::Builtin(v))
                .unwrap();
        };
        f(
            "write_constant_string_value",
            BuiltinRegister::WriteConstantString,
        );
        f("write_character_value", BuiltinRegister::WriteCharacter);
        f("write_integer32_value", BuiltinRegister::WriteInteger32);
        f("write_natural32_value", BuiltinRegister::WriteNatural32);

        let mut f = |k: &str, v| {
            self.functions
                .try_insert(k.into(), Function::Builtin(v))
                .unwrap();
        };
        f(
            "write_constant_string",
            BuiltinFunction::WriteConstantString,
        );
        f("write_character", BuiltinFunction::WriteCharacter);
        f("write_integer32", BuiltinFunction::WriteInteger32);
        f("write_natural32", BuiltinFunction::WriteNatural32);
        f("write_newline", BuiltinFunction::WriteNewline);

        Ok(())
    }

    pub fn parse_text(&mut self, prefix: &str, text: &str) -> Result<(), Error> {
        let mut lines = text.lines();
        while let Some(line) = lines.next() {
            let line = line.trim_end();
            let Ok((word, line)) = next_word(line) else {
                continue;
            };
            match word {
                "%" => self.line_parse_type(line),
                "$" => self.line_parse_register(line),
                "?" => self.line_parse_switch(&mut lines, line),
                ">" => self.line_parse_function(&mut lines, line),
                tk => todo!(),
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
        let ty = Type::User(variants.into());
        self.types
            .try_insert(name.into(), ty)
            .map_err(|_| ErrorKind::DuplicateType)
            .map(|_| ())
    }

    fn line_parse_register(&mut self, line: &str) -> Result<(), ErrorKind> {
        let (name, line) = next_word(line)?;
        let (ty, line) = next_word(line)?;
        next_eol(line)?;
        let reg = Register::User(ty.into());
        self.registers
            .try_insert(name.into(), reg)
            .map_err(|_| ErrorKind::DuplicateRegister)
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
                "~" => {
                    let (value, line) = next_word(line)?;
                    let (function, line) = next_word(line)?;
                    branches.push((value.into(), function.into()));
                    next_eol(line)?;
                }
                "!" => {
                    next_eol(line)?;
                    break None;
                }
                "@" => {
                    let (name, line) = next_word(line)?;
                    next_eol(line)?;
                    break Some(name);
                }
                tk => todo!(),
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
            .map_err(|_| todo!())
    }

    fn line_parse_function(&mut self, lines: &mut Lines<'_>, line: &str) -> Result<(), ErrorKind> {
        let (name, line) = next_word(line)?;
        next_eol(line)?;

        let mut instructions = Vec::new();

        let next = loop {
            let line = lines.next().ok_or(ErrorKind::ExpectedLine)?.trim_end();
            let Ok((word, line)) = next_word(line) else {
                continue;
            };
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
                tk => todo!(),
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

impl Switch {}

impl Function {}

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
        //let s = include_str!("../examples2/hello_world.pil");
        let s = include_str!("../examples2/union.pil");
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
            match exec.step(&vm).inspect_err(|_| { dbg!(&exec); }).unwrap() {
                super::executor::wordvm::Yield::Preempt => {}
                super::executor::wordvm::Yield::Finish => break,
                super::executor::wordvm::Yield::Sys { id } => {
                    match vm.sys_id_to_builtin(id).inspect_err(|_| { dbg!(&exec); }).unwrap() {
                        super::BuiltinFunction::WriteConstantString => {
                            let reg = vm.builtin_register_map().write_constant_string;
                            let offt = exec.register(reg).unwrap();
                            let len = exec.register(reg + 1).unwrap();
                            if offt == 42 { dbg!(&exec); }
                            let s = &vm.strings_buffer()[offt as usize..][..len as usize];
                            let s = core::str::from_utf8(s).unwrap();
                            dbg!(s);
                        }
                        super::BuiltinFunction::WriteCharacter => {
                            let reg = vm.builtin_register_map().write_character;
                            let chr = exec.register(reg).unwrap();
                            let chr = char::from_u32(chr).unwrap();
                            dbg!(chr);
                        }
                        super::BuiltinFunction::WriteInteger32 => {
                            let reg = vm.builtin_register_map().write_integer32;
                            let n = exec.register(reg).unwrap();
                            dbg!(n as i32);
                        }
                        super::BuiltinFunction::WriteNatural32 => {
                            let reg = vm.builtin_register_map().write_natural32;
                            let n = exec.register(reg).unwrap();
                            dbg!(n as u32);
                        }
                        super::BuiltinFunction::WriteNewline => {
                            dbg!("newline");
                        }
                    }
                }
            }
        }
        todo!();
    }
}
