#![feature(slice_split_once, map_try_insert, iterator_try_collect)]
#![deny(unused_must_use)]

mod function;
//mod program;
mod token;
mod executor;

//pub use util::str::PoolBoxU8 as Str;
//pub use program::Program;

pub(crate) use function::Function;

use token::{Token, Tokenizer};

type Str = Box<str>;
type Map<K, V> = std::collections::HashMap<K, V>;

#[derive(Debug)]
pub enum Record {
    User { fields: Map<Str, Box<[Str]>> },
    Opaque { size: u32 },
}

#[derive(Debug, Default)]
pub struct Collection {
    data: Map<Str, Record>,
    functions: Map<Str, Vec<Function>>,
}

#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
}

#[derive(Debug)]
enum ErrorKind {
    UnterminatedString,
    ExpectedIdentifier,
    UnexpectedToken {
        got: Option<Token>,
        expected: Option<Option<Token>>,
    },
    DuplicateFunction,
    DuplicateObject,
}

impl Collection {
    pub fn parse_text(&mut self, prefix: &str, text: &str) -> Result<(), Error> {
        let mut tokens = Tokenizer::from_text(text);

        while let Some(tk) = tokens.next().transpose()? {
            let mut name = || {
                match tokens.next().transpose()? {
                    Some(Token::Ident(s)) => Ok(s),
                    tk => Err(Error::unexpected(tk)),
                }
            };

            match tk {
                Token::Function => {
                    let name = name()?;
                    let f = Function::parse_tokens(&mut tokens)?;
                    let path = [prefix, &*name].concat();
                    self.functions.entry(path.into()).or_default().push(f);
                }
                Token::Record => {
                    let name = name()?;
                    todo!()
                }
                Token::Object => {
                    let name = name()?;
                    todo!()
                }
                tk => return Err(Error::unexpected(Some(tk))),
            }
        }

        Ok(())
    }

    pub fn add_record(&mut self, name: &str, record: Record) -> Result<(), Error> {
        self.data.try_insert(name.into(), record).map_err(|_| Error { kind: ErrorKind::DuplicateObject }).map(|_| ())
    }
}

impl Error {
    fn expected(expected: Token, got: Option<Token>) -> Self {
        Self {
            kind: ErrorKind::UnexpectedToken {
                got,
                expected: Some(Some(expected)),
            },
        }
    }

    fn expected_eof(got: Option<Token>) -> Self {
        Self {
            kind: ErrorKind::UnexpectedToken {
                got,
                expected: Some(None),
            },
        }
    }

    fn unexpected(got: Option<Token>) -> Self {
        Self {
            kind: ErrorKind::UnexpectedToken {
                got,
                expected: None,
            },
        }
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn stuff() {
        let mut col = super::Collection::default();
        let s = include_str!("../examples/hello_world.pil");
        col.parse_text("", s).unwrap();
        col.add_record("Env", super::Record::Opaque { size: 0 }).unwrap();
        dbg!(&col);
        //let prog = super::Program::from_collection(&col);
        //dbg!(&prog);
        todo!();
    }
}
