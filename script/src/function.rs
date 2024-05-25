use crate::{token::Tokenizer, Error, Str, Token};

#[derive(Clone, Debug)]
pub(crate) struct Type(Str);

#[derive(Clone, Debug)]
pub(crate) struct Variant(Str);

#[derive(Clone, Debug)]
pub(crate) struct Register(i32);

#[derive(Debug)]
pub(crate) struct Function {
    pub input: Box<[Type]>,
    pub output: Box<[Type]>,
    pub arms: Box<[Arm]>,
    pub debug: Option<Debug>,
}

#[derive(Debug)]
pub(crate) struct Arm {
    pub pattern: Box<[Variant]>,
    pub registers: Box<[Type]>,
    pub calls: Box<[Call]>,
}

#[derive(Debug)]
pub(crate) struct Call {
    pub function: Str,
    pub input: Box<[Register]>,
    pub output: Box<[Register]>,
}

#[derive(Debug)]
pub(crate) struct Debug {
    pub arg_names: Box<[Str]>,
    pub arms: Box<[DebugArm]>,
}

#[derive(Debug)]
pub(crate) struct DebugArm {
    pub register_names: Box<[Str]>,
}

#[derive(Debug)]
pub(crate) enum Value {
    Register(Str),
    String(Str),
}

impl Function {
    pub fn parse_tokens(tokens: &mut Tokenizer) -> Result<Self, Error> {
        let mut input = Vec::new();
        let mut output = Vec::new();
        let mut arg_names = Vec::new();

        parse_args(None, tokens, &mut |tokens: &mut Tokenizer<'_>| {
            let typ = match tokens.next().transpose()? {
                Some(Token::Ident(s)) => s,
                tk => return Err(Error::unexpected(tk)),
            };
            let name = match tokens.next().transpose()? {
                Some(Token::Ident(s)) => s,
                tk => return Err(Error::unexpected(tk)),
            };
            input.push(typ);
            arg_names.push(name);
            Ok(None)
        })?;

        match tokens.next().transpose()? {
            Some(Token::Returns) => {},
            tk => return Err(Error::expected(Token::Returns, tk)),
        }

        parse_args(None, tokens, &mut |tokens: &mut Tokenizer<'_>| {
            let name = match tokens.next().transpose()? {
                Some(Token::Ident(s)) => s,
                tk => return Err(Error::unexpected(tk)),
            };
            match tokens.next().transpose()? {
                Some(Token::Descriptor) => {}
                tk => return Err(Error::expected(Token::Descriptor, tk)),
            }
            let typ = match tokens.next().transpose()? {
                Some(Token::Ident(s)) => s,
                tk => return Err(Error::unexpected(tk)),
            };
            output.push(typ);
            Ok(None)
        })?;

        let (arms, arms_debug) = parse_arms(tokens)?;

        let slf = Self {
            input,
            output,
            arms: arms.into(),
            debug: Some(Debug {
                arg_names: arg_names.into(),
                arms: arms_debug.into(),
            }),
        };

        Ok(slf)
    }
}

fn parse_args(
    pre_token: Option<Token>,
    tokens: &mut Tokenizer,
    field_handler: &mut dyn FnMut(&mut Tokenizer) -> Result<Option<Token>, Error>,
) -> Result<(), Error> {
    match pre_token.map(Ok).or_else(|| tokens.next()).transpose()? {
        Some(Token::GroupStart) => {}
        tk => return Err(Error::expected(Token::GroupStart, tk)),
    }

    loop {
        match tokens.next().transpose()? {
            Some(Token::GroupEnd) => break,
            tk => return Err(Error::unexpected(tk)),
        };
        let tk = (field_handler)(tokens)?;
        match tk.map(Ok).or_else(|| tokens.next()).transpose()? {
            Some(Token::GroupEnd) => break,
            Some(Token::ElementSeparator) => {}
            tk => return Err(Error::unexpected(tk)),
        }
    }

    Ok(())
}

fn parse_arms(tokens: &mut Tokenizer) -> Result<(Vec<Arm>, Vec<DebugArm>), Error> {
    parse_args(None, tokens, &mut |tokens: &mut Tokenizer<'_>| {
        let typ = match tokens.next().transpose()? {
            Some(Token::Ident(s)) => s,
            tk => return Err(Error::unexpected(tk)),
        };
        let name = match tokens.next().transpose()? {
            Some(Token::Ident(s)) => s,
            tk => return Err(Error::unexpected(tk)),
        };
        input.push(typ);
        arg_names.push(name);
        Ok(None)
    })?;
}

fn parse_statements(tokens: &mut Tokenizer) -> Result<Vec<Statement>, Error> {
    let mut statements = Vec::new();

    match tokens.next().transpose()? {
        Some(Token::ScopeStart) => {}
        tk => return Err(Error::expected(Token::ScopeEnd, tk)),
    }

    loop {
        match tokens.next().transpose()? {
            Some(Token::ScopeEnd) => break,
            Some(Token::Ident(name)) => {
                let (function, tk) = parse_path(name, tokens)?;
                let mut args = Vec::new();
                parse_args(tk, tokens, &mut |mode, tokens| {
                    let (value, tk) = match tokens.next().transpose()? {
                        Some(Token::Ident(s)) => {
                            let (path, tk) = parse_path(s, tokens)?;
                            (Value::Register(path), tk)
                        }
                        Some(Token::Str(s)) => (Value::String(s), None),
                        tk => return Err(Error::unexpected(tk)),
                    };
                    args.push((mode, value));
                    Ok(tk)
                })?;
                statements.push(Statement::Call {
                    function,
                    args: args.into(),
                });
            }
            Some(Token::Const) => {
                todo!();
            }
            Some(Token::Var) => {
                todo!();
            }
            tk => return Err(Error::unexpected(tk)),
        };
    }

    Ok(statements)
}

fn parse_path(start: Str, tokens: &mut Tokenizer) -> Result<(Str, Option<Token>), Error> {
    let mut path = Vec::new();

    path.push(start);
    let tk = loop {
        match tokens.next().transpose()? {
            Some(Token::MemberRef) => {}
            tk => break tk,
        };
        let member = match tokens.next().transpose()? {
            Some(Token::Ident(s)) => s,
            tk => return Err(Error::unexpected(tk)),
        };
        path.push(member);
    };
    Ok((path.join(".").into(), tk))
}
