use crate::{Error, Str};

#[derive(Debug)]
pub(crate) enum Token {
    Ident(Str),
    Str(Str),
    Char(Str),
    Data,
    Function,
    Import,
    Export,
    Linear,
    Var,
    GroupStart,
    GroupEnd,
    ScopeStart,
    ScopeEnd,
    ElementSeparator,
    Descriptor,
    Less,
    Greater,
    Equal,
    Returns,
    Finish,
}

pub(crate) struct Tokenizer<'a> {
    text: &'a str,
}

impl<'a> Tokenizer<'a> {
    pub fn from_text(text: &'a str) -> Self {
        Self { text }
    }
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = Result<Token, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.text.is_empty() {
                return None;
            }
            let tk;
            let i = self
                .text
                .bytes()
                .position(|c| !is_ident(c))
                .map_or(self.text.len(), |i| i.max(1));
            (tk, self.text) = self.text.split_at(i);
            use Token::*;
            let tk = match tk {
                "\n" | " " | "\t" => continue,
                "\"" => {
                    let mut i = 0;
                    let tk = parse_string(self.text, '\"', &mut i);
                    self.text = &self.text[i..];
                    return Some(tk.map(Token::Str));
                }
                "\'" => {
                    let mut i = 0;
                    let tk = parse_string(self.text, '\'', &mut i);
                    self.text = &self.text[i..];
                    return Some(tk.map(Token::Char));
                }
                "(" => GroupStart,
                ")" => GroupEnd,
                "{" => ScopeStart,
                "}" => ScopeEnd,
                "," => ElementSeparator,
                ":" => Descriptor,
                "<" => Less,
                ">" => Greater,
                "=" => Equal,
                "->" => Returns,
                "@" => Finish,
                "data" => Data,
                "function" => Function,
                "import" => Import,
                "export" => Export,
                "linear" => Linear,
                s if s.bytes().all(is_ident) => Ident(s.into()),
                s => todo!("{:?}", s),
            };
            return Some(Ok(tk));
        }
    }
}

fn is_ident(c: u8) -> bool {
    matches!(c, b'a'..=b'z' | b'A'..=b'Z' | b'_' | b'0'..=b'9')
}

fn parse_string(text: &str, og_c: char, i: &mut usize) -> Result<Str, Error> {
    let mut s = String::new();
    let mut it = text.char_indices();
    loop {
        let Some((k, c)) = it.next() else { return Err(Error { kind: crate::ErrorKind::UnterminatedString }) };
        match c {
            '"' | '\'' if c == og_c => {
                *i = k + 1;
                return Ok(s.into());
            }
            '\\' => todo!("handle \\"),
            c => s.push(c),
        }
    }
}
