#[derive(Debug, PartialEq)]
pub enum Token {
    Int(i32),
    Plus,
    Minus,
    Star,
    Slash,
    Ident(String),
    Equal,
    Let,
    Function,
    Colon,
    Comma,
    LParen,
    RParen,
    LSquirly,
    RSquirly,
    EndLine,
}