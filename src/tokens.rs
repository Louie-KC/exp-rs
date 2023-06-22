#[derive(Debug, PartialEq)]
pub enum Token {
    // Literal
    Int(i32),
    Ident(String),

    // Operator
    Plus,
    Minus,
    Star,
    Slash,
    Equal,

    // Keyword
    Let,
    Function,
    Print,

    // Grouping
    Comma,
    LParen,
    RParen,
    LSquirly,
    RSquirly,

    // Etc
    Colon,
    EndLine,
    EOF
}