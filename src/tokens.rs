#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    // Literal
    Int(i32),
    Ident(String),
    Boolean(bool),

    // Operator
    Plus,
    Minus,
    Star,
    Slash,
    Equal,
    Negate,

    // Comparator
    EqualTo,
    LessThan,
    LessEquals,
    GreaterThan,
    GreaterEquals,

    // Keyword
    Let,
    Function,
    Print,
    If,

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