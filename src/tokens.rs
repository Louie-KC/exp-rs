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
    Percent,
    Equal,
    Negate,
    Or,
    And,

    // Comparator
    EqualTo,
    LessThan,
    LessEquals,
    GreaterThan,
    GreaterEquals,

    // Keyword
    Var,
    Let,
    Function,
    Print,
    If,
    Else,
    While,

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