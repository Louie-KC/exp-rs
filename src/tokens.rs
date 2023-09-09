#[derive(Debug, PartialEq, Clone)]
pub struct Token {
    pub line_num: u32,
    pub kind: TokenKind
}

#[derive(Debug, PartialEq, Clone)]
pub enum TokenKind {
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
    For,

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
