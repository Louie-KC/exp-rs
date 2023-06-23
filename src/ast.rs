#[derive(Debug, PartialEq)]
pub enum Stmt {
    Expr(Expr),
    Print(Expr),
    // If(Expr, Box<Stmt>),
    If{
        cond: Expr,
        then: Box<Stmt>
    },
    Block(Vec<Stmt>),
}

#[derive(Debug, PartialEq)]
pub enum Operator {
    Plus,
    Minus,
    Star,
    Slash,
    // Assign,
    EqualTo,
}

#[derive(Debug, PartialEq)]
pub enum Expr {
    Int(i32),
    Ident(String),
    Boolean(bool),
    Monadic {
        operator: Operator,
        operand: Box<Expr>
    },
    Dyadic {
        operator: Operator,
        left: Box<Expr>,
        right: Box<Expr>
    },
}