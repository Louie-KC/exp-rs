#[derive(Debug, PartialEq)]
pub enum Stmt {
    Expr(Expr),
    Print(Expr)
}

#[derive(Debug, PartialEq)]
pub enum Operator {
    Plus,
    Minus,
    Star,
    Slash,
}

#[derive(Debug, PartialEq)]
pub enum Expr {
    Int(i32),
    Ident(String),
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