#[derive(Debug, PartialEq, Clone)]
pub enum Stmt {
    Expr(Expr),
    Print(Expr),
    If{
        cond: Expr,
        then: Box<Stmt>,
        els: Box<Stmt>
    },
    Block(Vec<Stmt>),
    VarDecl(String, Option<Box<Stmt>>)
}

#[derive(Debug, PartialEq, Clone)]
pub enum Operator {
    Plus,
    Minus,
    Star,
    Slash,
    Modulo,
    // Assign,
    EqualTo,
    NotEqualTo,
    LessThan,
    LessEquals,
    GreaterThan,
    GreaterEquals,
    LogicalOr,
    LogicalAnd,
}

#[derive(Debug, PartialEq, Clone)]
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
    Logical {
        operator: Operator,
        left: Box<Expr>,
        right: Box<Expr>
    }
}