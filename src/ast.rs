#[derive(Debug, PartialEq)]
pub enum Operator {
    Plus,
    Minus,
}

#[derive(Debug, PartialEq)]
pub enum Expr {
    Int(i32),
    Monadic {
        operator: Operator,
        operand: Box<Expr>
    },
    Dyadic {
        operator: Operator,
        left: Box<Expr>,
        right: Box<Expr>
    }
}