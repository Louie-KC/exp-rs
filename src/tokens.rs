#[derive(Debug, PartialEq)]
pub enum Token {
    Int(i32),
    Plus,
    Minus
}