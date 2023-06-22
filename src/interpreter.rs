use std::{collections::HashMap};

use crate::ast::*;

pub struct Interpreter {
    variables: HashMap<String, Expr>
}

impl Interpreter {
    pub fn new() -> Self {
        Self { variables: HashMap::new() }
    }

    pub fn interpret(&mut self, expr: Expr) -> Result<i32, String> {
        self.evalulate(&expr)
    }

    // leftover notes:  Groupings with parentheses?
    //                  Sub-expressions with {}                    
    fn evalulate(&mut self, expr: &Expr) -> Result<i32, String> {
        let result: i32 = match expr {
            Expr::Int(n) => *n,
            Expr::Ident(s) => {
                // Removing map from interpreter then resetting.
                // Avoids immutable AND mutable borrow of 'self' in a single scope
                // Revisit: feels improper
                let vars = std::mem::take(&mut self.variables);
                let result = match vars.get(s) {
                    Some(expr) => self.evalulate(expr)?,
                    None => return Err(format!("Undefined variable {}", s))
                };
                self.variables = vars;  // reset

                result
            },
            Expr::Monadic { operator, operand } => {
                let next = self.evalulate(operand)?;
                match operator {
                    Operator::Plus => next,
                    Operator::Minus => -next,
                    _ => {
                        return Err(format!("Bad operator: {:?} onto {}", operator, next))
                    }
                }
            },
            Expr::Dyadic { operator, left, right } => {
                let lhs = self.evalulate(left)?;
                let rhs = self.evalulate(right)?;
                match operator {
                    Operator::Plus  => lhs + rhs,
                    Operator::Minus => lhs - rhs,
                    Operator::Star  => lhs * rhs,
                    Operator::Slash => lhs / rhs,
                }
            }
        };
        Ok(result)
    }
}


#[cfg(test)]
mod tests {
    use crate::ast::*;
    use crate::interpreter::Interpreter;

    #[test]
    fn basic_calculation() {
        let mut interpreter = Interpreter::new();

        assert_eq!(0, interpreter.interpret(Expr::Int(0)).unwrap());  // 0
        assert_eq!(1, interpreter.interpret(Expr::Monadic {  // +1
            operator: Operator::Plus,
            operand: Box::new(Expr::Int(1))
        }).unwrap());
        assert_eq!(-1, interpreter.interpret(Expr::Monadic {  // -1
            operator: Operator::Minus,
            operand: Box::new(Expr::Int(1))
        }).unwrap());
        assert_eq!(256, interpreter.interpret(Expr::Dyadic {  // 192 + 64
            operator: Operator::Plus,
            left: Box::new(Expr::Int(192)),
            right: Box::new(Expr::Int(64))
        }).unwrap());
        assert_eq!(-8, interpreter.interpret(Expr::Dyadic {  // 16 / 4 * -2
            operator: Operator::Star,
            left: Box::new(Expr::Dyadic {
                operator: Operator::Slash,
                left: Box::new(Expr::Int(16)),
                right: Box::new(Expr::Int(4))
            }),
            right: Box::new(Expr::Monadic {
                operator: Operator::Minus,
                operand: Box::new(Expr::Int(2))
            })
        }).unwrap());
    }

    #[test]
    fn variables() {
        let mut interpreter = Interpreter::new();
        
        assert_eq!(Err("Undefined variable my_var".into()),
                   interpreter.interpret(Expr::Ident("my_var".into())));
    }
}