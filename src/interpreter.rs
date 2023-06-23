use std::{collections::HashMap};

use crate::ast::*;

pub struct Interpreter {
    variables: HashMap<String, Expr>
}

impl Interpreter {
    pub fn new() -> Self {
        Self { variables: HashMap::new() }
    }
    
    pub fn interpret(&mut self, statements: &Vec<Stmt>) -> Result<i32, String> {
        let mut result = 0;  // Program exits with status code (default of 0)
        for statement in statements {
            result = self.evaluate_stmt(&statement)?;
        }
        Ok(result)
    }

    pub fn interpret_one(&mut self, statement: &Stmt) -> Result<i32, String> {
        self.evaluate_stmt(&statement)
    }
    
    fn evaluate_stmt(&mut self, stmt: &Stmt) -> Result<i32, String> {
        let result = match stmt {
            Stmt::Print(expr) => {
                let result = self.evalulate_expr(expr);
                println!("Print: {:?}", result.unwrap());
                0
            },
            Stmt::Expr(expr) => {
                self.evalulate_expr(expr)?
            },
            Stmt::If {cond, then} => {
                match self.evalulate_expr(cond).unwrap() {
                    0 => self.interpret_one(then).unwrap(),
                    _ => 1
                }
            }
            Stmt::Block(body) => {
                self.interpret(body).unwrap()
            }
        };
        Ok(result)
    }
                 
    fn evalulate_expr(&mut self, expr: &Expr) -> Result<i32, String> {
        let result: i32 = match expr {
            Expr::Int(n)   => *n,
            Expr::Boolean(true)  => 0,
            Expr::Boolean(false) => 1,
            Expr::Ident(s) => {
                // Removing map from interpreter then resetting.
                // Avoids immutable AND mutable borrow of 'self' in a single scope
                // Revisit: feels improper
                let vars = std::mem::take(&mut self.variables);
                let result = match vars.get(s) {
                    Some(expr) => self.evalulate_expr(expr)?,
                    None => return Err(format!("Undefined variable {}", s))
                };
                self.variables = vars;  // reset/restore variable map

                result
            },
            Expr::Monadic { operator, operand } => {
                let next = self.evalulate_expr(operand)?;
                match operator {
                    Operator::Plus => next,
                    Operator::Minus => -next,
                    _ => {
                        return Err(format!("Bad operator: {:?} onto {}", operator, next))
                    }
                }
            },
            Expr::Dyadic { operator, left, right } => {
                let lhs = self.evalulate_expr(left)?;
                let rhs = self.evalulate_expr(right)?;
                match operator {
                    Operator::Plus    => lhs + rhs,
                    Operator::Minus   => lhs - rhs,
                    Operator::Star    => lhs * rhs,
                    Operator::Slash   => lhs / rhs,
                    Operator::EqualTo => if lhs == rhs {0} else {1},
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

        assert_eq!(0, interpreter.interpret_one(&Stmt::Expr(Expr::Int(0))).unwrap());  // 0
        assert_eq!(1, interpreter.interpret_one(&Stmt::Expr(  // +1
            Expr::Monadic {
                operator: Operator::Plus,
                operand: Box::new(Expr::Int(1))
            })).unwrap());
        assert_eq!(-1, interpreter.interpret_one(&Stmt::Expr(  // -1
            Expr::Monadic {
                operator: Operator::Minus,
                operand: Box::new(Expr::Int(1))
            })).unwrap());
        assert_eq!(256, interpreter.interpret_one(&Stmt::Expr(  // 192 + 64
            Expr::Dyadic {
                operator: Operator::Plus,
                left: Box::new(Expr::Int(192)),
                right: Box::new(Expr::Int(64))
            })).unwrap());
        assert_eq!(-8, interpreter.interpret_one(&Stmt::Expr(  // 16 / 4 * -2
            Expr::Dyadic {
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
            })).unwrap());
    }

    #[test]
    fn variables() {
        let mut interpreter = Interpreter::new();
        
        assert_eq!(Err("Undefined variable my_var".into()),
                   interpreter.interpret_one(&Stmt::Expr(Expr::Ident("my_var".into()))));
    }
}