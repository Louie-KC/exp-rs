use std::{collections::HashMap};

use crate::ast::*;

pub struct Interpreter {
    variables: HashMap<String, Box<Stmt>>
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
                println!("{:?}", result.unwrap());
                0
            },
            Stmt::Expr(expr) => {
                self.evalulate_expr(expr)?
            },
            Stmt::If {cond, then, els} => {
                match self.evalulate_expr(cond).unwrap() {
                    0 => self.interpret_one(then).unwrap(),
                    _ => self.interpret_one(els).unwrap()
                }
            },
            Stmt::While { cond, body } => {
                let mut result = 0;
                while self.evalulate_expr(cond)? == 0 {
                    result = self.interpret_one(&body)?
                }
                result
            },
            Stmt::Block(body) => {
                self.interpret(body).unwrap()
            },
            Stmt::VarDecl(ident, value) => {
                let val = match value {
                    Some(v) => v.clone(),
                    None => Box::new(Stmt::Expr(Expr::Int(0)))
                };
                self.variables.insert(ident.into(), val);
                // println!("variables after insert/update: {:?}", self.variables);
                0
            },
        };
        Ok(result)
    }
                 
    fn evalulate_expr(&mut self, expr: &Expr) -> Result<i32, String> {
        // println!("evaluate_expr: {:?}", expr);
        let result: i32 = match expr {
            Expr::Int(n)   => *n,
            Expr::Boolean(true)  => 0,
            Expr::Boolean(false) => 1,
            Expr::Ident(s) => {
                match self.retrieve_ident(s) {
                    Some(stmt) => self.interpret_one(&stmt).unwrap(),
                    None => return Err(format!("Undefined variable {}", s))
                }
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
                    Operator::Modulo  => lhs % rhs,
                    Operator::EqualTo       => if lhs == rhs {0} else {1},
                    Operator::NotEqualTo    => if lhs != rhs {0} else {1},
                    Operator::LessThan      => if lhs <  rhs {0} else {1},
                    Operator::LessEquals    => if lhs <= rhs {0} else {1},
                    Operator::GreaterThan   => if lhs >  rhs {0} else {1},
                    Operator::GreaterEquals => if lhs >= rhs {0} else {1},
                    Operator::LogicalAnd
                    | Operator::LogicalOr   => panic!("Logical operation in Dyadic expression"),
                }
            },
            Expr::Logical { operator, left, right } => {
                let lhs = self.evalulate_expr(left)?;
                match (lhs, operator) {  // short circuit eval
                    (1, Operator::LogicalAnd)  => lhs,
                    (0, Operator::LogicalOr)   => lhs,
                    (_, Operator::LogicalAnd)
                    | (_, Operator::LogicalOr) => self.evalulate_expr(right)?,
                    _ => panic!("Non-logical operator in Logical expression")
                }

            }
            Expr::Assign { var_name, new_value } => {
                let val = self.evalulate_expr(new_value)?;
                match self.variables.contains_key(var_name) {
                    true  => self.variables.insert(var_name.into(), Box::new(Stmt::Expr(Expr::Int(val)))),
                    false => panic!("Cannot assign to {} as it is not declared", var_name),
                };
                val
            },
        };
        Ok(result)
    }

    fn retrieve_ident(&mut self, name: &String) -> Option<Box<Stmt>> {
        self.variables.get(name).cloned()
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
        
        // my_var;
        assert_eq!(Err("Undefined variable my_var".into()),
                   interpreter.interpret_one(&Stmt::Expr(Expr::Ident("my_var".into()))));

        // var not_initialised;
        assert_eq!(0, interpreter.interpret(&vec![
            Stmt::VarDecl("not_initialised".into(), None),
            Stmt::Expr(Expr::Ident("not_initialised".into()))
        ]).unwrap());

        // var tomato = 127;
        assert_eq!(127, interpreter.interpret(&vec![
            Stmt::VarDecl("tomato".into(), Some(Box::new(Stmt::Expr(Expr::Int(127))))),
            Stmt::Expr(Expr::Ident("tomato".into()))
        ]).unwrap());

        // var b = if (false) { 1; } else { 1023; };
        assert_eq!(1023, interpreter.interpret(&vec![
            Stmt::VarDecl("b".into(), Some(Box::new(Stmt::If {
                cond: Expr::Boolean(false),
                then: Box::new(Stmt::Expr(Expr::Int(1))),
                els: Box::new(Stmt::Expr(Expr::Int(1023)))
            }))),
            Stmt::Expr(Expr::Ident("b".into()))
        ]).unwrap());

        // var a = 10;
        // var b = 20;
        // a = a + b;
        // a;
        assert_eq!(30, interpreter.interpret(&vec![
            Stmt::VarDecl("a".into(), Some(Box::new(Stmt::Expr(Expr::Int(10))))),
            Stmt::VarDecl("b".into(), Some(Box::new(Stmt::Expr(Expr::Int(20))))),
            Stmt::Expr(Expr::Assign {
                var_name: "a".into(),
                new_value: Box::new(Expr::Dyadic {
                    operator: Operator::Plus,
                    left: Box::new(Expr::Ident("a".into())),
                    right: Box::new(Expr::Ident("b".into()))
                })
            }),
            Stmt::Expr(Expr::Ident("a".into()))
        ]).unwrap());
    }

    #[test]
    fn branch() {
        let mut interpreter = Interpreter::new();

        // if (256 == 256) { 1; } else { 2; }
        assert_eq!(1, interpreter.interpret(
            &vec![
                Stmt::If {
                    cond: Expr::Dyadic {
                        operator: Operator::EqualTo,
                        left: Box::new(Expr::Int(256)),
                        right: Box::new(Expr::Int(256))
                    },
                    then: Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(1))])),
                    els: Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(2))]))
                }]
        ).unwrap());

        // if (512 != 512) { 1; } else { 2; }
        assert_eq!(2, interpreter.interpret(
            &vec![
                Stmt::If {
                    cond: Expr::Dyadic {
                        operator: Operator::NotEqualTo,
                        left: Box::new(Expr::Int(512)),
                        right: Box::new(Expr::Int(512))
                    },
                    then: Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(1))])),
                    els: Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(2))]))
                }]
        ).unwrap());

        // if (512 == 2048) { 11; }  // Default output of 0 without any statements
        assert_eq!(0, interpreter.interpret(
            &vec![
                Stmt::If {
                    cond: Expr::Dyadic {
                        operator: Operator::EqualTo,
                        left: Box::new(Expr::Int(512)),
                        right: Box::new(Expr::Int(2048))
                    },
                    then: Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(11))])),
                    els: Box::new(Stmt::Block(vec![]))
                }]
        ).unwrap());
        
        // var a = 5;
        // var b = 50;
        // var max = if (a > b) { a; } else { b; };
        // max;
        assert_eq!(50, interpreter.interpret(
            &vec![
                Stmt::VarDecl("a".into(), Some(Box::new(Stmt::Expr(Expr::Int(5))))),
                Stmt::VarDecl("b".into(), Some(Box::new(Stmt::Expr(Expr::Int(50))))),
                Stmt::VarDecl("max".into(), Some(Box::new(Stmt::If {
                    cond: Expr::Dyadic {
                        operator: Operator::GreaterThan,
                        left: Box::new(Expr::Ident("a".into())),
                        right: Box::new(Expr::Ident("b".into()))
                    },
                    then: Box::new(Stmt::Expr(Expr::Ident("a".into()))),
                    els: Box::new(Stmt::Expr(Expr::Ident("b".into())))
                }))),
                Stmt::Expr(Expr::Ident("max".into()))
                ]
        ).unwrap());

        // if (true && true) { 1; } else { 2; }
        assert_eq!(1, interpreter.interpret(
            &vec![
                Stmt::If {
                    cond: Expr::Logical {
                        operator: Operator::LogicalAnd,
                        left: Box::new(Expr::Boolean(true)),
                        right: Box::new(Expr::Boolean(true))
                    },
                    then: Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(1))])),
                    els: Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(2))]))
                }]
        ).unwrap());

        // if (true && false) { 1; } else { 2; }
        assert_eq!(2, interpreter.interpret(
            &vec![
                Stmt::If {
                    cond: Expr::Logical {
                        operator: Operator::LogicalAnd,
                        left: Box::new(Expr::Boolean(true)),
                        right: Box::new(Expr::Boolean(false))
                    },
                    then: Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(1))])),
                    els: Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(2))]))
                }]
        ).unwrap());

    }

    #[test]
    fn short_circuit_eval() {
        let mut interpreter = Interpreter::new();

        // // Short circuit evaluation
        // var flag = false;  // true = 0, false = 1
        // if (false && flag = true) {}
        // flag;  // should remain as false/1
        assert_eq!(1, interpreter.interpret(
            &vec![
                Stmt::VarDecl("flag".into(), Some(Box::new(Stmt::Expr(Expr::Boolean(false))))),
                Stmt::If {
                    cond: Expr::Logical {
                        operator: Operator::LogicalAnd,
                        left: Box::new(Expr::Boolean(false)),
                        right: Box::new(Expr::Assign {
                            var_name: "flag".into(),
                            new_value: Box::new(Expr::Boolean(true))
                        })
                    },
                    then: Box::new(Stmt::Block(vec![])),
                    els: Box::new(Stmt::Block(vec![]))
                },
                Stmt::Expr(Expr::Ident("flag".into()))
            ]
        ).unwrap());

        // // Short circuit evaluation
        // var flag = false;  // true = 0, false = 1
        // if (true || flag = true) {}
        // flag;  // should remain as false/1
        assert_eq!(1, interpreter.interpret(
            &vec![
                Stmt::VarDecl("flag".into(), Some(Box::new(Stmt::Expr(Expr::Boolean(false))))),
                Stmt::If {
                    cond: Expr::Logical {
                        operator: Operator::LogicalOr,
                        left: Box::new(Expr::Boolean(true)),
                        right: Box::new(Expr::Assign {
                            var_name: "flag".into(),
                            new_value: Box::new(Expr::Boolean(true))
                        })
                    },
                    then: Box::new(Stmt::Block(vec![])),
                    els: Box::new(Stmt::Block(vec![]))
                },
                Stmt::Expr(Expr::Ident("flag".into()))
            ]
        ).unwrap());
    }

    #[test]
    fn loops() {
        let mut interpreter = Interpreter::new();

        // var i = 0;
        // while (i < 5) {
        //     i = i + 1;
        // }
        // i;
        assert_eq!(5, interpreter.interpret(
            &vec![
                Stmt::VarDecl("i".into(), Some(Box::new(Stmt::Expr(Expr::Int(0))))),
            Stmt::While {
                cond: Expr::Dyadic {
                    operator: Operator::LessThan,
                    left: Box::new(Expr::Ident("i".into())),
                    right: Box::new(Expr::Int(5))
                }, body: Box::new(Stmt::Block(vec![
                    Stmt::Expr(Expr::Assign {
                        var_name: "i".into(),
                        new_value: Box::new(Expr::Dyadic {
                            operator: Operator::Plus,
                            left: Box::new(Expr::Ident("i".into())),
                            right: Box::new(Expr::Int(1))
                        })
                    })
                ]))
            },
            Stmt::Expr(Expr::Ident("i".into()))
            ]
        ).unwrap());
    }
}