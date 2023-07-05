use std::{collections::HashMap, mem::discriminant};

use crate::ast::*;

struct Function {
    params: Vec<String>,
    body: Stmt
}

impl Function {
    fn new(parameters: Vec<String>, body: Stmt) -> Self {
        if discriminant(&body) != discriminant(&Stmt::Block(vec![])) {
            panic!("Function::new bad body statement")
        }
        Self { params: parameters, body: body }
    }
}

struct Environment {
    variables: HashMap<String, i32>,
    functions: HashMap<String, Function>
}

impl Environment {
    fn new() -> Self {
        Self { variables: HashMap::new(), functions: HashMap::new() }
    }
}

pub struct Interpreter {
    // env_stack: Vec<HashMap<String, i32>>
    env_stack: Vec<Environment>
}

impl Interpreter {
    pub fn new() -> Self {
        // Self { env_stack: vec![HashMap::new()] }
        Self { env_stack: vec![Environment::new()] }
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
        // println!("Evaluating: {:?}", stmt);
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
                let needs_own_env = self.block_needs_stack(&body);
                if needs_own_env {
                    self.env_stack.push(Environment::new());
                }
                let result = self.interpret(body).unwrap();
                if needs_own_env {
                    self.env_stack.pop();
                }
                result
            },
            Stmt::VarDecl(ident, value) => {
                if self.current_env_has(ident.into()) {
                    panic!("Variable \"{}\" already declared in current scope", ident)
                }
                let val = match value {
                    Some(v) => self.evaluate_stmt(v).unwrap(),
                    None => 0
                };
                self.add_var(ident.into(), val);
                0
            },
            Stmt::FnDecl { name, parameters, body } => {
                let fn_env = &mut self.env_stack.last_mut().unwrap().functions;
                if fn_env.contains_key(name) {
                    panic!("function \"{}\" already declared in scope", name)
                }
                let function = Function::new(parameters.clone(), *body.clone());
                fn_env.insert(name.into(), function);
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
                match self.get_var(s).cloned() {
                    Some(val) => val,
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
                match self.get_var(var_name) {
                    Some(_) => self.update_var(var_name.into(), val),
                    None    => panic!("Cannot assign to {} as it is not declared", var_name),
                };
                val
            },
            Expr::Call { callee, args } => {
                let mut function: Option<&Function> = None;
                for env in self.env_stack.iter().rev() {
                    function = env.functions.get(callee);
                    if function.is_some() {
                        break;
                    }
                }
                let (param_names, body) = (function.unwrap().params.clone(), function.unwrap().body.clone());

                if param_names.len() != args.len() {
                    panic!("Incorrect number of parameter arguments")
                }

                let mut block: Vec<Stmt> = param_names.into_iter()
                    .enumerate()
                    .map(|(i, n)| Stmt::VarDecl(n.into(), Some(Box::new(Stmt::Expr(args.get(i).unwrap().clone())))))
                    .collect();

                block.push(body);

                self.evaluate_stmt(&Stmt::Block(block)).unwrap()
                
                // 0
            },
        };
        Ok(result)
    }

    fn get_stack_size(&self) -> usize {
        self.env_stack.len()
    }

    /// Determines whether a block statement needs its own variable environment added onto the
    /// environment stack by checking if the statements of the block make any declarations.
    fn block_needs_stack(&self, stack_body: &Vec<Stmt>) -> bool {
        for stmt in stack_body {
            if discriminant(stmt) == discriminant(&Stmt::VarDecl("".into(), None)) {
                return true
            }
        }
        false
    }

    /// Searches for the variable specified by the 'name' parameter and returns
    /// its value.
    /// The variable is searched for beginning from the current/inner-most scope.
    fn get_var(&self, name: &String) -> Option<&i32> {
        // println!("get_from_env:");
        // dbg!(self.env_stack.clone());
        let mut result = None;
        // Begin outward search from inner most scope
        for env in self.env_stack.iter().rev() {
            result = env.variables.get(name);
            if result.is_some() { break; }
        }
        result
    }

    /// Determines whether a variable with the parameter 'name' already exists
    /// in the current/inner-most scope.
    fn current_env_has(&self, name: &String) -> bool {
        // println!("current_env_has:");
        // dbg!(self.env_stack.clone());
        match self.env_stack.last().unwrap().variables.get(name) {
            Some(_) => true,
            None    => false
        }
    }

    /// Updates the value of an existing variable with the 'name' parameter.
    /// The search for the specified variable works outwards from the inner most
    /// scope.
    fn update_var(&mut self, name: &String, value: i32) -> () {
        for env in self.env_stack.iter_mut().rev() {
            if env.variables.contains_key(name) {
                env.variables.insert(name.into(), value);
                return;
            }
        }
        panic!("Variable \"{}\" does not exist", name);
    }

    /// Adds/declares a variable with the value to the current or inner most scope.
    fn add_var(&mut self, name: &String, value: i32) -> () {
        match self.env_stack.last_mut() {
            Some(env) => { env.variables.insert(name.into(), value); },
            None    => panic!("All enviroments have been cleared")
        }
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
        // not_initialised;
        assert_eq!(0, interpreter.interpret(&vec![
            Stmt::VarDecl("not_initialised".into(), None),
            Stmt::Expr(Expr::Ident("not_initialised".into()))
        ]).unwrap());

        // var tomato = 127;
        // tomato;
        assert_eq!(127, interpreter.interpret(&vec![
            Stmt::VarDecl("tomato".into(), Some(Box::new(Stmt::Expr(Expr::Int(127))))),
            Stmt::Expr(Expr::Ident("tomato".into()))
        ]).unwrap());

        // var c = if (false) { 1; } else { 1023; };
        // c;
        assert_eq!(1023, interpreter.interpret(&vec![
            Stmt::VarDecl("c".into(), Some(Box::new(Stmt::If {
                cond: Expr::Boolean(false),
                then: Box::new(Stmt::Expr(Expr::Int(1))),
                els: Box::new(Stmt::Expr(Expr::Int(1023)))
            }))),
            Stmt::Expr(Expr::Ident("c".into()))
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
        // var flag_one = false;  // true = 0, false = 1
        // if (false && flag_one = true) {}
        // flag_one;  // should remain as false/1
        assert_eq!(1, interpreter.interpret(
            &vec![
                Stmt::VarDecl("flag_one".into(), Some(Box::new(Stmt::Expr(Expr::Boolean(false))))),
                Stmt::If {
                    cond: Expr::Logical {
                        operator: Operator::LogicalAnd,
                        left: Box::new(Expr::Boolean(false)),
                        right: Box::new(Expr::Assign {
                            var_name: "flag_one".into(),
                            new_value: Box::new(Expr::Boolean(true))
                        })
                    },
                    then: Box::new(Stmt::Block(vec![])),
                    els: Box::new(Stmt::Block(vec![]))
                },
                Stmt::Expr(Expr::Ident("flag_one".into()))
            ]
        ).unwrap());

        // // Short circuit evaluation
        // var flag_two = false;  // true = 0, false = 1
        // if (true || flag_two = true) {}
        // flag_two;  // should remain as false/1
        assert_eq!(1, interpreter.interpret(
            &vec![
                Stmt::VarDecl("flag_two".into(), Some(Box::new(Stmt::Expr(Expr::Boolean(false))))),
                Stmt::If {
                    cond: Expr::Logical {
                        operator: Operator::LogicalOr,
                        left: Box::new(Expr::Boolean(true)),
                        right: Box::new(Expr::Assign {
                            var_name: "flag_two".into(),
                            new_value: Box::new(Expr::Boolean(true))
                        })
                    },
                    then: Box::new(Stmt::Block(vec![])),
                    els: Box::new(Stmt::Block(vec![]))
                },
                Stmt::Expr(Expr::Ident("flag_two".into()))
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

    #[test]
    fn environment_scoping() {
        let mut interpreter = Interpreter::new();
        assert_eq!(false, interpreter.block_needs_stack(&vec![
            Stmt::Print(Expr::Ident("aaa".into())),
            Stmt::Block(vec![
                Stmt::VarDecl("aaa".into(), None)  // ignored as not at the top level of statements
            ]),
            Stmt::Expr(Expr::Assign {
                var_name: "aaa".into(),
                new_value: Box::new(Expr::Int(2))
            })
        ]));

        assert_eq!(true, interpreter.block_needs_stack(&vec![
            Stmt::VarDecl("aaa".into(), None),  // declaration at top level of block statements
            Stmt::Block(vec![Stmt::Print(Expr::Ident("aaa".into()))]),
        ]));

        assert_eq!(1, interpreter.get_stack_size());

        // var toast = 5;
        // if (true) {
        //     var toast = 10;
        //     toast = toast + 1;  // should not modify `toast` declared on the first line
        // }
        // toast;
        assert_eq!(5, interpreter.interpret(&vec![
            Stmt::VarDecl("toast".into(), Some(Box::new(Stmt::Expr(Expr::Int(5))))),
            Stmt::If {
                cond: Expr::Boolean(true),
                then: Box::new(Stmt::Block(vec![
                    Stmt::VarDecl("toast".into(), Some(Box::new(Stmt::Expr(Expr::Int(10))))),
                    Stmt::Expr(Expr::Dyadic {
                        operator: Operator::Plus,
                        left: Box::new(Expr::Ident("toast".into())),
                        right: Box::new(Expr::Int(1))
                    })
                ])),
                els: Box::new(Stmt::Block(vec![]))
            },
            Stmt::Expr(Expr::Ident("toast".into()))
        ]).unwrap());

        // var abc = 5;
        // var result = if (true) {
        //     var abc = 10;
        //     abc;  // Should refer to the var `abc` declared in the line above
        // }
        // result;
        assert_eq!(10, interpreter.interpret(&vec![
            Stmt::VarDecl("abc".into(), Some(Box::new(Stmt::Expr(Expr::Int(5))))),
            Stmt::VarDecl("result".into(), Some(Box::new(Stmt::If {
                cond: Expr::Boolean(true),
                then: Box::new(Stmt::Block(vec![
                    Stmt::VarDecl("abc".into(), Some(Box::new(Stmt::Expr(Expr::Int(10))))),
                    Stmt::Expr(Expr::Ident("abc".into()))
                ])),
                els: Box::new(Stmt::Block(vec![]))
            }))),
            Stmt::Expr(Expr::Ident("result".into()))
        ]).unwrap());

        // var abcd = 5;
        // var output = if (false) {
        //     var abcd = 10;
        //     abcd; // Should refer to the var `abcd` declared in the line above
        // } else {
        //    var abcd = abcd;  // LHS: separate from `then` block, RHS: from the first line
        //    abcd = abcd + 1;
        //    abcd;  // should refer to `abcd` declared in the two lines above
        // }
        // output;
        assert_eq!(6, interpreter.interpret(&vec![
            Stmt::VarDecl("abcd".into(), Some(Box::new(Stmt::Expr(Expr::Int(5))))),
            Stmt::VarDecl("output".into(), Some(Box::new(Stmt::If {
                cond: Expr::Boolean(false),
                then: Box::new(Stmt::Block(vec![
                    Stmt::VarDecl("abcd".into(), Some(Box::new(Stmt::Expr(Expr::Int(10))))),
                    Stmt::Expr(Expr::Ident("abcd".into()))
                ])),
                els: Box::new(Stmt::Block(vec![
                    Stmt::VarDecl("abcd".into(), Some(Box::new(Stmt::Expr(Expr::Ident("abcd".into()))))),
                    Stmt::Expr(Expr::Assign {
                        var_name: "abcd".into(),
                        new_value: Box::new(Expr::Dyadic {
                            operator: Operator::Plus,
                            left: Box::new(Expr::Ident("abcd".into())),
                            right: Box::new(Expr::Int(1))
                        })
                    }),
                    Stmt::Expr(Expr::Ident("abcd".into()))
                ]))
            }))),
            Stmt::Expr(Expr::Ident("output".into()))
        ]).unwrap());

        // var abcde = 5;
        // var ignored = if (true) {
        //     var abcde = 10;
        //     abcde;  // Should refer to the var `abcde` declared in the line above
        // }
        // abcde;  // original abcde from the first line
        assert_eq!(5, interpreter.interpret(&vec![
            Stmt::VarDecl("abcde".into(), Some(Box::new(Stmt::Expr(Expr::Int(5))))),
            Stmt::VarDecl("ignored".into(), Some(Box::new(Stmt::If {
                cond: Expr::Boolean(true),
                then: Box::new(Stmt::Block(vec![
                    Stmt::VarDecl("abcde".into(), Some(Box::new(Stmt::Expr(Expr::Int(10))))),
                    Stmt::Expr(Expr::Ident("abcde".into()))
                ])),
                els: Box::new(Stmt::Block(vec![]))
            }))),
            Stmt::Expr(Expr::Ident("abcde".into()))
        ]).unwrap());

        assert_eq!(1, interpreter.get_stack_size());
    }

    #[test]
    fn functions() {
        let mut interpreter = Interpreter::new();

        // fn abs(n) {
        //     if (n < 0) { n = n * -1; }
        //     n;
        // }
        // abs(-5);
        assert_eq!(5, interpreter.interpret(&vec![
            Stmt::FnDecl {
                name: "abs".into(),
                parameters: vec!["n".to_string()],
                body: Box::new(Stmt::Block(vec![
                    Stmt::If {
                        cond: Expr::Dyadic {
                            operator: Operator::LessThan,
                            left: Box::new(Expr::Ident("n".into())),
                            right: Box::new(Expr::Int(0))
                        },
                        then: Box::new(Stmt::Expr(Expr::Assign {
                            var_name: "n".into(),
                            new_value: Box::new(Expr::Dyadic {
                                operator: Operator::Star,
                                left: Box::new(Expr::Ident("n".into())),
                                right: Box::new(Expr::Monadic {
                                    operator: Operator::Minus,
                                    operand: Box::new(Expr::Int(1))
                                })
                            })
                        })),
                        els: Box::new(Stmt::Block(vec![]))
                    },
                Stmt::Expr(Expr::Ident("n".into()))
                ]))
            },
            Stmt::Expr(Expr::Call {
                callee: "abs".into(),
                args: vec![Expr::Int(-5)]
            })
        ]).unwrap());

        // fn max(a, b) {
        //     if (a > b) {
        //        a;
        //     } else {
        //        b;
        //     }
        // }
        // max(128, 64);
        assert_eq!(128, interpreter.interpret(&vec![
            Stmt::FnDecl {
                name: "max".into(),
                parameters: vec!["a".to_string(), "b".to_string()],
                body: Box::new(Stmt::Block(vec![
                    Stmt::If {
                        cond: Expr::Dyadic {
                            operator: Operator::GreaterThan,
                            left: Box::new(Expr::Ident("a".into())),
                            right: Box::new(Expr::Ident("b".into()))
                        },
                        then: Box::new(Stmt::Expr(Expr::Ident("a".into()))),
                        els: Box::new(Stmt::Expr(Expr::Ident("b".into())))}
                ]))
            },
            Stmt::Expr(Expr::Call { callee: "max".to_string(), args: vec![Expr::Int(128), Expr::Int(64)] })
        ]).unwrap());
        
        assert_eq!(1, interpreter.get_stack_size());
    }
}