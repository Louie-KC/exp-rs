use std::{collections::HashMap, mem::discriminant};

use crate::ast::*;
use crate::errors::InterpretError;
use crate::errors::InterpretErrorKind as IEK;

type InterpretResult<T> = Result<T, InterpretError>;

macro_rules! interpret_err {
    ($kind:expr) => {
        Err(InterpretError { kind: $kind })
    }
}

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
    functions: HashMap<String, Function>,
    variables: HashMap<String, i32>
}

impl Environment {
    fn new() -> Self {
        Self { functions: HashMap::new(), variables: HashMap::new() }
    }

    fn declare_function(&mut self, name: &String, func: Function) -> InterpretResult<()> {
        if self.functions.contains_key(name) {
            interpret_err!(IEK::FnRedeclaration(name.clone()))
        } else {
            self.functions.insert(name.into(), func);
            Ok(())
        }
    }
    
    fn get_function(&self, name: &String) -> Option<&Function> {
        self.functions.get(name)
    }

    fn get_var(&self, name: &String) -> Option<&i32> {
        self.variables.get(name)
    }

    fn set_var(&mut self, name: &String, value: i32) -> () {
        self.variables.insert(name.into(), value);
    }

}

pub struct Interpreter {
    env_stack: Vec<Environment>
}

impl Interpreter {
    pub fn new() -> Self {
        Self { env_stack: vec![Environment::new()] }
    }
    
    pub fn interpret(&mut self, statements: &Vec<Stmt>) -> InterpretResult<i32> {
        let mut result = 0;  // Program exits with status code (default of 0)
        for statement in statements {
            result = self.evaluate_stmt(&statement)?
        }
        Ok(result)
    }

    pub fn interpret_one(&mut self, statement: &Stmt) -> InterpretResult<i32> {
        self.evaluate_stmt(&statement)
    }
    
    fn evaluate_stmt(&mut self, stmt: &Stmt) -> InterpretResult<i32> {
        // println!("Evaluating: {:?}", stmt);
        match stmt {
            Stmt::Print(expr) => {
                let result = self.evalulate_expr(expr)?;
                println!("{:?}", result);
                Ok(1)
            },
            Stmt::Expr(expr) => {
                self.evalulate_expr(expr)
            },
            Stmt::If { cond, then, els: None } => {
                match self.evalulate_expr(cond)? {
                    0 => Ok(0),
                    _ => self.interpret_one(then)
                }
            },
            Stmt::If {cond, then, els: Some(els)} => {
                match self.evalulate_expr(cond)? {
                    0 => self.interpret_one(els),
                    _ => self.interpret_one(then)
                }
            },
            Stmt::While { cond, body } => {
                let mut result = 0;
                while self.evalulate_expr(cond)? != 0 {
                    result = self.interpret_one(&body)?
                }
                Ok(result)
            },
            Stmt::Block(body) => {
                let needs_own_env = self.block_needs_stack(&body);
                if needs_own_env {
                    self.env_stack.push(Environment::new());
                }
                let result = self.interpret(body)?;
                if needs_own_env && self.env_stack.len() > 1 {
                    self.env_stack.pop();
                }
                Ok(result)
            },
            Stmt::VarDecl(ident, value) => {
                if self.current_env_has_var(ident.into())? {
                    return interpret_err!(IEK::VarRedeclaration(ident.into()))
                }
                let val = match value {
                    Some(v) => self.evaluate_stmt(v)?,
                    None => 0
                };
                self.add_var(ident.into(), val)?;
                Ok(val)  // use the value we just assigned
            },
            Stmt::FnDecl { name, parameters, body } => {
                let func = Function::new(parameters.to_owned(), *body.to_owned());
                self.declare_function(name, func)?;
                Ok(0)
            },
        }
    }
                 
    fn evalulate_expr(&mut self, expr: &Expr) -> InterpretResult<i32> {
        // println!("evaluate_expr: {:?}", expr);
        match expr {
            Expr::Int(n)   => Ok(*n),
            Expr::Boolean(true)  => Ok(1),
            Expr::Boolean(false) => Ok(0),
            Expr::Ident(s) => self.get_var(s).cloned(),
            Expr::Monadic { operator, operand } => {
                let operand_value = self.evalulate_expr(operand)?;
                match operator {
                    Operator::Plus  => Ok(operand_value),
                    Operator::Minus => Ok(-operand_value),
                    _ => interpret_err!(IEK::InvalidOperation(operator.clone(), operand.clone()))
                }
            },
            Expr::Dyadic { operator, left, right } => {
                let lhs = self.evalulate_expr(left)?;
                match (lhs, operator) {  // (short circuit) evaluation of logical operators
                    (0, Operator::LogicalAnd) => return Ok(0),
                    (0, Operator::LogicalOr)  => return self.evalulate_expr(right),
                    (_, Operator::LogicalOr)  => return Ok(1),
                    (_, Operator::LogicalAnd) => return self.evalulate_expr(right),
                    _ => {}  // Move on to the remaining dyadic operators
                }

                let rhs = self.evalulate_expr(right)?;
                match operator {
                    Operator::Plus          => Ok(lhs + rhs),
                    Operator::Minus         => Ok(lhs - rhs),
                    Operator::Star          => Ok(lhs * rhs),
                    Operator::Slash         => Ok(lhs / rhs),
                    Operator::Modulo        => Ok(lhs % rhs),
                    Operator::EqualTo       => if lhs == rhs {Ok(1)} else {Ok(0)},
                    Operator::NotEqualTo    => if lhs != rhs {Ok(1)} else {Ok(0)},
                    Operator::LessThan      => if lhs <  rhs {Ok(1)} else {Ok(0)},
                    Operator::LessEquals    => if lhs <= rhs {Ok(1)} else {Ok(0)},
                    Operator::GreaterThan   => if lhs >  rhs {Ok(1)} else {Ok(0)},
                    Operator::GreaterEquals => if lhs >= rhs {Ok(1)} else {Ok(0)},
                    Operator::LogicalAnd
                    | Operator::LogicalOr   => interpret_err!(IEK::PlaceHolderError),
                    // | Operator::LogicalOr   => Err("Logical operators missed by evaluation".into()),
                }
            },
            Expr::Assign { var_name, new_value } => {
                self.get_var(var_name)?;
                let val = self.evalulate_expr(new_value)?;
                self.update_var(var_name, val)?;
                Ok(val)
            },
            Expr::Call { callee, args } => {
                let function = self.get_function(callee)?;

                let params = &function.params;
                if params.len() != args.len() {
                    return interpret_err!(IEK::FnIncorrectNumArgs(callee.into(), params.len(), args.len()))
                }

                // Prepend function body/block with variable declarations of parameters
                // with argument values.
                let mut block: Vec<Stmt> = params.iter()
                    .zip(args.iter())
                    .map(|(param_name, arg)| {
                        Stmt::VarDecl(param_name.into(), Some(Box::new(Stmt::Expr(arg.clone()))))
                    })
                    .collect::<Vec<Stmt>>();

                block.push(function.body.clone());

                self.evaluate_stmt(&Stmt::Block(block))
            },
        }
    }

    /// Determines whether a block statement needs its own variable environment added onto the
    /// environment stack by checking if the statements of the block make any declarations.
    fn block_needs_stack(&self, stack_body: &Vec<Stmt>) -> bool {
        let var_decl_disc = discriminant(&Stmt::VarDecl("".into(), None));
        let fn_decl_disc = discriminant(&Stmt::FnDecl {
            name: "".into(), parameters: vec![], body: Box::new(Stmt::Block(vec![]))
        });
        stack_body.iter()
                  .map(|stmt| discriminant(stmt))
                  .any(|disc| disc.eq(&var_decl_disc) || disc.eq(&fn_decl_disc))
    }

    /// Searches for the variable specified by the 'name' parameter and returns
    /// its value.
    /// The variable is searched for beginning from the current/inner-most scope.
    fn get_var(&self, name: &String) -> InterpretResult<&i32> {
        self.env_stack.iter()
                      .rev()
                      .find_map(|env| env.get_var(name))
                      .ok_or(InterpretError { kind: IEK::VarNotInScope(name.into()) })
    }

    /// Determines whether a variable with the parameter 'name' already exists
    /// in the current/inner-most scope.
    fn current_env_has_var(&self, name: &String) -> InterpretResult<bool> {
        match self.env_stack.last() {
            Some(env) => Ok(env.get_var(name).is_some()),
            None => return interpret_err!(IEK::NoEnvironments)
        }
    }

    /// Updates the value of an existing variable with the 'name' parameter. The
    /// search for the specified variable works outwards from the inner most scope.
    fn update_var(&mut self, name: &String, value: i32) -> InterpretResult<()> {
        for env in self.env_stack.iter_mut().rev() {
            if env.get_var(name).is_some() {
                env.set_var(name, value);
                return Ok(());
            }
        }
        return interpret_err!(IEK::VarNotInScope(name.clone()))
    }

    /// Adds/declares a variable with the value to the current or inner most scope.
    fn add_var(&mut self, name: &String, value: i32) -> InterpretResult<()> {
        match self.env_stack.last_mut() {
            Some(env) => {
                env.set_var(name, value);
                Ok(())
            },
            None => interpret_err!(IEK::NoEnvironments)
        }
    }

    /// Retrieve a function from the environment stack by `name`. The search
    /// starts from the inner-most scope, and works outwards.
    /// An InterpretError is returned if the function cannot be found.
    fn get_function(&self, name: &String) -> InterpretResult<&Function> {
        self.env_stack
            .iter()
            .rev()
            .find_map(|env| env.get_function(name))
            .ok_or(InterpretError { kind: IEK::FnNotInScope(name.into()) })
    }

    fn declare_function(&mut self, name: &String, func: Function) -> InterpretResult<()> {
        if let Some(env) = self.env_stack.last_mut() {
            env.declare_function(name, func)
        } else {
            interpret_err!(IEK::NoEnvironments)
        }
    }

    #[cfg(test)]
    fn get_stack_size(&self) -> usize {
        self.env_stack.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::*;
    use crate::errors::InterpretError;
    use crate::errors::InterpretErrorKind as IEK;
    use crate::interpreter::Interpreter;

    #[test]
    fn basic_calculation() {
        let mut interpreter = Interpreter::new();

        assert_eq!(Ok(0), interpreter.interpret_one(&Stmt::Expr(Expr::Int(0))));  // 0
        assert_eq!(Ok(1), interpreter.interpret_one(&Stmt::Expr(  // +1
            Expr::Monadic {
                operator: Operator::Plus,
                operand: Box::new(Expr::Int(1))
            })));
        assert_eq!(Ok(-1), interpreter.interpret_one(&Stmt::Expr(  // -1
            Expr::Monadic {
                operator: Operator::Minus,
                operand: Box::new(Expr::Int(1))
            })));
        assert_eq!(Ok(256), interpreter.interpret_one(&Stmt::Expr(  // 192 + 64
            Expr::Dyadic {
                operator: Operator::Plus,
                left: Box::new(Expr::Int(192)),
                right: Box::new(Expr::Int(64))
            })));
        assert_eq!(Ok(-8), interpreter.interpret_one(&Stmt::Expr(  // 16 / 4 * -2
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
            })));
    }

    #[test]
    fn variables() {
        let mut interpreter = Interpreter::new();
        
        // my_var;
        assert_eq!(interpret_err!(IEK::VarNotInScope("my_var".into())),
                   interpreter.interpret_one(&Stmt::Expr(Expr::Ident("my_var".into()))));

        // undefined_var = 1;  // not declared
        assert_eq!(interpret_err!(IEK::VarNotInScope("undefined_var".into())),
            interpreter.interpret_one(&Stmt::Expr(Expr::Assign {
                var_name: "undefined_var".into(),
                new_value: Box::new(Expr::Int(1))
        })));

        // var defined_twice;
        // var defined_twice;
        assert_eq!(interpret_err!(IEK::VarRedeclaration("defined_twice".into())),
            interpreter.interpret(&vec![
                Stmt::VarDecl("defined_twice".into(), None),
                Stmt::VarDecl("defined_twice".into(), None)
            ]));

        // var not_initialised;
        // not_initialised;
        assert_eq!(Ok(0), interpreter.interpret(&vec![
            Stmt::VarDecl("not_initialised".into(), None),
            Stmt::Expr(Expr::Ident("not_initialised".into()))
        ]));

        // var tomato = 127;
        // tomato;
        assert_eq!(Ok(127), interpreter.interpret(&vec![
            Stmt::VarDecl("tomato".into(), Some(Box::new(Stmt::Expr(Expr::Int(127))))),
            Stmt::Expr(Expr::Ident("tomato".into()))
        ]));

        // var c = if (false) { 1; } else { 1023; };
        // c;
        assert_eq!(Ok(1023), interpreter.interpret(&vec![
            Stmt::VarDecl("c".into(), Some(Box::new(Stmt::If {
                cond: Expr::Boolean(false),
                then: Box::new(Stmt::Expr(Expr::Int(1))),
                els: Some(Box::new(Stmt::Expr(Expr::Int(1023))))
            }))),
            Stmt::Expr(Expr::Ident("c".into()))
        ]));

        // var a = 10;
        // var b = 20;
        // a = a + b;
        // a;
        assert_eq!(Ok(30), interpreter.interpret(&vec![
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
        ]));
    }

    #[test]
    fn branch() {
        let mut interpreter = Interpreter::new();

        // if (256 == 256) { 1; } else { 2; }
        assert_eq!(Ok(1), interpreter.interpret(
            &vec![
                Stmt::If {
                    cond: Expr::Dyadic {
                        operator: Operator::EqualTo,
                        left: Box::new(Expr::Int(256)),
                        right: Box::new(Expr::Int(256))
                    },
                    then: Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(1))])),
                    els: Some(Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(2))])))
                }]
        ));

        // if (512 != 512) { 1; } else { 2; }
        assert_eq!(Ok(2), interpreter.interpret(
            &vec![
                Stmt::If {
                    cond: Expr::Dyadic {
                        operator: Operator::NotEqualTo,
                        left: Box::new(Expr::Int(512)),
                        right: Box::new(Expr::Int(512))
                    },
                    then: Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(1))])),
                    els: Some(Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(2))])))
                }]
        ));

        // if (512 == 2048) { 11; }  // Default output of 0 without any statements
        assert_eq!(Ok(0), interpreter.interpret(
            &vec![
                Stmt::If {
                    cond: Expr::Dyadic {
                        operator: Operator::EqualTo,
                        left: Box::new(Expr::Int(512)),
                        right: Box::new(Expr::Int(2048))
                    },
                    then: Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(11))])),
                    els: None
                }]
        ));
        
        // var a = 5;
        // var b = 50;
        // var max = if (a > b) { a; } else { b; };
        // max;
        assert_eq!(Ok(50), interpreter.interpret(
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
                    els: Some(Box::new(Stmt::Expr(Expr::Ident("b".into()))))
                }))),
                Stmt::Expr(Expr::Ident("max".into()))
                ]
        ));

        // if (true && true) { 1; } else { 2; }
        assert_eq!(Ok(1), interpreter.interpret(
            &vec![
                Stmt::If {
                    cond: Expr::Dyadic {
                        operator: Operator::LogicalAnd,
                        left: Box::new(Expr::Boolean(true)),
                        right: Box::new(Expr::Boolean(true))
                    },
                    then: Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(1))])),
                    els: Some(Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(2))])))
                }]
        ));

        // if (true && false) { 1; } else { 2; }
        assert_eq!(Ok(2), interpreter.interpret(
            &vec![
                Stmt::If {
                    cond: Expr::Dyadic {
                        operator: Operator::LogicalAnd,
                        left: Box::new(Expr::Boolean(true)),
                        right: Box::new(Expr::Boolean(false))
                    },
                    then: Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(1))])),
                    els: Some(Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(2))])))
                }]
        ));

    }

    #[test]
    fn short_circuit_eval() {
        let mut interpreter = Interpreter::new();

        // // Short circuit evaluation
        // var flag_one = false;  // true = 1, false = 0
        // if (false && flag_one = true) {}
        // flag_one;  // should remain as false/0
        assert_eq!(Ok(0), interpreter.interpret(
            &vec![
                Stmt::VarDecl("flag_one".into(), Some(Box::new(Stmt::Expr(Expr::Boolean(false))))),
                Stmt::If {
                    cond: Expr::Dyadic {
                        operator: Operator::LogicalAnd,
                        left: Box::new(Expr::Boolean(false)),
                        right: Box::new(Expr::Assign {
                            var_name: "flag_one".into(),
                            new_value: Box::new(Expr::Boolean(true))
                        })
                    },
                    then: Box::new(Stmt::Block(vec![])),
                    els: None
                },
                Stmt::Expr(Expr::Ident("flag_one".into()))
            ]
        ));

        // // Short circuit evaluation
        // var flag_two = false;  // true = 1, false = 0
        // if (true || flag_two = true) {}
        // flag_two;  // should remain as false/0 as right side of || does not evaluate
        assert_eq!(Ok(0), interpreter.interpret(
            &vec![
                Stmt::VarDecl("flag_two".into(), Some(Box::new(Stmt::Expr(Expr::Boolean(false))))),
                Stmt::If {
                    cond: Expr::Dyadic {
                        operator: Operator::LogicalOr,
                        left: Box::new(Expr::Boolean(true)),
                        right: Box::new(Expr::Assign {
                            var_name: "flag_two".into(),
                            new_value: Box::new(Expr::Boolean(true))
                        })
                    },
                    then: Box::new(Stmt::Block(vec![])),
                    els: None
                },
                Stmt::Expr(Expr::Ident("flag_two".into()))
            ]
        ));
    }

    #[test]
    fn loops() {
        let mut interpreter = Interpreter::new();

        // var i = 0;
        // while (i < 5) {
        //     i = i + 1;
        // }
        // i;
        assert_eq!(Ok(5), interpreter.interpret(
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
        ));
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
        assert_eq!(Ok(5), interpreter.interpret(&vec![
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
                els: None
            },
            Stmt::Expr(Expr::Ident("toast".into()))
        ]));

        // var abc = 5;
        // var result = if (true) {
        //     var abc = 10;
        //     abc;  // Should refer to the var `abc` declared in the line above
        // }
        // result;
        assert_eq!(Ok(10), interpreter.interpret(&vec![
            Stmt::VarDecl("abc".into(), Some(Box::new(Stmt::Expr(Expr::Int(5))))),
            Stmt::VarDecl("result".into(), Some(Box::new(Stmt::If {
                cond: Expr::Boolean(true),
                then: Box::new(Stmt::Block(vec![
                    Stmt::VarDecl("abc".into(), Some(Box::new(Stmt::Expr(Expr::Int(10))))),
                    Stmt::Expr(Expr::Ident("abc".into()))
                ])),
                els: None
            }))),
            Stmt::Expr(Expr::Ident("result".into()))
        ]));

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
        assert_eq!(Ok(6), interpreter.interpret(&vec![
            Stmt::VarDecl("abcd".into(), Some(Box::new(Stmt::Expr(Expr::Int(5))))),
            Stmt::VarDecl("output".into(), Some(Box::new(Stmt::If {
                cond: Expr::Boolean(false),
                then: Box::new(Stmt::Block(vec![
                    Stmt::VarDecl("abcd".into(), Some(Box::new(Stmt::Expr(Expr::Int(10))))),
                    Stmt::Expr(Expr::Ident("abcd".into()))
                ])),
                els: Some(Box::new(Stmt::Block(vec![
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
                ])))
            }))),
            Stmt::Expr(Expr::Ident("output".into()))
        ]));

        // var abcde = 5;
        // var ignored = if (true) {
        //     var abcde = 10;
        //     abcde;  // Should refer to the var `abcde` declared in the line above
        // }
        // abcde;  // original abcde from the first line
        assert_eq!(Ok(5), interpreter.interpret(&vec![
            Stmt::VarDecl("abcde".into(), Some(Box::new(Stmt::Expr(Expr::Int(5))))),
            Stmt::VarDecl("ignored".into(), Some(Box::new(Stmt::If {
                cond: Expr::Boolean(true),
                then: Box::new(Stmt::Block(vec![
                    Stmt::VarDecl("abcde".into(), Some(Box::new(Stmt::Expr(Expr::Int(10))))),
                    Stmt::Expr(Expr::Ident("abcde".into()))
                ])),
                els: None
            }))),
            Stmt::Expr(Expr::Ident("abcde".into()))
        ]));

        assert_eq!(1, interpreter.get_stack_size());
    }

    #[test]
    fn functions() {
        let mut interpreter = Interpreter::new();

        // mystery();
        assert_eq!(interpret_err!(IEK::FnNotInScope("mystery".into())),
            interpreter.interpret(&vec![
                Stmt::Expr(Expr::Call { callee: "mystery".into(), args: vec![] })
            ])
        );

        /*
            fn duplicate() {}
            fn duplicate() {}
         */
        assert_eq!(interpret_err!(IEK::FnRedeclaration("duplicate".into())),
            interpreter.interpret(&vec![
                Stmt::FnDecl {
                    name: "duplicate".into(),
                    parameters: vec![],
                    body: Box::new(Stmt::Block(vec![]))
                },
                Stmt::FnDecl {
                    name: "duplicate".into(),
                    parameters: vec![],
                    body: Box::new(Stmt::Block(vec![]))
                }
            ])
        );

        /*
            fn no_params() { 256; }
            no_params(0);
         */
        assert_eq!(interpret_err!(IEK::FnIncorrectNumArgs("no_params".into(), 0, 1)),
            interpreter.interpret(&vec![
                Stmt::FnDecl {
                    name: "no_params".into(),
                    parameters: vec![],
                    body: Box::new(Stmt::Block(vec![]))
                },
                Stmt::Expr(Expr::Call {
                    callee: "no_params".into(),
                    args: vec![Expr::Int(0)]
                })
            ]
        ));

        /*
            fn three_params(a, b, c) {}
            three_params(32);
         */
        assert_eq!(interpret_err!(IEK::FnIncorrectNumArgs(
            "three_params".into(), 3, 1)),
            interpreter.interpret(&vec![
                Stmt::FnDecl {
                    name: "three_params".into(),
                    parameters: vec!["a".into(), "b".into(), "c".into()],
                    body: Box::new(Stmt::Block(vec![]))
                },
                Stmt::Expr(Expr::Call {
                    callee: "three_params".into(),
                    args: vec![Expr::Int(32)]
                })
            ])
        );

        // fn abs(n) {
        //     if (n < 0) { n = n * -1; }
        //     n;
        // }
        // abs(-5);
        assert_eq!(Ok(5), interpreter.interpret(&vec![
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
                        els: None
                    },
                Stmt::Expr(Expr::Ident("n".into()))
                ]))
            },
            Stmt::Expr(Expr::Call {
                callee: "abs".into(),
                args: vec![Expr::Int(-5)]
            })
        ]));

        // fn max(a, b) {
        //     if (a > b) {
        //        a;
        //     } else {
        //        b;
        //     }
        // }
        // max(128, 64);
        assert_eq!(Ok(128), interpreter.interpret(&vec![
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
                        els: Some(Box::new(Stmt::Expr(Expr::Ident("b".into()))))}
                ]))
            },
            Stmt::Expr(Expr::Call { callee: "max".to_string(), args: vec![Expr::Int(128), Expr::Int(64)] })
        ]));

        /*
        fn ree() { 0; }
        
        {
            fn ree() { 1; }
            ree();
        }
         */
        assert_eq!(Ok(1), interpreter.interpret(&vec![
            Stmt::FnDecl {
                name: "ree".into(),
                parameters: vec![],
                body: Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(0))]))
            },
            Stmt::Block(vec![
                Stmt::FnDecl {
                    name: "ree".into(),
                    parameters: vec![],
                    body: Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(1))]))
                },
                Stmt::Expr(Expr::Call { callee: "ree".into(), args: vec![] })
            ])
        ]));
        
        assert_eq!(1, interpreter.get_stack_size());
    }
}