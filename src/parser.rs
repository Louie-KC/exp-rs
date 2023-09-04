use std::mem::{discriminant};

use crate::tokens::{Token, self};
use crate::ast::*;

/*
 * Recursive Descent
 * 
 * rules:
 *    program -> statement*
 *  statement -> print
 *               | var
 *               | if
 *               | while
 *               | for
 *               | block
 *               | expression ";"
 *      print -> "print(" expression ");"
 *    fnDeclr -> "fn" function
 *   function -> Identifier "(" ( Identifier ( "," Identifier )* )? ")" block
 *    varDecl -> "var" Identifier ( "=" statement )? ";"
 *         if -> "if(" expression ")" block ( else block )?
 *      while -> "while" "(" expression ")" block
 *        for -> "for" "(" ( varDeclr | expression )? ";" expression? ";" expression ")" block
 *      block -> "{" statement* "}"
 * 
 * expression -> assignment
 * assignment -> logic_or | Identifier "=" expression
 *   logic_or -> logic_and ( "||" logic_and )*
 *  logic_and -> comparator ( "&&" comparator )*
 * comparator -> term ( ("==" | "!=" | "<" | "<=" | ">" | ">=" ) term )*
 *       term -> factor ( ( "+" | "-" ) factor)*
 *     factor -> unary ( ( "*" | "/" | "%" ) unary)*
 *      unary -> fn_call | "+" fn_call | "-" fn_call
 *    fn_call -> atom | atom "(" expression ( "," expression )* ")"
 *       atom -> "(" expression ")" | Integer | Identifier | Boolean
 */

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens: tokens, pos: 0 }
    }

    pub fn parse(&mut self) -> Result<Vec<Stmt>, String> {
        if !self.brackets_balanced() {
            return Err("Inbalanced brackets".into())
        }
        if let Err(e) = self.illegal_token_seq_check() {
            return Err(e);
        }

        // println!("Parsing: {:?}", self.tokens);
        let mut statements: Vec<Stmt> = Vec::new();

        while *self.peek() != Token::EOF {
            statements.push(self.statement()?);
        }

        Ok(statements)
    }

    fn statement(&mut self) -> Result<Stmt, String> {
        // println!("statement: {:?}", self.peek());
        match *self.peek() {
            Token::Print => self.print_statement(),
            Token::Function => self.function_declaration(),
            Token::Var   => self.var_statement(),
            Token::If    => self.if_statement(),
            Token::While => self.while_statement(),
            Token::For   => self.for_statement(),
            Token::LSquirly => self.block_statement(),
            _            => self.expression_statement()
        }
    }

    // Rule: print -> "print(" expression ");"
    fn print_statement(&mut self) -> Result<Stmt, String> {
        self.advance();  // Print
        let val: Expr = self.expression()?;
        self.consume(Token::EndLine, "Expected \";\" to end print statement")?;
        Ok(Stmt::Print(val))
    }

    // Rule: fnDeclr -> "fn" function
    fn function_declaration(&mut self) -> Result<Stmt, String> {
        self.advance();  // fn
        self.function()
    }

    // Rule: function -> Identifier "(" ( Identifier ( "," Identifier )* )? ")" block
    fn function(&mut self) -> Result<Stmt, String> {
        let name = match self.atom() {
            Ok(Expr::Ident(n)) => n,
            _ => return Err("Expected name identifier in function declaration".into())
        };
        self.consume(Token::LParen, "Missing \"(\" on function declaration")?;

        let mut parameters: Vec<String> = vec![];
        while !self.matches_type(Token::RParen) {
            parameters.push(match self.atom() {
                Ok(Expr::Ident(n)) => n,
                _ => return Err("Expected named parameter for function declaration".into())
            });
            if self.matches_type(Token::Comma) {
                self.advance();
            }
        }
        self.advance();  // RParen
        if parameters.len() > 255 {
            return Err(format!("Over 255 parameters for function {}", name))
        }
        if !self.matches_type(Token::LSquirly) {
            return Err(format!("Expect function body in function declaration. Found: {:?}", self.peek()))
        }
        let body: Stmt = self.block_statement()?;
        Ok(Stmt::FnDecl { name: name, parameters: parameters, body: Box::new(body) })
    }

    // Rule: varDecl -> "var" Identifier ( "=" statement )? ";"
    fn var_statement(&mut self) -> Result<Stmt, String> {
        self.advance();  // var
        if !self.matches_type(Token::Ident("".into())) {
            return Err("Expected identifier after \"var\"".into());
        }
        let ident: String = self.get_var_name();
        self.advance();
        let value = match self.peek() {
            Token::Equal => {
                self.advance();  // =
                let deterministic: bool = self.matches_type(Token::If);
                let v: Stmt = self.statement()?;
                match (deterministic, self.matches_type(Token::EndLine)) {
                    (true, true)  => self.advance(),
                    (true, false) => return Err("Expected ; after var declaration".into()),
                    _        => {}
                }
                Some(Box::new(v))
            },
            Token::EndLine => {
                self.advance();  // ;
                None
            },
            _ => return Err("Expected \";\" or value to end variable declaration".into())
        };
        Ok(Stmt::VarDecl(ident.into(), value))
    }

    fn get_var_name(&self) -> String {
        // Assumes self.pos points at an Ident Token
        match self.peek() {
            Token::Ident(s) => s.to_string(),
            _ => panic!("bad get_var_name() call")
        }
    }

    // Rule: if -> "if(" expression ")" block ( else block )?
    fn if_statement(&mut self) -> Result<Stmt, String> {
        self.advance();  // if
        self.consume(Token::LParen, "Expected \"(\" after if")?;

        let cond: Expr = self.expression()?;
        
        self.consume(Token::RParen, "Expected \")\" after if condition")?;

        let then: Stmt = self.statement()?;
        let els: Stmt = if self.matches_type(Token::Else) {
            self.advance();  // else
            self.statement()?
        } else {
            Stmt::Block(vec![])
        };
        Ok(Stmt::If{ cond: cond, then: Box::new(then), els: Box::new(els)})
    }

    // Rule: while -> "while(" expression ")" block
    fn while_statement(&mut self) -> Result<Stmt, String> {
        self.advance();  // while
        self.advance();  // LParen

        let cond: Expr = self.expression()?;
        self.consume(Token::RParen, "Expected \")\" after while condition")?;

        let body = self.block_statement()?;
        Ok(Stmt::While { cond: cond, body: Box::new(body) })
    }

    // Rule: for -> "for" "(" ( varDeclr | expression )? ";" expression? ";" expression ")" block
    fn for_statement(&mut self) -> Result<Stmt, String> {
        self.advance(); // for
        self.advance(); // LParen

        let init = match self.peek() {
            Token::EndLine => {
                self.advance();
                None
            },
            _ => Some(self.statement()?)
        };

        let cond = match self.peek() {
            Token::EndLine => Expr::Boolean(true),
            _ => self.expression()?
        };
        self.advance();  // ;

        let step = match self.peek() {
            Token::RParen => None,
            _ => Some(self.expression()?)
        };

        // println!("{:?}, {:?}, {:?}", init, cond, step);
        self.consume(Token::RParen, "Expected \"(\" to close loop condition")?;

        // assembling
        let mut loop_body = match self.block_statement() {
            Ok(Stmt::Block(stmts)) => stmts,
            _ => return Err("For loop expected a body of block statement".into())
        };
        if let Some(step_expr) = step {
            loop_body.push(Stmt::Expr(step_expr));
        }
        let main_loop = Stmt::While {
            cond: cond,
            body: Box::new(Stmt::Block(loop_body))
        };
        match init {
            Some(init_stmt) => Ok(Stmt::Block(vec![init_stmt, main_loop])),
            None => Ok(main_loop)
        }
    }

    // Rule: block -> "{" statement* "}"
    fn block_statement(&mut self) -> Result<Stmt, String> {
        self.advance();  // {
        let mut peek = self.peek();
        let mut statements:Vec<Stmt> = Vec::new();
        while *peek != Token::EOF && *peek != Token::RSquirly {
            statements.push(self.statement()?);
            peek = self.peek();
        }
        self.consume(Token::RSquirly, "Expected \"}\" to close block")?;

        Ok(Stmt::Block(statements))
    }

    // Rule: expression -> comparator | term
    fn expression_statement(&mut self) -> Result<Stmt, String> {
        let expr: Expr = self.expression()?;
        self.consume(Token::EndLine, "Expected \";\" to end expression stmt")?;

        Ok(Stmt::Expr(expr))
    }
    
    fn eval_operator(&mut self) -> Result<Operator, String> {
        let op = match self.peek() {
            Token::Plus    => Operator::Plus,
            Token::Minus   => Operator::Minus,
            Token::Star    => Operator::Star,
            Token::Slash   => Operator::Slash,
            Token::Percent => Operator::Modulo,
            Token::EqualTo => Operator::EqualTo,
            Token::Negate  => {
                match self.peek_ahead() {
                    Some(Token::Equal) => {
                        self.advance();
                        Operator::NotEqualTo
                    },
                    _ => todo!("negation of variables")
                }
            },
            Token::LessThan      => Operator::LessThan,
            Token::LessEquals    => Operator::LessEquals,
            Token::GreaterThan   => Operator::GreaterThan,
            Token::GreaterEquals => Operator::GreaterEquals,
            Token::Or      => Operator::LogicalOr,
            Token::And     => Operator::LogicalAnd,
            _ => return Err("Invalid operator".into())
        };
        self.advance();
        Ok(op)
    }

    fn peek(&self) -> &Token {
        &self.tokens.get(self.pos).unwrap()
    }

    fn peek_ahead(&self) -> Option<&Token> {
        self.tokens.get(self.pos + 1)
    }

    fn matches_type(&self, check: Token) -> bool {
        discriminant(&check) == discriminant(self.peek())
    }

    fn matches_types(&self, types: Vec<Token>) -> bool {
        if self.pos == self.tokens.len() {
            return false
        }
        for t in types.iter() {
            // Compare enum variant (ignoring any held data)
            if discriminant(t) == discriminant(self.peek()) {
                return true
            }
        }
        false
    }

    fn illegal_token_seq_check(&self) -> Result<(), String> {
        use tokens::Token::*;
        for i in 1..self.tokens.len() {
            let prev = &self.tokens[i-1];
            let cur = &self.tokens[i];

            // Int and Ident token checking. Cannot seem to get working with pattern matching :(
            let int_discrim = discriminant(&Token::Int(0));
            let ident_discrim = discriminant(&Token::Ident("".into()));
            let prev_discrim = discriminant(prev);
            let cur_discrim = discriminant(cur);
            if prev_discrim == int_discrim && cur_discrim == int_discrim {
                return Err("Two Int tokens in a row".into());
            }
            if prev_discrim == ident_discrim && cur_discrim == ident_discrim {
                return Err("Two Ident tokens in a row".into());
            }
            if prev_discrim == int_discrim && cur_discrim == ident_discrim {
                return Err("Int token immediately followed by Ident token".into())
            }
            if prev_discrim == ident_discrim && cur_discrim == int_discrim {
                return Err("Ident token immediately followed by Int token".into())
            }

            // Remaining illegal token combinations
            match (&self.tokens[i-1], &self.tokens[i]) {
                (Function, Function) => return Err("Two Function tokens in a row".into()),
                (Star, Star)   => return Err("Two Star tokens in a row".into()),
                (Slash, Slash) => return Err("Two Slash tokens in a row".into()),
                (Let, Let)     => return Err("Two Let tokens in a row".into()),
                (_, _) => {}
            }
        }
        Ok(())
    }

    fn brackets_balanced(&self) -> bool {
        let mut stack: Vec<Token> = Vec::new();

        for token in self.tokens.iter() {
            match token {
                Token::LParen   => stack.push(Token::LParen),
                Token::LSquirly => stack.push(Token::LSquirly),
                Token::RParen   if stack.pop() != Some(Token::LParen)   => return false,
                Token::RSquirly if stack.pop() != Some(Token::LSquirly) => return false,
                _ => {}
            }
        }

        stack.is_empty()
    }

    fn advance(&mut self) -> () {
        self.pos += 1
    }

    /// Check the current token against the `expected` token, advance parser pos if matching.
    /// If Tokens don't match, return a Result Err with the `err` message.
    fn consume(&mut self, expected: Token, err: &str) -> Result<(), String> {
        if !self.matches_type(expected) {
            return Err(format!("{}. Found {:?}", err, self.peek()))
        }
        self.advance();
        Ok(())
    }

    // Rule: expression -> assignment
    fn expression(&mut self) -> Result<Expr, String> {
        // println!("expression called");
        Ok(self.assignment()?)
    }

    // Rule: assignment -> logic_or | Identifier "=" expression
    fn assignment(&mut self) -> Result<Expr, String> {
        let mut expr: Expr = self.logic_or()?;

        expr = match (&expr, self.peek()) {
            (Expr::Ident(v_name), Token::Equal) => {
                self.advance();
                let value: Expr = self.assignment()?;
                Expr::Assign { var_name: v_name.into(), new_value: Box::new(value) }
            },
            _ => expr
        };

        Ok(expr)
    }

    // Rule: logic_or -> logic_and ( "||" logic_and )*
    fn logic_or(&mut self) -> Result<Expr, String> {
        let mut expr: Expr = self.logic_and()?;

        if self.matches_type(Token::Or) {
            let op: Operator = Operator::LogicalOr;
            self.advance();  // ||
            let right: Expr = self.logic_and()?;
            expr = Expr::Logical { operator: op, left: Box::new(expr), right: Box::new(right) };
        }
    
        Ok(expr)
    }

    // Rule: logic_and -> term ( "&&" term )*
    fn logic_and(&mut self) -> Result<Expr, String> {
        let mut expr: Expr = self.comparator()?;
        if self.matches_type(Token::And) {
            let op: Operator = Operator::LogicalAnd;
            self.advance();
            let right: Expr = self.comparator()?;
            expr = Expr::Logical { operator: op, left: Box::new(expr), right: Box::new(right) }
        }
        Ok(expr)
    }

    // Rule: comparator -> term ( ("==" | "!=" | "<" | "<=" | ">" | ">=" ) term )*
    fn comparator(&mut self) -> Result<Expr, String> {
        let mut expr: Expr = self.term()?;
        let comparators = vec![Token::EqualTo, Token::Negate, Token::LessThan,
                                           Token::LessEquals, Token::GreaterThan, Token::GreaterEquals];
        
        if self.matches_types(comparators) {
            if let Ok(op) = self.eval_operator() {
                let right: Expr = self.term()?;
                expr = Expr::Dyadic {
                    operator: op,
                    left: Box::new(expr),
                    right: Box::new(right)
                }
            }
        }
        Ok(expr)
    }
    
    // Rule: term -> factor ( ( "+" | "-" ) factor)*
    fn term(&mut self) -> Result<Expr, String> {
        let mut expr: Expr = self.factor()?;

        while self.matches_types(vec![Token::Plus, Token::Minus]) {
            let op: Operator = self.eval_operator()?;
            let right: Expr = self.factor()?;
            expr = Expr::Dyadic { operator: op, left: Box::new(expr), right: Box::new(right) }
        }
        Ok(expr)
    }

    // Rule: factor -> unary ( ( "*" | "/" | "%" ) unary)*
    fn factor(&mut self) -> Result<Expr, String> {
        let mut expr: Expr = self.unary()?;
        // allowed tokens
        while self.matches_types(vec![Token::Star, Token::Slash, Token::Percent]) {
            let op: Operator = self.eval_operator()?;
            let right: Expr = self.unary()?;
            expr = Expr::Dyadic { operator: op, left: Box::new(expr), right: Box::new(right) }
        }
        Ok(expr)
    }

    // Rule: unary -> fn_call | "+" fn_call | "-" fn_call
    fn unary(&mut self) -> Result<Expr, String> {
        if self.matches_type(Token::Plus) {
            self.advance();
            let operand: Expr = self.fn_call()?;
            return Ok(Expr::Monadic { operator: Operator::Plus, operand: Box::new(operand) })
        }
        if self.matches_type(Token::Minus) {
            self.advance();
            let operand: Expr = self.fn_call()?;
            return Ok(Expr::Monadic { operator: Operator::Minus, operand: Box::new(operand) })
        }
        self.fn_call()
    }

    // fn_call -> atom | atom "(" expression ( "," expression )* ")"
    fn fn_call(&mut self) -> Result<Expr, String> {
        let expr: Expr = self.atom()?;
        // println!("fn_call pre-match: {:?}, {:?}", expr, peek);
        match (&expr, self.peek()) {
            (Expr::Ident(callee), Token::LParen) => {
                self.advance();  // LParen
                let mut args: Vec<Expr> = Vec::new();
                while !self.matches_type(Token::RParen) {
                    args.push(self.expression()?);
                    if self.matches_type(Token::Comma) {
                        self.advance();
                    }
                }
                if args.len() > 255 {
                    return Err(format!("Too many arguments for function {}", callee))
                }
                self.advance();  // RParen
                Ok(Expr::Call { callee: callee.into(), args: args })
            },
            _ => Ok(expr)
        }
    }

    // Rule: atom -> "(" expression ")" | Integer | Identifier | Boolean
    fn atom(&mut self) -> Result<Expr, String> {
        let expr: Expr = match self.peek() {
            Token::Int(n) => Expr::Int(*n),
            Token::Ident(s) => Expr::Ident(s.into()),
            Token::Boolean(b) => Expr::Boolean(*b),
            Token::LParen => {
                self.advance();  // Move past the LParen "("
                let expr: Expr = self.expression()?;
                // Revisit: Is it still needed after parentheses balance checking?
                // if self.pos >= self.tokens.len() || self.peek() != &Token::RParen {
                //     return Err("Missing closing parenthesis".into())
                // }
                expr
            },
            _ => return Err(format!("Invalid atom: {:?}", *self.peek()))
        };
        self.advance();
        Ok(expr)
    }

}

#[cfg(test)]
mod tests {
    use std::vec;

    use crate::parser::Parser;
    use crate::tokens::Token as T;
    use crate::ast::*;

    #[test]
    fn token_type_check() {
        let p = Parser::new(vec![T::Int(5)]);

        assert_eq!(p.matches_type(T::Int(-9999)), true);
        assert_eq!(p.matches_type(T::RSquirly), false);

        assert_eq!(p.matches_types(vec![T::Int(0)]), true);
        assert_eq!(p.matches_types(vec![T::EndLine, T::Colon, T::Int(9999)]), true);
        assert_eq!(p.matches_types(vec![T::LParen, T::Let, T::Function]), false);
        assert_eq!(p.matches_types(vec![T::Equal]), false);
    }

    #[test]
    fn simple() {
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Int(0))]), Parser::new(vec![T::Int(0), T::EndLine, T::EOF]).parse()
        );
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Int(5))]), Parser::new(vec![T::Int(5), T::EndLine, T::EOF]).parse()
        );
    }

    #[test]
    fn bad_token_combo_error_check() {
        assert_eq!(
            Err("Two Int tokens in a row".into()),
            Parser::new(vec![T::Int(0), T::Int(0), T::EOF]).parse()
        );
        assert_eq!(
            Err("Two Int tokens in a row".into()),
            Parser::new(vec![T::Int(0), T::Plus, T::Int(0), T::Int(0), T::EndLine, T::EOF]).parse()
        );
        assert_eq!(
            Err("Two Ident tokens in a row".into()),
            Parser::new(vec![T::Ident("one".into()), T::Ident("two".into()), T::EndLine, T::EOF]).parse()
        );
        assert_eq!(
            Err("Ident token immediately followed by Int token".into()),
            Parser::new(vec![T::Ident("one".into()), T::Int(0), T::EndLine, T::EOF]).parse()
        );
        assert_eq!(
            Err("Ident token immediately followed by Int token".into()),
            Parser::new(vec![T::Ident("one".into()), T::Int(0), T::EndLine, T::EOF]).parse()
        );
    }

    #[test]
    fn bracket_balance() {
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Int(0))]),
            Parser::new(vec![T::LParen, T::LParen, T::LParen, T::Int(0),
                             T::RParen, T::RParen, T::RParen, T::EndLine, T::EOF]).parse()
        );
        assert_eq!(
            Err("Inbalanced brackets".into()),
            Parser::new(vec![T::RParen, T::EOF]).parse()
        );
        assert_eq!(
            Err("Inbalanced brackets".into()),
            Parser::new(vec![T::LParen, T::LParen, T::RParen, T::EndLine, T::EOF]).parse()
        );
        assert_eq!(
            Err("Inbalanced brackets".into()),
            Parser::new(vec![T::LParen, T::RParen, T::LParen,
                             T::Ident("a".into()), T::EndLine, T::EOF]).parse()
        );
        assert_eq!(
            Err("Inbalanced brackets".into()),
            Parser::new(vec![T::LParen, T::LParen, T::LParen,
                             T::RParen, T::RParen, T::RParen, T::RParen, T::EndLine, T::EOF]).parse()
        );

    }

    #[test]        
    fn simple_monadic() {
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Monadic {
                operator: Operator::Plus,
                operand: Box::new(Expr::Int(256))
            })]),
            // +256;
            Parser::new(vec![T::Plus, T::Int(256), T::EndLine, T::EOF]).parse()
        );
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Monadic {
                operator: Operator::Minus,
                operand: Box::new(Expr::Int(15))
            })]),
            // -15;
            Parser::new(vec![T::Minus, T::Int(15), T::EndLine, T::EOF]).parse()
        );
    }

    #[test]
    fn simple_dyadic() {
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Dyadic {
                operator: Operator::Plus,
                left: Box::new(Expr::Int(5)),
                right: Box::new(Expr::Int(6))
            })]),
            // 5 + 6;
            Parser::new(vec![T::Int(5), T::Plus, T::Int(6), T::EndLine, T::EOF]).parse()
        );
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Dyadic {
                operator: Operator::Minus,
                left: Box::new(Expr::Int(20)),
                right: Box::new(Expr::Int(5))
            })]),
            // 20 - 5;
            Parser::new(vec![T::Int(20), T::Minus, T::Int(5), T::EndLine, T::EOF]).parse()
        );
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Dyadic {
                operator: Operator::Minus,
                left: Box::new(Expr::Int(256)),
                right: Box::new(Expr::Monadic {
                    operator: Operator::Minus,
                    operand: Box::new(Expr::Int(256))
                })
            })]),
            // 256 - -256;
            Parser::new(vec![T::Int(256), T::Minus, T::Minus, T::Int(256), T::EndLine, T::EOF]).parse()
        );
    }

    #[test]
    fn multiple_dyadic() {
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Dyadic {
                operator: Operator::Plus,
                left: Box::new(Expr::Dyadic {
                    operator: Operator::Plus,
                    left: Box::new(Expr::Int(1)),
                    right: Box::new(Expr::Int(2))
                }),
                right: Box::new(Expr::Int(3))
            })]),
            // 1 + 2 + 3;
            Parser::new(vec![T::Int(1), T::Plus, T::Int(2),
                             T::Plus, T::Int(3), T::EndLine, T::EOF]).parse()
        );
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Dyadic {
                operator: Operator::Plus,
                left: Box::new(Expr::Dyadic {
                    operator: Operator::Minus,
                    left: Box::new(Expr::Dyadic {
                        operator: Operator::Plus,
                        left: Box::new(Expr::Int(1)),
                        right: Box::new(Expr::Ident("two".into()))
                    }),
                    right: Box::new(Expr::Int(4))
                }),
                right: Box::new(Expr::Monadic {
                    operator: Operator::Minus,
                    operand: Box::new(Expr::Int(8))
                })
            })]),
            // 1 + two - 4 + -8;
            Parser::new(vec![T::Int(1), T::Plus, T::Ident("two".into()), T::Minus, T::Int(4),
                             T::Plus, T::Minus, T::Int(8), T::EndLine, T::EOF]).parse()
        );

        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Dyadic {
                operator: Operator::Plus,
                left: Box::new(Expr::Dyadic {
                    operator: Operator::Plus,
                    left: Box::new(Expr::Int(10)),
                    right: Box::new(Expr::Monadic {
                        operator: Operator::Minus,
                        operand: Box::new(Expr::Int(20))
                    })
                }),
                right: Box::new(Expr::Monadic {
                    operator: Operator::Minus,
                    operand: Box::new(Expr::Int(30))
                })
            })]),
            // 10 + -20 + -30;
            Parser::new(vec![T::Int(10), T::Plus, T::Minus, T::Int(20),
                             T::Plus, T::Minus, T::Int(30), T::EndLine, T::EOF]).parse()
        );
    }

    #[test]
    fn precedence() {
        // Operator precedence
        let n_plus_n_minus_n = vec![Stmt::Expr(Expr::Dyadic {
            operator: Operator::Minus,
            left: Box::new(Expr::Dyadic {
                operator: Operator::Plus,
                left: Box::new(Expr::Int(512)),
                right: Box::new(Expr::Int(256))
            }),
            right: Box::new(Expr::Int(128))
        })];
        assert_eq!(
            Ok(&n_plus_n_minus_n),
            // 512 + 256 - 128;
            Parser::new(vec![T::Int(512), T::Plus, T::Int(256),
                             T::Minus, T::Int(128), T::EndLine, T::EOF]).parse().as_ref()
        );
        let n_plus_n_slash_n = vec![Stmt::Expr(Expr::Dyadic {
            operator: Operator::Plus,
            left: Box::new(Expr::Int(512)),
            right: Box::new(Expr::Dyadic {
                operator: Operator::Slash,
                left: Box::new(Expr::Int(256)),
                right: Box::new(Expr::Int(128))
            }),
        })];
        assert_eq!(
            Ok(&n_plus_n_slash_n),
            // 512 + 256 / 128;
            Parser::new(vec![T::Int(512), T::Plus, T::Int(256),
                             T::Slash, T::Int(128), T::EndLine, T::EOF]).parse().as_ref()
        );
        // Parentheses precedence
        assert_eq!(
            Ok(&n_plus_n_minus_n),
            // ( 512 + 256 - 128);
            Parser::new(vec![T::LParen, T::Int(512), T::Plus, T::Int(256),
                             T::Minus, T::Int(128), T::RParen, T::EndLine, T::EOF]).parse().as_ref()
        );
        assert_eq!(
            Ok(&n_plus_n_slash_n),
            // ( 512 + 256 / 128 );
            Parser::new(vec![T::LParen, T::Int(512), T::Plus, T::Int(256),
                             T::Slash, T::Int(128), T::RParen, T::EndLine, T::EOF]).parse().as_ref()
        );
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Dyadic {
                operator: Operator::Plus,
                left: Box::new(Expr::Int(512)),
                right: Box::new(Expr::Dyadic {
                    operator: Operator::Plus,
                    left: Box::new(Expr::Int(256)),
                    right: Box::new(Expr::Ident("one_two_eight".into()))
                }),
            })]),
            // 512 + (256 + one_two_eight);
            Parser::new(vec![T::Int(512), T::Plus, T::LParen,
                             T::Int(256), T::Plus, T::Ident("one_two_eight".into()),
                             T::RParen, T::EndLine, T::EOF]).parse()
        );

        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Dyadic {
                operator: Operator::Star,
                left: Box::new(Expr::Dyadic {
                    operator: Operator::Slash,
                    left: Box::new(Expr::Int(16)),
                    right: Box::new(Expr::Int(4))
                }),
                right: Box::new(Expr::Monadic {
                    operator: Operator::Minus,
                    operand: Box::new(Expr::Int(1))
                })
            })]),
            // 16 / 4 * -1;
            Parser::new(vec![T::Int(16), T::Slash, T::Int(4),
                             T::Star, T::Minus, T::Int(1), T::EndLine, T::EOF]).parse()
        )
    }

    #[test]
    fn control_flow() {
        assert_eq!(
            // if (true) {}
            Ok(vec![Stmt::If {cond: Expr::Boolean(true),
                              then: Box::new(Stmt::Block(vec![])),
                              els: Box::new(Stmt::Block(vec![]))} ]),
            Parser::new(vec![T::If, T::LParen, T::Boolean(true), T::RParen,
                             T::LSquirly, T::RSquirly, T::EOF]).parse()
        );
        assert_eq!(
            // if (a == 8) { print(0); }
            Ok(vec![Stmt::If{
                cond: Expr::Dyadic {
                    operator: Operator::EqualTo,
                    left: Box::new(Expr::Ident("a".into())),
                    right: Box::new(Expr::Int(8))
                },
                then: Box::new(Stmt::Block(vec![Stmt::Print(Expr::Int(0))])),
                els: Box::new(Stmt::Block(vec![]))}]),
            Parser::new(vec![T::If, T::LParen, T::Ident("a".into()), T::EqualTo, T::Int(8), T::RParen,
                            T::LSquirly,
                                T::Print, T::LParen, T::Int(0), T::RParen, T::EndLine,
                            T::RSquirly, T::EOF]).parse()
        );
        assert_eq!(
            // if (a != 8) { print(0); }
            Ok(vec![Stmt::If{
                cond: Expr::Dyadic {
                    operator: Operator::NotEqualTo,
                    left: Box::new(Expr::Ident("a".into())),
                    right: Box::new(Expr::Int(8))
                },
                then: Box::new(Stmt::Block(vec![Stmt::Print(Expr::Int(0))])),
                els: Box::new(Stmt::Block(vec![]))}]),
            Parser::new(vec![T::If, T::LParen, T::Ident("a".into()), T::Negate, T::Equal, T::Int(8), T::RParen,
                            T::LSquirly,
                                T::Print, T::LParen, T::Int(0), T::RParen, T::EndLine,
                            T::RSquirly, T::EOF]).parse()
        );
        assert_eq!(
            // if (a < b) { print(0); } else { print(1); print(2); }
            Ok(vec![Stmt::If{
                cond: Expr::Dyadic {
                    operator: Operator::LessThan,
                    left: Box::new(Expr::Ident("abc".into())),
                    right: Box::new(Expr::Monadic {
                        operator: Operator::Minus,
                        operand: Box::new(Expr::Int(8))
                    })
                },
                then: Box::new(Stmt::Block(vec![Stmt::Print(Expr::Int(0))])),
                els: Box::new(Stmt::Block(vec![Stmt::Print(Expr::Int(1)), Stmt::Print(Expr::Int(2))]))}]),
            Parser::new(vec![T::If, T::LParen, T::Ident("abc".into()), T::LessThan, T::Minus, T::Int(8), T::RParen,
                            T::LSquirly,
                                T::Print, T::LParen, T::Int(0), T::RParen, T::EndLine,
                            T::RSquirly, T::Else, T::LSquirly,
                                T::Print, T::LParen, T::Int(1), T::RParen, T::EndLine,
                                T::Print, T::LParen, T::Int(2), T::RParen, T::EndLine,
                            T::RSquirly, T::EOF]).parse()
        );
        assert_eq!(
            // if (false || true) {}
            Ok(vec![Stmt::If {
                cond: Expr::Logical {
                    operator: Operator::LogicalOr,
                    left: Box::new(Expr::Boolean(false)),
                    right: Box::new(Expr::Boolean(true))
                },
                then: Box::new(Stmt::Block(vec![])),
                els: Box::new(Stmt::Block(vec![]))
            }]),
            Parser::new(vec![T::If, T::LParen, T::Boolean(false), T::Or, T::Boolean(true), T::RParen,
                    T::LSquirly, T::RSquirly, T::EOF]).parse()
        );
        assert_eq!(
            // if (0 <= a && a <= 4) {}
            Ok(vec![Stmt::If {
                cond: Expr::Logical {
                    operator: Operator::LogicalAnd,
                    left: Box::new(Expr::Dyadic {
                        operator: Operator::LessEquals,
                        left: Box::new(Expr::Int(0)),
                        right: Box::new(Expr::Ident("a".into()))
                    }),
                    right: Box::new(Expr::Dyadic {
                        operator: Operator::LessEquals,
                        left: Box::new(Expr::Ident("a".into())),
                        right: Box::new(Expr::Int(4))
                    })
                },
                then: Box::new(Stmt::Block(vec![])),
                els: Box::new(Stmt::Block(vec![]))
            }]),
            Parser::new(vec![T::If, T::LParen,
                                T::Int(0), T::LessEquals, T::Ident("a".into()), T::And,
                                T::Ident("a".into()), T::LessEquals, T::Int(4), T::RParen,
                    T::LSquirly, T::RSquirly, T::EOF]).parse()
        );
        // if (true || true && true) {}  // Logical And has higher precedence over Logical Or
        assert_eq!(
            Ok(vec![Stmt::If {
                cond: Expr::Logical {
                    operator: Operator::LogicalOr,
                    left: Box::new(Expr::Boolean(true)),
                    right: Box::new(Expr::Logical {
                        operator: Operator::LogicalAnd,
                        left: Box::new(Expr::Boolean(true)),
                        right: Box::new(Expr::Boolean(true))
                    })
                },
                then: Box::new(Stmt::Block(vec![])),
                els: Box::new(Stmt::Block(vec![]))
            }]),
            Parser::new(vec![T::If, T::LParen, T::Boolean(true), T::Or, T::Boolean(true), T::And,
                             T::Boolean(true), T::RParen, T::LSquirly, T::RSquirly, T::EOF]).parse()
        )
    }

    #[test]
    fn variables() {
        // var abcdefg;
        assert_eq!(
            Ok(vec![Stmt::VarDecl("abcdefg".into(), None)]),
            Parser::new(vec![T::Var, T::Ident("abcdefg".into()), T::EndLine, T::EOF]).parse()
        );
        // var abc = 6;
        assert_eq!(
            Ok(vec![Stmt::VarDecl("abc".into(),
                            Some(Box::new(Stmt::Expr(Expr::Int(6)))))]),
            Parser::new(vec![T::Var, T::Ident("abc".into()), T::Equal,
                             T::Int(6), T::EndLine, T::EOF]).parse()
        );
        // var abc = 3 + 2;
        assert_eq!(
            Ok(vec![Stmt::VarDecl("abc".into(),
                            Some(Box::new(Stmt::Expr(Expr::Dyadic {
                                                        operator: Operator::Plus,
                                                        left: Box::new(Expr::Int(3)),
                                                        right: Box::new(Expr::Int(2))
                                                    }))))]),
            Parser::new(vec![T::Var, T::Ident("abc".into()), T::Equal, T::Int(3),
                             T::Plus, T::Int(2), T::EndLine, T::EOF]).parse()
        );
        // var b = if (false) { 0; } else { a; };
        assert_eq!(
            Ok(vec![Stmt::VarDecl("b".into(),
                            Some(Box::new(Stmt::If {
                                cond: Expr::Boolean(false),
                                then: Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(0))])),
                                els: Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Ident("a".into()))]))
                            })))]),
            Parser::new(vec![T::Var, T::Ident("b".into()), T::Equal,
                             T::If,
                                T::LParen, T::Boolean(false), T::RParen,
                                T::LSquirly, T::Int(0), T::EndLine, T::RSquirly,
                             T::Else,
                                T::LSquirly, T::Ident("a".into()), T::EndLine, T::RSquirly,
                             T::EndLine, T::EOF]).parse()
        );
        // abc = 10;  // assumes declared abc variable
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Assign {
                var_name: "abc".into(), new_value: Box::new(Expr::Int(10))
            })]),
            Parser::new(vec![T::Ident("abc".into()), T::Equal,
                             T::Int(10), T::EndLine, T::EOF]).parse()
        );
    }

    #[test]
    fn loops() {
        // while (true) {}
        assert_eq!(
            Ok(vec![Stmt::While { cond: Expr::Boolean(true), body: Box::new(Stmt::Block(vec![])) }]),
            Parser::new(vec![T::While, T::LParen, T::Boolean(true), T::RParen,
                T::LSquirly, T::RSquirly, T::EOF]).parse()
        );

        // var i = 0;
        // while (i < 5) {
        //     i = i + 1;
        // }
        // i;
        assert_eq!(Ok(vec![
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
            ]),
            Parser::new(vec![
                T::Var, T::Ident("i".into()), T::Equal, T::Int(0), T::EndLine,
                T::While, T::LParen, T::Ident("i".into()), T::LessThan, T::Int(5), T::RParen, T::LSquirly,
                    T::Ident("i".into()), T::Equal, T::Ident("i".into()), T::Plus, T::Int(1), T::EndLine,
                T::RSquirly,
                T::Ident("i".into()), T::EndLine, T::EOF
            ]).parse()
        );

        // for (;;) {}
        assert_eq!(
            Ok(vec![Stmt::While { cond: Expr::Boolean(true), body: Box::new(Stmt::Block(vec![])) }]),
            Parser::new(vec![T::For, T::LParen, T::EndLine, T::EndLine, T::RParen,
                T::LSquirly, T::RSquirly, T::EOF]).parse()
        );
        
        // for (var i = 0; i < 10; i = i + 1) {}
        assert_eq!(Ok(vec![
                Stmt::Block(vec![
                    Stmt::VarDecl("i".into(), Some(Box::new(Stmt::Expr(Expr::Int(0))))),
                    Stmt::While {
                        cond: Expr::Dyadic {
                            operator: Operator::LessThan,
                            left: Box::new(Expr::Ident("i".into())),
                            right: Box::new(Expr::Int(10))
                        },
                        body: Box::new(Stmt::Block(vec![
                            Stmt::Expr(Expr::Assign {
                                var_name: "i".into(),
                                new_value: Box::new(Expr::Dyadic {
                                    operator: Operator::Plus,
                                    left: Box::new(Expr::Ident("i".into())),
                                    right: Box::new(Expr::Int(1))
                                })
                            })
                        ]))
                    }
                ]),
            ]),
            Parser::new(vec![
                T::For, T::LParen, T::Var, T::Ident("i".into()), T::Equal, T::Int(0), T::EndLine,
                    T::Ident("i".into()), T::LessThan, T::Int(10), T::EndLine,
                    T::Ident("i".into()), T::Equal, T::Ident("i".into()), T::Plus, T::Int(1),
                T::RParen, T::LSquirly, T::RSquirly, T::EOF
            ]).parse()
        );

        // for (var i = 0; i < 10; i = i + 1) { print(i/2); }
        assert_eq!(Ok(vec![
            Stmt::Block(vec![
                Stmt::VarDecl("i".into(), Some(Box::new(Stmt::Expr(Expr::Int(0))))),
                Stmt::While {
                    cond: Expr::Dyadic {
                        operator: Operator::LessThan,
                        left: Box::new(Expr::Ident("i".into())),
                        right: Box::new(Expr::Int(10))
                    },
                    body: Box::new(Stmt::Block(vec![
                        Stmt::Print(Expr::Dyadic {
                            operator: Operator::Slash,
                            left: Box::new(Expr::Ident("i".into())),
                            right: Box::new(Expr::Int(2))
                        }),
                        Stmt::Expr(Expr::Assign {
                            var_name: "i".into(),
                            new_value: Box::new(Expr::Dyadic {
                                operator: Operator::Plus,
                                left: Box::new(Expr::Ident("i".into())),
                                right: Box::new(Expr::Int(1))
                            })
                        })
                    ]))
                }
            ]),
        ]),
        Parser::new(vec![
            T::For, T::LParen, T::Var, T::Ident("i".into()), T::Equal, T::Int(0), T::EndLine,
                T::Ident("i".into()), T::LessThan, T::Int(10), T::EndLine,
                T::Ident("i".into()), T::Equal, T::Ident("i".into()), T::Plus, T::Int(1),
            T::RParen, T::LSquirly,
                T::Print, T::LParen, T::Ident("i".into()), T::Slash, T::Int(2), T::RParen, T::EndLine,
            T::RSquirly, T::EOF
        ]).parse());
    }

    #[test]
    fn function_calls() {
        // foo();
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Call { callee: "foo".into(), args: vec![] })]),
            Parser::new(vec![T::Ident("foo".into()), T::LParen, T::RParen, T::EndLine, T::EOF]).parse()
        );

        // bar(1);
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Call {
                callee: "bar".into(),
                args: vec![Expr::Int(1)]
            })]),
            Parser::new(vec![T::Ident("bar".into()), T::LParen, T::Int(1), T::RParen, T::EndLine, T::EOF]).parse()
        );

        // max(0, foo());
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Call {
                callee: "max".into(),
                args: vec![Expr::Int(0), Expr::Call { callee: "foo".into(), args: vec![] }]
            })]),
            Parser::new(vec![T::Ident("max".into()), T::LParen,
                                T::Int(0), T::Comma,
                                T::Ident("foo".into()), T::LParen,
                             T::RParen, T::RParen, T::EndLine, T::EOF]).parse()
        );

        // baz(1, 2, 3, 4);
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Call {
                callee: "baz".into(),
                args: vec![Expr::Int(1), Expr::Int(2), Expr::Int(3), Expr::Int(4)]
            })]),
            Parser::new(vec![T::Ident("baz".into()), T::LParen,
                                T::Int(1), T::Comma,
                                T::Int(2), T::Comma,
                                T::Int(3), T::Comma,
                                T::Int(4),
                             T::RParen, T::EndLine, T::EOF]).parse()
        );
    }

    #[test]
    fn function_declarations() {
        // fn no_params() {
        //     print(1);
        // }
        assert_eq!(
            Ok(vec![Stmt::FnDecl {
                name: "no_params".into(),
                parameters: vec![],
                body: Box::new(Stmt::Block(vec![Stmt::Print(Expr::Int(1))]))
            }]),
            Parser::new(vec![T::Function, T::Ident("no_params".into()), T::LParen, T::RParen, T::LSquirly,
                            T::Print, T::LParen, T::Int(1), T::RParen, T::EndLine, T::RSquirly, T::EOF]).parse()
        );

        // fn one_param(abc) {
        //     print(abc);
        // }
        assert_eq!(
            Ok(vec![Stmt::FnDecl {
                name: "one_param".into(),
                parameters: vec!["abc".into()],
                body: Box::new(Stmt::Block(vec![Stmt::Print(Expr::Ident("abc".into()))]))
            }]),
            Parser::new(vec![T::Function, T::Ident("one_param".into()), T::LParen, T::Ident("abc".into()), T::RParen,
                             T::LSquirly, T::Print, T::LParen, T::Ident("abc".into()), T::RParen, T::EndLine,
                             T::RSquirly, T::EOF]).parse()
        );

        // fn multi_param(abc, def) {
        //     abc = abc + def;
        //     print(abc);
        // }
        assert_eq!(
            Ok(vec![Stmt::FnDecl {
                name: "multi_param".into(),
                parameters: vec!["abc".into(), "def".into()],
                body: Box::new(Stmt::Block(vec![
                    Stmt::Expr(Expr::Assign {
                        var_name: "abc".into(),
                        new_value: Box::new(Expr::Dyadic {
                            operator: Operator::Plus,
                            left: Box::new(Expr::Ident("abc".into())),
                            right: Box::new(Expr::Ident("def".into()))
                        })
                    }),
                    Stmt::Print(Expr::Ident("abc".into()))
                ]))
            }]),
            Parser::new(vec![T::Function, T::Ident("multi_param".into()), T::LParen,
                                T::Ident("abc".into()), T::Comma, T::Ident("def".into()),
                             T::RParen, T::LSquirly,
                                T::Ident("abc".into()), T::Equal, T::Ident("abc".into()), T::Plus,
                                        T::Ident("def".into()), T::EndLine,
                                T::Print, T::LParen, T::Ident("abc".into()), T::RParen, T::EndLine,
                             T::RSquirly, T::EOF]).parse()
        );
    }

}