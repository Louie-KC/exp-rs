use std::mem::{discriminant};

use crate::tokens::{Token, self};
use crate::ast::*;

/*
 * Recursive Descent
 * 
 * rules:
 *    program -> statement*
 *  statement -> print | expression ";"
 *      print -> "print(" expression ");"
 * expression -> term
 *       term -> factor ( ( "+" | "-" ) factor)*
 *     factor -> unary ( ( "*" | "/" ) unary)*
 *      unary -> atom | "+" unary | "-" unary
 *       atom -> "(" expression ")" | Integer | Identifier
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

        let mut statements: Vec<Stmt> = Vec::new();

        while *self.peek() != Token::EOF {
            statements.push(self.statement());
        }

        Ok(statements)
    }

    fn statement(&mut self) -> Stmt {
        match *self.peek() {
            Token::Print => self.print_statement(),
            _            => self.expression_statement()
        }
    }

    fn print_statement(&mut self) -> Stmt {
        self.advance();
        let val = self.expression();
        if *self.peek() != Token::EndLine {
            panic!("print statement expected \";\" to end the statement");
        }
        self.advance();  // ;
        Stmt::Print(val.unwrap())
    }

    fn expression_statement(&mut self) -> Stmt {
        let expr = self.expression();
        if *self.peek() != Token::EndLine {
            panic!("expected \";\" to end statement");
        }
        self.advance();  // ;
        Stmt::Expr(expr.unwrap())
    }

    fn eval_operator(&mut self) -> Result<Operator, String> {
        let op = match self.tokens.get(self.pos) {
            Some(Token::Plus) => Operator::Plus,
            Some(Token::Minus) => Operator::Minus,
            Some(Token::Star) => Operator::Star,
            Some(Token::Slash) => Operator::Slash,
            _ => return Err("Invalid operator".into())
        };
        Ok(op)
    }

    fn peek(&self) -> &Token {
        &self.tokens.get(self.pos).unwrap()
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

    fn illegal_token_seq_check(&self) -> Result<u8, String> {
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
        Ok(0)
    }

    fn brackets_balanced(&self) -> bool {
        let mut stack: Vec<Token> = Vec::new();

        for token in self.tokens.iter() {
            match token {
                Token::LParen => stack.push(Token::LParen),
                Token::LSquirly => stack.push(Token::LSquirly),
                Token::RParen if stack.pop() != Some(Token::LParen) => return false,
                // Token::RSquirly if stack.pop() != Some(Token::LSquirly) => return false,
                _ => {}
            }
        }

        stack.is_empty()
    }

    fn advance(&mut self) -> () {
        self.pos += 1
    }

    // Rule: expression -> term
    fn expression(&mut self) -> Result<Expr, String> {
        // println!("expression called");
        self.term()
    }

    // Rule: term -> factor ( ( "+" | "-" ) factor)*
    fn term(&mut self) -> Result<Expr, String> {
        // println!("term called");
        let mut expr = self.factor()?;

        while self.matches_types(vec![Token::Plus, Token::Minus]) {
            let op = self.eval_operator()?;
            self.advance();
            let right = self.factor()?;
            expr = Expr::Dyadic { operator: op, left: Box::new(expr), right: Box::new(right) }
        }
        Ok(expr)
    }

    // Rule: factor -> unary ( ( "*" | "/" ) unary)*
    fn factor(&mut self) -> Result<Expr, String> {
        // println!("factor called");
        let mut expr = self.unary()?;

        // allowed tokens
        while self.matches_types(vec![Token::Star, Token::Slash]) {
            let op = self.eval_operator()?;
            self.advance();
            let right = self.unary()?;
            expr = Expr::Dyadic { operator: op, left: Box::new(expr), right: Box::new(right) }
        }
        Ok(expr)
    }

    // Rule: unary -> atom | "+" unary | "-" unary
    fn unary(&mut self) -> Result<Expr, String> {
        // println!("unary called");
        if self.matches_type(Token::Plus) {
            self.advance();
            return Ok(Expr::Monadic { operator: Operator::Plus, operand: Box::new(self.unary()?) })
        }
        if self.matches_type(Token::Minus) {
            self.advance();
            return Ok(Expr::Monadic { operator: Operator::Minus, operand: Box::new(self.unary()?) })
        }
        self.atom()
    }

    // Rule: atom -> "(" expression ")" | Integer | Identifier
    fn atom(&mut self) -> Result<Expr, String> {
        let expr = match self.peek() {
            Token::Int(n) => Expr::Int(n.clone()),
            Token::Ident(s) => Expr::Ident(s.into()),
            Token::LParen => {
                self.advance();  // Move past the LParen "("
                let expr = self.expression()?;
                // Revisit: Is it still needed after parentheses balance checking?
                // if self.pos >= self.tokens.len() || self.peek() != &Token::RParen {
                //     return Err("Missing closing parenthesis".into())
                // }
                expr
            },
            _ => return Err("Invalid atom".into())
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

}