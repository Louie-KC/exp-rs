use std::mem::{discriminant};

use crate::tokens::{Token, self};
use crate::ast::*;

/*
 * Recursive Descent
 * 
 * rules:
 * expression -> term
 *       term -> factor ( ( "+" | "-" ) factor)*
 *     factor -> unary ( ( "*" | "/" ) unary)*
 *      unary -> atom | "-" unary
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

    pub fn parse(&mut self) -> Result<Expr, String> {
        self.eval()
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

    fn eval(&mut self) -> Result<Expr, String> {
        match self.tokens.len() {
            0 => Err("No tokens to evaluate".into()),
            _ => {
                if !self.brackets_balanced() {
                    return Err("Inbalanced brackets".into())
                }
                match self.illegal_token_seq_check() {
                    Err(e) => return Err(e),
                    _              => self.expression()
                }
            }
        }
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
        println!("factor called");
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

    // Rule: unary -> atom | "-" unary
    fn unary(&mut self) -> Result<Expr, String> {
        // println!("unary called");
        if self.matches_type(Token::Plus) {
            self.advance();
            return Ok(Expr::Monadic { operator: Operator::Plus, operand: Box::new(self.atom()?) })
        }
        if self.matches_type(Token::Minus) {
            self.advance();
            return Ok(Expr::Monadic { operator: Operator::Minus, operand: Box::new(self.atom()?) })
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
    use crate::tokens::Token;
    use crate::ast::*;

    #[test]
    fn token_type_check() {
        let p = Parser::new(vec![Token::Int(5)]);

        assert_eq!(p.matches_type(Token::Int(-9999)), true);
        assert_eq!(p.matches_type(Token::RSquirly), false);

        assert_eq!(p.matches_types(vec![Token::Int(0)]), true);
        assert_eq!(p.matches_types(vec![Token::EndLine, Token::Colon, Token::Int(9999)]), true);
        assert_eq!(p.matches_types(vec![Token::LParen, Token::Let, Token::Function]), false);
        assert_eq!(p.matches_types(vec![Token::Equal]), false);
    }

    #[test]
    fn simple() {
        assert_eq!(
            Err("No tokens to evaluate".into()), Parser::new(vec![]).parse()
        );
        assert_eq!(
            Ok(Expr::Int(0)), Parser::new(vec![Token::Int(0)]).parse()
        );
        assert_eq!(
            Ok(Expr::Int(5)), Parser::new(vec![Token::Int(5)]).parse()
        );
    }

    #[test]
    fn bad_token_combo_error_check() {
        assert_eq!(
            Err("Two Int tokens in a row".into()),
            Parser::new(vec![Token::Int(0), Token::Int(0)]).parse()
        );
        assert_eq!(
            Err("Two Int tokens in a row".into()),
            Parser::new(vec![Token::Int(0), Token::Plus, Token::Int(0), Token::Int(0)]).parse()
        );
        assert_eq!(
            Err("Two Ident tokens in a row".into()),
            Parser::new(vec![Token::Ident("one".into()), Token::Ident("two".into())]).parse()
        );
        assert_eq!(
            Err("Ident token immediately followed by Int token".into()),
            Parser::new(vec![Token::Ident("one".into()), Token::Int(0)]).parse()
        );
        assert_eq!(
            Err("Ident token immediately followed by Int token".into()),
            Parser::new(vec![Token::Ident("one".into()), Token::Int(0)]).parse()
        );
    }

    #[test]
    fn bracket_balance() {
        assert_eq!(
            Ok(Expr::Int(0)),
            Parser::new(vec![Token::LParen, Token::LParen, Token::LParen, Token::Int(0),
                             Token::RParen, Token::RParen, Token::RParen]).parse()
        );
        assert_eq!(
            Err("Inbalanced brackets".into()),
            Parser::new(vec![Token::RParen]).parse()
        );
        assert_eq!(
            Err("Inbalanced brackets".into()),
            Parser::new(vec![Token::LParen, Token::LParen, Token::RParen]).parse()
        );
        assert_eq!(
            Err("Inbalanced brackets".into()),
            Parser::new(vec![Token::LParen, Token::RParen, Token::LParen,
                             Token::Ident("a".into())]).parse()
        );
        assert_eq!(
            Err("Inbalanced brackets".into()),
            Parser::new(vec![Token::LParen, Token::LParen, Token::LParen,
                             Token::RParen, Token::RParen, Token::RParen, Token::RParen]).parse()
        );

    }

    #[test]        
    fn simple_monadic() {
        assert_eq!(
            Ok(Expr::Monadic {
                operator: Operator::Plus,
                operand: Box::new(Expr::Int(256))
            }),
            Parser::new(vec![Token::Plus, Token::Int(256)]).parse()
        );
        assert_eq!(
            Ok(Expr::Monadic {
                operator: Operator::Minus,
                operand: Box::new(Expr::Int(15))
            }),
            Parser::new(vec![Token::Minus, Token::Int(15)]).parse()
        );
    }

    #[test]
    fn simple_dyadic() {
        assert_eq!(
            Ok(Expr::Dyadic {
                operator: Operator::Plus,
                left: Box::new(Expr::Int(5)),
                right: Box::new(Expr::Int(6))
            }),
            Parser::new(vec![Token::Int(5), Token::Plus, Token::Int(6)]).parse()
        );
        assert_eq!(
            Ok(Expr::Dyadic {
                operator: Operator::Minus,
                left: Box::new(Expr::Int(20)),
                right: Box::new(Expr::Int(5))
            }),
            Parser::new(vec![Token::Int(20), Token::Minus, Token::Int(5)]).parse()
        );
        assert_eq!(
            Ok(Expr::Dyadic {
                operator: Operator::Minus,
                left: Box::new(Expr::Int(256)),
                right: Box::new(Expr::Monadic {
                    operator: Operator::Minus,
                    operand: Box::new(Expr::Int(256))
                })
            }),
            Parser::new(vec![Token::Int(256), Token::Minus, Token::Minus, Token::Int(256)]).parse()
        );
    }

    #[test]
    fn multiple_dyadic() {
        assert_eq!(
            Ok(Expr::Dyadic {
                operator: Operator::Plus,
                left: Box::new(Expr::Dyadic {
                    operator: Operator::Plus,
                    left: Box::new(Expr::Int(1)),
                    right: Box::new(Expr::Int(2))
                }),
                right: Box::new(Expr::Int(3))
            }),
            Parser::new(vec![Token::Int(1), Token::Plus, Token::Int(2),
                             Token::Plus, Token::Int(3)]).parse()
        );
        assert_eq!(
            Ok(Expr::Dyadic {
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
            }),
            Parser::new(vec![Token::Int(1), Token::Plus,
                             Token::Ident("two".into()), Token::Minus,
                             Token::Int(4), Token::Plus,
                             Token::Minus, Token::Int(8)]).parse()
        );

        assert_eq!(
            Ok(Expr::Dyadic {
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
            }),
            Parser::new(vec![Token::Int(10), Token::Plus, Token::Minus, Token::Int(20),
                             Token::Plus, Token::Minus, Token::Int(30)]).parse()
        );
    }

    #[test]
    fn precedence() {
        // Operator precedence
        let n_plus_n_minus_n = Expr::Dyadic {
            operator: Operator::Minus,
            left: Box::new(Expr::Dyadic {
                operator: Operator::Plus,
                left: Box::new(Expr::Int(512)),
                right: Box::new(Expr::Int(256))
            }),
            right: Box::new(Expr::Int(128))
        };
        assert_eq!(
            Ok(&n_plus_n_minus_n),
            Parser::new(vec![Token::Int(512), Token::Plus, Token::Int(256),
                             Token::Minus, Token::Int(128)]).parse().as_ref()
        );
        let n_plus_n_slash_n = Expr::Dyadic {
            operator: Operator::Plus,
            left: Box::new(Expr::Int(512)),
            right: Box::new(Expr::Dyadic {
                operator: Operator::Slash,
                left: Box::new(Expr::Int(256)),
                right: Box::new(Expr::Int(128))
            }),
        };
        assert_eq!(
            Ok(&n_plus_n_slash_n),
            Parser::new(vec![Token::Int(512), Token::Plus, Token::Int(256),
                             Token::Slash, Token::Int(128)]).parse().as_ref()
        );
        // Parentheses precedence
        assert_eq!(
            Ok(&n_plus_n_minus_n),
            Parser::new(vec![Token::LParen, Token::Int(512), Token::Plus, Token::Int(256),
                             Token::Minus, Token::Int(128), Token::RParen]).parse().as_ref()
        );
        assert_eq!(
            Ok(&n_plus_n_slash_n),
            Parser::new(vec![Token::LParen, Token::Int(512), Token::Plus, Token::Int(256),
                             Token::Slash, Token::Int(128), Token::RParen]).parse().as_ref()
        );
        assert_eq!(
            Ok(Expr::Dyadic {
                operator: Operator::Plus,
                left: Box::new(Expr::Int(512)),
                right: Box::new(Expr::Dyadic {
                    operator: Operator::Plus,
                    left: Box::new(Expr::Int(256)),
                    right: Box::new(Expr::Ident("one_two_eight".into()))
                }),
            }),
            Parser::new(vec![Token::Int(512), Token::Plus, Token::LParen,
                             Token::Int(256), Token::Plus, Token::Ident("one_two_eight".into()),
                             Token::RParen]).parse()
        );
    }

}