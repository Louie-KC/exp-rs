use crate::tokens::Token;
use crate::ast::*;

pub fn parse(tokens: Vec<Token>) -> Result<Expr, String> {
    eval(tokens.into_iter().rev().collect())
}

fn eval(mut tokens: Vec<Token>) -> Result<Expr, String> {
    println!("parser eval called with {:?}", tokens); 
    match tokens.len() {
        0 => Err("No tokens to evaluate".to_string()),
        1 => eval_one(tokens.remove(0)),
        _ => eval_right_known(tokens.remove(0), tokens)
    }
}

fn eval_one(token: Token) -> Result<Expr, String> {
    match token {
        Token::Int(n) => Ok(Expr::Int(n)),
        _ => {
            println!("eval_one: bad token {:?}", token);
            Err("evaluate_one: invalid single token".to_string())
        }
    }
}

fn eval_right_known(right: Token, mut tokens: Vec<Token>) -> Result<Expr, String> {
    match tokens.len() {
        1 => eval_monadic(right, tokens.remove(0)),
        _ => eval_dyadic(right, tokens)
    }
}

fn eval_monadic(right: Token, op: Token) -> Result<Expr, String> {
    // println!("eval_monadic: {:?}, {:?}", right, op);
    match (right, op) {
        (Token::Int(n), Token::Plus)  => Ok(Expr::Int(n)),
        (Token::Int(n), Token::Minus) => Ok(Expr::Int(-n)),
        (_, Token::Plus)  => todo!("monadic plus"),
        (_, Token::Minus) => todo!("monadic minus"),
        (_, _) => Err("Invalid monadic operation".to_string())
    }
}

fn eval_dyadic(right: Token, mut tokens: Vec<Token>) -> Result<Expr, String> {
    let right_e = match tokens.get(1) {
        Some(Token::Plus) | Some(Token::Minus) => {
            // Two operators in a row = monadic op first
            // Remove first operator from tokens list as its used in the monadic evaluation
            Box::new(eval_monadic(right, tokens.remove(0))?)
        },
        _ => Box::new(eval_one(right)?),
    };
    let op = tokens.remove(0);
    let next = Box::new(eval(tokens)?);

    match op {
        Token::Plus => Ok(Expr::Dyadic {
            operator: Operator::Plus,
            left: next,
            right: right_e
        }),
        Token::Minus => Ok(Expr::Dyadic {
            operator: Operator::Minus,
            left: next,
            right: right_e
        }),
        _ => Err("Invalid dyadic operation".to_string())
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::parse;
    use crate::tokens::Token;
    use crate::ast::*;

    #[test]
    fn simple() {
        assert_eq!(
            Err("No tokens to evaluate".to_string()),
            parse(vec![])
        );
        assert_eq!(
            Ok(Expr::Int(0)),
            parse(vec![Token::Int(0)])
        );
        assert_eq!(
            Ok(Expr::Int(5)),
            parse(vec![Token::Int(5)])
        );
    }

    #[test]        
    fn simple_monadic() {
        assert_eq!(
            Err("Invalid monadic operation".to_string()),
            parse(vec![Token::Int(0), Token::Int(0)])
        );
        assert_eq!(
            Ok(Expr::Int(256)),
            parse(vec![Token::Plus, Token::Int(256)])
        );

        assert_eq!(
            Ok(Expr::Int(-15)),
            parse(vec![Token::Minus, Token::Int(15)])
        );
    }
    
    #[test]
    fn simple_dyadic() {
        assert_eq!(
            Err("Invalid dyadic operation".to_string()),
            parse(vec![Token::Int(0), Token::Int(0), Token::Int(0)])
        );
        assert_eq!(
            Ok(Expr::Dyadic {
                operator: Operator::Plus,
                left: Box::new(Expr::Int(5)),
                right: Box::new(Expr::Int(6))
            }),
            parse(vec![Token::Int(5), Token::Plus, Token::Int(6)])
        );
        assert_eq!(
            Ok(Expr::Dyadic {
                operator: Operator::Minus,
                left: Box::new(Expr::Int(20)),
                right: Box::new(Expr::Int(5))
            }),
            parse(vec![Token::Int(20), Token::Minus, Token::Int(5)])
        );
        assert_eq!(
            Ok(Expr::Dyadic {
                operator: Operator::Minus,
                left: Box::new(Expr::Int(256)),
                right: Box::new(Expr::Int(-256))
            }),
            parse(vec![Token::Int(256), Token::Minus, Token::Minus, Token::Int(256)])

        )
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
            parse(vec![Token::Int(1), Token::Plus, Token::Int(2), Token::Plus, Token::Int(3)])
        );

        assert_eq!(
            Ok(Expr::Dyadic {
                operator: Operator::Plus,
                left: Box::new(Expr::Dyadic {
                    operator: Operator::Minus,
                    left: Box::new(Expr::Dyadic {
                        operator: Operator::Plus,
                        left: Box::new(Expr::Int(1)),
                        right: Box::new(Expr::Int(2))
                    }),
                    right: Box::new(Expr::Int(4))
                }),
                right: Box::new(Expr::Int(-8))
            }),
            parse(vec![Token::Int(1), Token::Plus,
                       Token::Int(2), Token::Minus,
                       Token::Int(4), Token::Plus,
                       Token::Minus, Token::Int(8)])
        );

        assert_eq!(
            Ok(Expr::Dyadic {
                operator: Operator::Plus,
                left: Box::new(Expr::Dyadic {
                    operator: Operator::Plus,
                    left: Box::new(Expr::Int(10)),
                    right: Box::new(Expr::Int(-20))
                }),
                right: Box::new(Expr::Int(-30))
            }),
            parse(vec![Token::Int(10), Token::Plus, Token::Minus, Token::Int(20),
                       Token::Plus, Token::Minus, Token::Int(30)])
        );
    }
}