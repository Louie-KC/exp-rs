use crate::tokens::Token;
use crate::ast::*;

// NOTE: Currently no recursive parsing. Accepts up to a single dyadic expression.

pub fn parse(tokens: Vec<Token>) -> Result<Expr, String> {
    eval(tokens.into_iter().rev().collect())
}

fn eval(mut tokens: Vec<Token>) -> Result<Expr, String> {
    // println!("parser eval called with {:?}", tokens); 
    match tokens.len() {
        0 => Err("No tokens to evaluate".to_string()),
        1 => eval_one(tokens.remove(0)),
        _ => eval_right_known(tokens.remove(0), tokens)
    }
}

fn eval_one(token: Token) -> Result<Expr, String> {
    match token {
        Token::Int(n) => Ok(Expr::Int(n)),
        _ => Err("evaluate_one: invalid single token".to_string())
    }
}

fn eval_right_known(right: Token, mut tokens: Vec<Token>) -> Result<Expr, String> {
    match tokens.len() {
        1 => eval_monadic(right, tokens.remove(0)),
        _ => eval_dyadic(right, tokens.remove(0), tokens.remove(0))
    }
}

fn eval_monadic(right: Token, op: Token) -> Result<Expr, String> {
    match (right, op) {
        (Token::Int(n), Token::Plus)  => Ok(Expr::Int(n)),
        (Token::Int(n), Token::Minus) => Ok(Expr::Int(-n)),
        (_, Token::Plus)  => todo!("monadic plus"),
        (_, Token::Minus) => todo!("monadic minus"),
        (_, _) => Err("Invalid monadic operation".to_string())
    }
}

fn eval_dyadic(right: Token, op: Token, left: Token) -> Result<Expr, String> {
    let left_eval = Box::new(eval_one(left)?);
    let right_eval = Box::new(eval_one(right)?);

    match op {
        Token::Plus => Ok(Expr::Dyadic {
            operator: Operator::Plus,
            left: left_eval,
            right: right_eval
        }),
        Token::Minus => Ok(Expr::Dyadic {
            operator: Operator::Minus,
            left: left_eval,
            right: right_eval
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
    }
}