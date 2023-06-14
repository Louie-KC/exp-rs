use std::str::Chars;
use crate::tokens::Token;

pub fn tokenise(source: String) -> Vec<Token> {
    let mut tokens: Vec<Token> = Vec::new();
    let mut iter: Chars = source.chars();

    while let Some(token) = next_token(&mut iter) {
        tokens.push(token);
    }

    tokens
}

fn next_token(iter: &mut Chars) -> Option<Token> {
    let mut first = iter.next()?;
    while first.eq(&' ') {
        first = iter.next()?;
    }

    match first {
        '+' => Some(Token::Plus),
        '-' => Some(Token::Minus),
        '0'..='9' => number_token(iter, first),
        'a'..='z' |
        'A'..='Z' => text_token(iter, first),
        _ => None
    }
}

fn number_token(iter: &mut Chars, first: char) -> Option<Token> {
    let mut number = first.to_digit(10)? as i32;

    while let Some(next_char) = iter.clone().next() {
        if let Some(digit) = next_char.to_digit(10) {
            number = number * 10 + digit as i32;
            iter.next();
        } else {
            break;
        }
    }

    Some(Token::Int(number))
}

fn text_token(iter: &Chars, first: char) -> Option<Token> {
    // todo!("lexer text_token - TODO")
    None
}

#[cfg(test)]
mod tests {
    use crate::tokens::*;
    use crate::lexer::tokenise;

    #[test]
    fn simple() {
        assert_eq!(
            vec![
                Token::Int(12),
            ],
            tokenise(String::from("12"))
        );
        assert_eq!(
            vec![
                Token::Minus,
                Token::Int(1024),
                Token::Plus,
                Token::Int(2)
            ],
            tokenise(String::from("-1024 + 2"))
        );
        assert_eq!(
            vec![
                Token::Minus,
                Token::Plus,
                Token::Minus,
                Token::Minus,
                Token::Minus,
                Token::Plus,
                Token::Plus,
                Token::Plus,
            ],
            tokenise(String::from("   -+--  -++    +"))
        );
    }
}