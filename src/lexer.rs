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
    while first.eq(&' ') || first.eq(&'\n') {
        first = iter.next()?;
    }

    let token = match first {
        '+' => Token::Plus,
        '-' => Token::Minus,
        '*' => Token::Star,
        '/' => Token::Slash,
        '=' => Token::Equal,
        '(' => Token::LParen,
        ')' => Token::RParen,
        '{' => Token::LSquirly,
        '}' => Token::RSquirly,
        ';' => Token::EndLine,
        ':' => Token::Colon,
        ',' => Token::Comma,
        '0'..='9' => number_token(iter, first)?,
        'a'..='z' |
        'A'..='Z' => text_token(iter, first)?,
        _ => return None
    };
    Some(token)
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

fn text_token(iter: &mut Chars, first: char) -> Option<Token> {
    // first char will be alphabetic per `next_token` implementation
    let mut word: String = String::from(first);
    while let Some(next_char) = iter.clone().next() {
        if next_char.is_alphanumeric() || next_char == '_' {
            word.push(next_char);
            iter.next();
        } else {
            break;
        }
    }
    match word.as_str() {
        "let" => Some(Token::Let),
        "fn" => Some(Token::Function),
        _ => Some(Token::Ident(word))
    }
}

#[cfg(test)]
mod tests {
    use crate::tokens::*;
    use crate::lexer::tokenise;

    #[test]
    fn simple() {
        assert_eq!(Vec::<Token>::new(), tokenise("".to_string()));
        assert_eq!(
            vec![Token::Int(12)],
            tokenise(String::from("12"))
        );
        assert_eq!(
            vec![Token::Minus, Token::Int(1024), Token::Plus, Token::Int(2)],
            tokenise(String::from("-1024 + 2"))
        );
        assert_eq!(
            vec![Token::Minus, Token::Equal, Token::Plus],
            tokenise(String::from("-=+"))
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

    #[test]
    fn has_ident() {
        assert_eq!(
            vec![Token::Ident("on_its_own".to_string())],
            tokenise("on_its_own".to_string())
        );
        assert_eq!(
            vec![Token::Ident("a".to_string()), Token::Equal, Token::Int(0)],
            tokenise("a = 0".to_string())
        );
        assert_eq!(
            vec![Token::Ident("a".to_string()), Token::Equal,
                 Token::Minus, Token::Int(5), Token::Plus, Token::Minus, Token::Int(2)],
            tokenise("a = -5 + -2".to_string())
        );
    }

    #[test]
    fn def_func() {
        let input1 = r#"
        let add = fn(a: int, b: int):int {
            a + b
        };

        let result:int = add(16, 8);
        "#;

        assert_eq!(
            vec![Token::Let, Token::Ident("add".into()), Token::Equal, Token::Function,
                Token::LParen,
                    Token::Ident("a".into()), Token::Colon, Token::Ident("int".into()),
                    Token::Comma,
                    Token::Ident("b".into()), Token::Colon, Token::Ident("int".into()),
                Token::RParen, Token::Colon, Token::Ident("int".into()),
                Token::LSquirly,
                    Token::Ident("a".into()), Token::Plus, Token::Ident("b".into()),
                Token::RSquirly, Token::EndLine,

                Token::Let, Token::Ident("result".into()), Token::Colon, Token::Ident("int".into()),
                Token::Equal, Token::Ident("add".into()),
                Token::LParen,
                    Token::Int(16), Token::Comma, Token::Int(8),
                Token::RParen, Token::EndLine],
            tokenise(input1.into())
        );

        let input2 = r#"
        let half = fn(n:int):int {
            n/2
        };

        let eight:int = 8;

        let oneAndHalf:int = half(eight) * 3;
        "#;

        assert_eq!(
            vec![Token::Let, Token::Ident("half".into()), Token::Equal, Token::Function,
                Token::LParen,
                    Token::Ident("n".into()), Token::Colon, Token::Ident("int".into()),
                Token::RParen, Token::Colon, Token::Ident("int".into()),
                Token::LSquirly,
                    Token::Ident("n".into()), Token::Slash, Token::Int(2),
                Token::RSquirly, Token::EndLine,

                Token::Let, Token::Ident("eight".into()), Token::Colon, Token::Ident("int".into()),
                Token::Equal, Token::Int(8), Token::EndLine,

                Token::Let, Token::Ident("oneAndHalf".into()), Token::Colon, Token::Ident("int".into()),
                Token::Equal, Token::Ident("half".into()),
                Token::LParen, Token::Ident("eight".into()), Token::RParen,
                Token::Star, Token::Int(3), Token::EndLine],
            tokenise(input2.into())
        );

        let input3 = r#"
        let incr = fn(n:int):int {
            n + 1;
            n
        };
        let result:int = incr(0);
        "#;

        assert_eq!(
            vec![Token::Let, Token::Ident("incr".into()), Token::Equal, Token::Function,
                Token::LParen,
                    Token::Ident("n".into()), Token::Colon, Token::Ident("int".into()),
                Token::RParen, Token::Colon, Token::Ident("int".into()),
                Token::LSquirly,
                    Token::Ident("n".into()), Token::Plus, Token::Int(1), Token::EndLine,
                    Token::Ident("n".into()),
                Token::RSquirly, Token::EndLine,

                Token::Let, Token::Ident("result".into()), Token::Colon, Token::Ident("int".into()),
                Token::Equal, Token::Ident("incr".into()),
                Token::LParen, Token::Int(0), Token::RParen, Token::EndLine],
            tokenise(input3.into())
        );
    }
}