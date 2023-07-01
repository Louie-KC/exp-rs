use std::str::Chars;
use crate::tokens::Token;

pub fn tokenise(source: String) -> Vec<Token> {
    let mut tokens: Vec<Token> = Vec::new();
    let mut iter: Chars = source.chars();

    while let Some(token) = next_token(&mut iter) {
        tokens.push(token);
    }
    tokens.push(Token::EOF);
    tokens
}

fn next_token(iter: &mut Chars) -> Option<Token> {
    let mut first = iter.next()?;
    while first.eq(&' ') || first.eq(&'\n') {
        first = iter.next()?;
    }
    while is_comment(&first, iter) {
        first = char_after_skipping_comment(iter);
    }

    let token = match first {
        '+' => Token::Plus,
        '-' => Token::Minus,
        '*' => Token::Star,
        '/' => Token::Slash,
        '%' => Token::Percent,
        '!' => Token::Negate,
        '(' => Token::LParen,
        ')' => Token::RParen,
        '{' => Token::LSquirly,
        '}' => Token::RSquirly,
        ';' => Token::EndLine,
        ':' => Token::Colon,
        ',' => Token::Comma,
        '|' | '&' => logical_operator(iter, first),
        '=' | '<' | '>' => equals_comparator_token(iter, first),
        '0'..='9' => number_token(iter, first),
        'a'..='z' |
        'A'..='Z' => text_token(iter, first),
        _ => return None
    };
    Some(token)
}

fn is_comment(first: &char, iter: &mut Chars) -> bool {
    // lookahead one
    match (first, iter.clone().next().unwrap_or(' ')) {
        ('/', '/') => true,
        _ => false
    }
}

fn char_after_skipping_comment(iter: &mut Chars) -> char {
    let mut result = iter.next().unwrap();
    while !result.is_control() {
        result = iter.next().unwrap();
    }
    result = iter.next().unwrap();
    while result.is_whitespace(){
        result = iter.next().unwrap_or(3 as char);
    }
    result

}

fn logical_operator(iter: &mut Chars, first: char) -> Token {
    let op = match (first, iter.clone().next()) {
        ('&', Some('&')) => Token::And,
        ('|', Some('|')) => Token::Or,
        _ => panic!("Missing second {} in logical operator", first)
    };
    iter.next();
    op
}

fn equals_comparator_token(iter: &mut Chars, first: char) -> Token {
    let next = iter.clone().next().unwrap_or(' ');
    if next == '=' {
        iter.next();
    }
    match (first, next) {
        ('<', '=') => Token::LessEquals,
        ('<', _)   => Token::LessThan,
        ('>', '=') => Token::GreaterEquals,
        ('>', _)   => Token::GreaterThan,
        ('=', '=') => Token::EqualTo,
        (_, _)     => Token::Equal
    }
}

fn number_token(iter: &mut Chars, first: char) -> Token {
    let mut number = first.to_digit(10).unwrap() as i32;

    while let Some(next_char) = iter.clone().next() {
        if let Some(digit) = next_char.to_digit(10) {
            number = number * 10 + digit as i32;
            iter.next();
        } else {
            break;
        }
    }

    Token::Int(number)
}

fn text_token(iter: &mut Chars, first: char) -> Token {
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
        "print" => Token::Print,
        "let"   => Token::Let,
        "var"   => Token::Var,
        "fn"    => Token::Function,
        "if"    => Token::If,
        "else"  => Token::Else,
        "true"  => Token::Boolean(true),
        "false" => Token::Boolean(false),
        "while" => Token::While,
        _       => Token::Ident(word)
    }
}

#[cfg(test)]
mod tests {
    use crate::tokens::Token::*;
    use crate::lexer::tokenise;

    #[test]
    fn simple() {
        assert_eq!(vec![EOF], tokenise("".to_string()));
        assert_eq!(
            vec![Int(12), EOF],
            tokenise(String::from("12"))
        );
        assert_eq!(
            vec![Minus, Int(1024), Plus, Int(2), EOF],
            tokenise(String::from("-1024 + 2"))
        );
        assert_eq!(
            vec![Equal, Minus, Plus, Slash, Percent, EOF],
            tokenise(String::from("=-+/%"))
        );
        assert_eq!(
            vec![
                Minus, Plus, Minus, Minus, Minus, Plus, Plus, Plus, EOF
            ],
            tokenise(String::from("   -+--  -++    +"))
        );
        assert_eq!(
            vec![EqualTo, Equal, LessThan, LessEquals, GreaterThan, GreaterEquals, Negate, Equal, EOF],
            tokenise("== = < <= > >= !=".into())
        );
        assert_eq!(
            vec![EqualTo, Equal, LessThan, LessEquals, GreaterThan, GreaterEquals, Negate, Equal, EOF],
            tokenise("===<<=>>=!=".into())
        );
        assert_eq!(
            vec![Or, And, EOF],
            tokenise("|| &&".into())
        )
    }

    #[test]
    fn has_ident() {
        assert_eq!(
            vec![Ident("on_its_own".to_string()), EndLine, EOF],
            tokenise("on_its_own;".to_string())
        );
        assert_eq!(
            vec![Ident("a".to_string()), Equal, Int(0), EndLine, EOF],
            tokenise("a = 0;".to_string())
        );
        assert_eq!(
            vec![Var, Ident("a".to_string()), Equal,
                 Minus, Int(5), Plus, Minus, Int(2), EndLine, EOF],
            tokenise("var a = -5 + -2;".to_string())
        );
    }

    #[test]
    fn branch() {
        let input1 = r#"
            if (ab) {
                1;
            }
            0;
        "#.to_string();
        assert_eq!(
            vec![If, LParen, Ident("ab".into()), RParen, LSquirly, Int(1), EndLine, RSquirly,
                 Int(0), EndLine, EOF],
            tokenise(input1)
        );

        let input2 = r#"
            if (a > 5) {
                1;
            } else {
                2;
            }
        "#.to_string();
        assert_eq!(
            vec![If, LParen, Ident("a".into()), GreaterThan, Int(5), RParen,
                    LSquirly, Int(1), EndLine, RSquirly,
                Else, LSquirly, Int(2), EndLine, RSquirly, EOF],
        tokenise(input2));
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
            vec![Let, Ident("add".into()), Equal, Function,
                LParen,
                    Ident("a".into()), Colon, Ident("int".into()), Comma,
                    Ident("b".into()), Colon, Ident("int".into()),
                RParen, Colon, Ident("int".into()),
                LSquirly, Ident("a".into()), Plus, Ident("b".into()), RSquirly, EndLine,

                Let, Ident("result".into()), Colon, Ident("int".into()),
                Equal, Ident("add".into()), LParen, Int(16), Comma, Int(8), RParen, EndLine, EOF],
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
            vec![Let, Ident("half".into()), Equal, Function,
                LParen, Ident("n".into()), Colon, Ident("int".into()), RParen, Colon,
                Ident("int".into()), LSquirly, Ident("n".into()), Slash, Int(2), RSquirly, EndLine,

                Let, Ident("eight".into()), Colon, Ident("int".into()), Equal, Int(8), EndLine,

                Let, Ident("oneAndHalf".into()), Colon, Ident("int".into()), Equal, Ident("half".into()),
                LParen, Ident("eight".into()), RParen, Star, Int(3), EndLine, EOF],
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
            vec![Let, Ident("incr".into()), Equal, Function,
                LParen, Ident("n".into()), Colon, Ident("int".into()), RParen, Colon, Ident("int".into()),
                LSquirly, Ident("n".into()), Plus, Int(1), EndLine, Ident("n".into()), RSquirly, EndLine,

                Let, Ident("result".into()), Colon, Ident("int".into()), Equal, Ident("incr".into()),
                LParen, Int(0), RParen, EndLine, EOF],
            tokenise(input3.into())
        );
    }

    #[test]
    fn comments() {
        let input1 = r#"
        // All Comments
        // No actual code
        // :)
        "#;
        assert_eq!(vec![EOF], tokenise(input1.into()));

        let input2 = r#"
        // e
        print(5);  // prints 5
        // in between statements
        print(6);  // then prints 6
        // should be EOF here
        "#;
        assert_eq!(vec![Print, LParen, Int(5), RParen, EndLine,
                        Print, LParen, Int(6), RParen, EndLine,
                        EOF], tokenise(input2.into()))
    }
}