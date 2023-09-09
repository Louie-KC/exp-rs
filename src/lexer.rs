use std::str::Chars;
use crate::tokens::Token;
use crate::tokens::TokenKind;

pub fn tokenise(source: String) -> Vec<Token> {
    let mut tokens: Vec<Token> = Vec::new();
    let mut iter: Chars;
    for (line_number, line) in source.split("\n").into_iter().enumerate() {
        iter = line.chars();
        while let Some(token) = next_token(&mut iter, line_number as u32) {
            tokens.push(token);
        }
    }
    tokens.push(Token {line_num: 0, kind: TokenKind::EOF});
    tokens
}

fn next_token(iter: &mut Chars, line_number: u32) -> Option<Token> {
    let mut first = iter.next()?;
    while first.is_whitespace() || is_newline(&first) {
        first = iter.next()?;
    }
    while is_comment(&first, iter) {
        first = char_after_skipping_comment(iter);
    }

    let token_kind = match first {
        '+' => TokenKind::Plus,
        '-' => TokenKind::Minus,
        '*' => TokenKind::Star,
        '/' => TokenKind::Slash,
        '%' => TokenKind::Percent,
        '!' => TokenKind::Negate,
        '(' => TokenKind::LParen,
        ')' => TokenKind::RParen,
        '{' => TokenKind::LSquirly,
        '}' => TokenKind::RSquirly,
        ';' => TokenKind::EndLine,
        ':' => TokenKind::Colon,
        ',' => TokenKind::Comma,
        '|' | '&' => logical_operator(iter, first),
        '=' | '<' | '>' => equals_comparator_token(iter, first),
        '0'..='9' => number_token(iter, first),
        'a'..='z' |
        'A'..='Z' => text_token(iter, first),
        _ => return None
    };
    Some(Token {line_num: line_number, kind: token_kind})
}

fn is_comment(first: &char, iter: &mut Chars) -> bool {
    // lookahead one
    match (first, iter.clone().next().unwrap_or(' ')) {
        ('/', '/') => true,
        _ => false
    }
}

fn is_newline(first: &char) -> bool {
    match *first as u32 {
        10 | 13 => true,
        _       => false
    }
}

fn char_after_skipping_comment(iter: &mut Chars) -> char {
    let mut result = iter.next().unwrap();
    while !result.is_control() {
        result = iter.next().unwrap_or(13 as char);
    }
    result = iter.next().unwrap_or(3 as char);
    while result.is_whitespace(){
        result = iter.next().unwrap_or(3 as char);
    }
    result

}

fn logical_operator(iter: &mut Chars, first: char) -> TokenKind {
    let op = match (first, iter.clone().next()) {
        ('&', Some('&')) => TokenKind::And,
        ('|', Some('|')) => TokenKind::Or,
        _ => panic!("Missing second {} in logical operator", first)
    };
    iter.next();
    op
}

fn equals_comparator_token(iter: &mut Chars, first: char) -> TokenKind {
    let next = iter.clone().next().unwrap_or(' ');
    if next == '=' {
        iter.next();
    }
    match (first, next) {
        ('<', '=') => TokenKind::LessEquals,
        ('<', _)   => TokenKind::LessThan,
        ('>', '=') => TokenKind::GreaterEquals,
        ('>', _)   => TokenKind::GreaterThan,
        ('=', '=') => TokenKind::EqualTo,
        (_, _)     => TokenKind::Equal
    }
}

fn number_token(iter: &mut Chars, first: char) -> TokenKind {
    let mut number = first.to_digit(10).unwrap() as i32;

    while let Some(next_char) = iter.clone().next() {
        if let Some(digit) = next_char.to_digit(10) {
            number = number * 10 + digit as i32;
            iter.next();
        } else {
            break;
        }
    }

    TokenKind::Int(number)
}

fn text_token(iter: &mut Chars, first: char) -> TokenKind {
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
        "print" => TokenKind::Print,
        "let"   => TokenKind::Let,
        "var"   => TokenKind::Var,
        "fn"    => TokenKind::Function,
        "if"    => TokenKind::If,
        "else"  => TokenKind::Else,
        "true"  => TokenKind::Boolean(true),
        "false" => TokenKind::Boolean(false),
        "while" => TokenKind::While,
        "for"   => TokenKind::For,
        _       => TokenKind::Ident(word)
    }
}

#[cfg(test)]
mod tests {
    macro_rules! token {
        ($line_number:expr, $token_kind:expr) => {
            Token{line_num: $line_number, kind: $token_kind}
        }
    }

    use crate::tokens::Token;
    use crate::tokens::TokenKind::*;
    use crate::lexer::tokenise;

    #[test]
    fn simple() {
        assert_eq!(vec![token!(0, EOF)], tokenise("".into()));
        assert_eq!(
            vec![token!(0, Int(12)), token!(0, EOF)],
            tokenise("12".into())
        );
        assert_eq!(
            vec![token!(0, Minus), token!(0, Int(1024)), token!(0, Plus),
                token!(0, Int(2)), token!(0, EOF)],
            tokenise("-1024 + 2".into())
        );
        assert_eq!(
            vec![token!(0, Equal), token!(0, Minus), token!(0, Plus),
                token!(0, Slash), token!(0, Percent), token!(0, EOF)],
            tokenise("=-+/%".into())
        );
        assert_eq!(
            vec![token!(0, Minus), token!(0, Plus), token!(0, Minus), token!(0, Minus),
                token!(0, Minus), token!(0, Plus), token!(0, Plus), token!(0, Plus), token!(0, EOF)],
            tokenise("   -+--  -++    +".into())
        );
        assert_eq!(
            vec![token!(0, EqualTo), token!(0, Equal), token!(0, LessThan),
                token!(0, LessEquals), token!(0, GreaterThan), token!(0, GreaterEquals),
                token!(0, Negate), token!(0, Equal), token!(0, EOF)],
            tokenise("== = < <= > >= !=".into())
        );
        assert_eq!(
            vec![token!(0, EqualTo), token!(0, Equal), token!(0, LessThan),
                token!(0, LessEquals), token!(0, GreaterThan), token!(0, GreaterEquals),
                token!(0, Negate), token!(0, Equal), token!(0, EOF)],
            tokenise("===<<=>>=!=".into())
        );
        assert_eq!(vec![token!(0, Or), token!(0, And), token!(0, EOF)], tokenise("|| &&".into()));
    }

    #[test]
    fn has_ident() {
        assert_eq!(vec![token!(0, Ident("on_its_own".into())), token!(0, EndLine), token!(0, EOF)],
            tokenise("on_its_own;".into())
        );
        assert_eq!(vec![token!(0, Ident("a".into())), token!(0, Equal), token!(0, Int(0)),
            token!(0, EndLine), token!(0, EOF)],
            tokenise("a = 0;".to_string())
        );
        assert_eq!(vec![token!(0, Var), token!(0, Ident("a".into())), token!(0, Equal),
            token!(0, Minus), token!(0, Int(5)), token!(0, Plus), token!(0, Minus),
            token!(0, Int(2)), token!(0, EndLine), token!(0, EOF)],
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
            vec![token!(1, If), token!(1, LParen), token!(1, Ident("ab".into())),
                token!(1, RParen), token!(1, LSquirly), token!(2, Int(1)), token!(2, EndLine),
                token!(3, RSquirly), token!(4, Int(0)), token!(4, EndLine), token!(0, EOF)],
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
            vec![token!(1, If), token!(1, LParen), token!(1, Ident("a".into())),
                token!(1, GreaterThan), token!(1, Int(5)), token!(1, RParen),
                token!(1, LSquirly), token!(2, Int(1)), token!(2, EndLine),
                token!(3, RSquirly), token!(3, Else), token!(3, LSquirly),
                token!(4, Int(2)), token!(4, EndLine),
                token!(5, RSquirly), token!(0, EOF)],
            tokenise(input2)
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
            vec![token!(1, Let), token!(1, Ident("add".into())), token!(1, Equal),
                token!(1, Function), token!(1, LParen),
                    token!(1, Ident("a".into())), token!(1, Colon), token!(1, Ident("int".into())), token!(1, Comma),
                    token!(1, Ident("b".into())), token!(1, Colon), token!(1, Ident("int".into())),
                token!(1, RParen), token!(1, Colon), token!(1, Ident("int".into())), token!(1, LSquirly),
                    token!(2, Ident("a".into())), token!(2, Plus), token!(2, Ident("b".into())),
                token!(3, RSquirly), token!(3, EndLine),
                token!(5, Let), token!(5, Ident("result".into())), token!(5, Colon),
                    token!(5, Ident("int".into())), token!(5, Equal), token!(5, Ident("add".into())),
                    token!(5, LParen), token!(5, Int(16)), token!(5, Comma), token!(5, Int(8)), token!(5, RParen),
                    token!(5, EndLine),
            token!(0, EOF)],
            tokenise(input1.into())
        );

        let input2 = r#"
        let half = fn(n:int):int {
            n/2
        };

        let eight:int = 8;

        let oneAndHalf:int = half(eight) * 3;
        "#;

        assert_eq!(vec![
            token!(1, Let), token!(1, Ident("half".into())), token!(1, Equal), token!(1, Function),
                token!(1, LParen), token!(1, Ident("n".into())), token!(1, Colon),
                token!(1, Ident("int".into())), token!(1, RParen),
                token!(1, Colon), token!(1, Ident("int".into())),
            token!(1, LSquirly),
                token!(2, Ident("n".into())), token!(2, Slash), token!(2, Int(2)),
            token!(3, RSquirly), token!(3, EndLine),
            token!(5, Let), token!(5, Ident("eight".into())), token!(5, Colon),
                token!(5, Ident("int".into())), token!(5, Equal), token!(5, Int(8)), token!(5, EndLine),
            token!(7, Let), token!(7, Ident("oneAndHalf".into())), token!(7, Colon),
                token!(7, Ident("int".into())),token!(7, Equal),
                token!(7, Ident("half".into())), token!(7, LParen), token!(7, Ident("eight".into())),
                token!(7, RParen), token!(7, Star), token!(7, Int(3)), token!(7, EndLine), token!(0, EOF)
            ], tokenise(input2.into())
        );

        let input3 = r#"
        let incr = fn(n:int):int {
            n + 1;
            n
        };
        let result:int = incr(0);
        "#;

        assert_eq!(vec![
                token!(1, Let), token!(1, Ident("incr".into())), token!(1, Equal), token!(1, Function),
                token!(1, LParen), token!(1, Ident("n".into())), token!(1, Colon),
                token!(1, Ident("int".into())), token!(1, RParen),
                token!(1, Colon), token!(1, Ident("int".into())),
            token!(1, LSquirly),
                token!(2, Ident("n".into())), token!(2, Plus), token!(2, Int(1)), token!(2, EndLine),
                token!(3, Ident("n".into())),
            token!(4, RSquirly), token!(4, EndLine),
            token!(5, Let), token!(5, Ident("result".into())), token!(5, Colon),
                token!(5, Ident("int".into())), token!(5, Equal), token!(5, Ident("incr".into())),
                token!(5, LParen), token!(5, Int(0)), token!(5, RParen), token!(5, EndLine),
            token!(0, EOF)
            ], tokenise(input3.into())
        );
    }

    #[test]
    fn comments() {
        let input1 = r#"
        // All Comments
        // No actual code
        // :)"#;
        assert_eq!(vec![token!(0, EOF)], tokenise(input1.into()));

        let input2 = r#"
        // e
        print(5);  // prints 5
        // in between statements
        print(6);  // then prints 6
        // should be EOF here
        "#;
        assert_eq!(
            vec![token!(2, Print), token!(2, LParen), token!(2, Int(5)), token!(2, RParen), token!(2, EndLine),
                token!(4, Print), token!(4, LParen), token!(4, Int(6)), token!(4, RParen), token!(4, EndLine),
                token!(0, EOF)],
            tokenise(input2.into())
        );
    }
}