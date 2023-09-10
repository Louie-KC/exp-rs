use crate::tokens::TokenKind;

#[derive(Debug, PartialEq)]
pub struct ParseError {
    pub line_num: u32,
    pub kind: ParseErrorKind
}

#[derive(Debug, PartialEq)]
pub enum ParseErrorKind {
    IllegalTokenSequence,
    MissingOpeningParenthesis,
    MissingClosingParenthesis,
    MissingOpeningCurlyBrace,
    MissingClosingCurlyBrace,
    MissingSemicolon,
    MissingFunctionName,
    MissingIdentifier,
    ExcessParameters,
    MissingBlock,
    UnexpectedSymbol(TokenKind, TokenKind),
    UnrecognisedSymbol(String)
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ParseErrorKind as PEK;
        let err_msg: String = match &self.kind {
            PEK::IllegalTokenSequence  => "Illegal token sequence (e.g. ident ident, int int)".into(),
            PEK::MissingOpeningParenthesis => "Missing left/opening parenthesis".into(),
            PEK::MissingClosingParenthesis => "Missing right/closing parenthesis".into(),
            PEK::MissingOpeningCurlyBrace  => "Missing left/opening curly brace".into(),
            PEK::MissingClosingCurlyBrace  => "Missing right/closing curly brace".into(),
            PEK::MissingSemicolon      => "Missing semi-colon ; to end statement".into(),
            PEK::ExcessParameters      => "Number of parameters exceeds 255".into(),
            PEK::MissingBlock          => "Missing block/body for if/else/while/for".into(),
            PEK::UnrecognisedSymbol(s) => format!("Unrecognised symbol: \"{}\"", s),
            PEK::UnexpectedSymbol(expected, actual) => {
                format!("Expected token: {:?}\nFound token: {:?}", expected, actual)
            },
            PEK::MissingFunctionName => "Function name is not specified on declaration".into(),
            PEK::MissingIdentifier => "Parameter or variable name is not specified".into(),
        };
        write!(f, "Parse Error: {:?} on line: {}\nDescription: {}", self.kind, self.line_num, err_msg)
    }
}