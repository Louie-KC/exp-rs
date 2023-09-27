use crate::{tokens::TokenKind, ast::{Operator, Expr}};

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

#[derive(Debug, PartialEq)]
pub struct InterpretError {
    // pub line_num: u32,
    pub kind: InterpretErrorKind
}

#[derive(Debug, PartialEq)]
pub enum InterpretErrorKind {
    NoEnvironments,
    VarNotInScope(String),
    VarRedeclaration(String),
    FnNotInScope(String),
    FnRedeclaration(String),
    FnBadArgs(String, usize, usize),
    InvalidOperation(Operator, Box<Expr>),
    PlaceHolderError
}

impl std::fmt::Display for InterpretError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use InterpretErrorKind as IEK;
        let err_msg: String = match &self.kind {
            IEK::NoEnvironments => "All environments have been cleared".into(),
            IEK::VarNotInScope(name) => format!("Variable '{}' not in scope", name),
            IEK::FnNotInScope(name)  => format!("Function '{}' not in scope", name),
            IEK::FnBadArgs(name, expected, actual) => {
                format!("Incorrect number of arguments were supplied with a '{}' function call\
                \nFunction '{}' expected {} args, but found {}", name, name, expected, actual)
            },
            IEK::VarRedeclaration(name) => {
                format!("The variable '{}' is declared more than once in the same scope", name)
            },
            IEK::FnRedeclaration(name)  => {
                format!("The function '{}' is declared more than once in the same scope", name)
            },
            IEK::InvalidOperation(operator, operand) => {  // UNTESTED
                format!("Cannot use the operator '{:?}' on '{:?}'.", operator, operand)
            },
            IEK::PlaceHolderError => "DEV TEST PLACEHOLDER - TO BE REMOVED".into(),  // UNTESTED
            
        };
        write!(f, "Interpet/Run error: {:?}\nDescription: {}", self.kind, err_msg)
    }
}