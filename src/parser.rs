use std::mem::{discriminant};

use crate::errors::ParseError;
use crate::tokens::{Token, TokenKind, self};
use crate::ast::*;
use crate::errors::ParseErrorKind as PEK;

type ParseResult<T> = Result<T, ParseError>;

macro_rules! parse_err {
    ($line_num:expr, $kind:expr) => {
        Err(ParseError {line_num: $line_num, kind: $kind})
    };
}

/*
 * Recursive Descent
 * 
 * rules:
 *    program -> statement*
 *  statement -> print
 *               | var
 *               | if
 *               | while
 *               | for
 *               | block
 *               | expression ";"
 *      print -> "print(" expression ");"
 *    fnDeclr -> "fn" function
 *   function -> Identifier "(" ( Identifier ( "," Identifier )* )? ")" block
 *    varDecl -> "var" Identifier ( "=" statement )? ";"
 *         if -> "if(" expression ")" block ( else block )?
 *      while -> "while" "(" expression ")" block
 *        for -> "for" "(" ( varDeclr | expression )? ";" expression? ";" expression ")" block
 *      block -> "{" statement* "}"
 * 
 * expression -> assignment
 * assignment -> logic_or | Identifier "=" expression
 *   logic_or -> logic_and ( "||" logic_and )*
 *  logic_and -> comparator ( "&&" comparator )*
 * comparator -> term ( ("==" | "!=" | "<" | "<=" | ">" | ">=" ) term )*
 *       term -> factor ( ( "+" | "-" ) factor)*
 *     factor -> unary ( ( "*" | "/" | "%" ) unary)*
 *      unary -> fn_call | "+" fn_call | "-" fn_call
 *    fn_call -> atom | atom "(" expression ( "," expression )* ")"
 *       atom -> "(" expression ")" | Integer | Identifier | Boolean
 */

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens: tokens, pos: 0 }
    }

    pub fn parse(&mut self) -> ParseResult<Vec<Stmt>> {
        // if !self.brackets_balanced() {
        //     return Err("Inbalanced brackets".into())
        //     // return ParseError
        // }
        self.brackets_balanced()?;
        self.illegal_token_seq_check()?;
        // if let Err(e) = self.illegal_token_seq_check() {
        //     return Err(e);
        // }

        // println!("Parsing: {:?}", self.tokens);
        let mut statements: Vec<Stmt> = Vec::new();

        while self.peek().kind != TokenKind::EOF {
            statements.push(self.statement()?);
        }

        Ok(statements)
    }

    fn statement(&mut self) -> ParseResult<Stmt> {
        // println!("statement: {:?}", self.peek());
        match self.peek().kind {
            TokenKind::Print => self.print_statement(),
            TokenKind::Function => self.function_declaration(),
            TokenKind::Var   => self.var_statement(),
            TokenKind::If    => self.if_statement(),
            TokenKind::While => self.while_statement(),
            TokenKind::For   => self.for_statement(),
            TokenKind::LSquirly => self.block_statement(),
            _            => self.expression_statement()
        }
    }

    // Rule: print -> "print(" expression ");"
    fn print_statement(&mut self) -> ParseResult<Stmt> {
        self.advance();  // Print
        let val: Expr = self.expression()?;
        self.consume(TokenKind::EndLine)?; //, "Expected \";\" to end print statement")?;
        Ok(Stmt::Print(val))
    }

    // Rule: fnDeclr -> "fn" function
    fn function_declaration(&mut self) -> ParseResult<Stmt> {
        self.advance();  // fn
        self.function()
    }

    // Rule: function -> Identifier "(" ( Identifier ( "," Identifier )* )? ")" block
    fn function(&mut self) -> ParseResult<Stmt> {
        let name = match self.atom() {
            Ok(Expr::Ident(n)) => n,
            _ => return parse_err!(self.peek().line_num, PEK::MissingFunctionName)
        };
        self.consume(TokenKind::LParen)?;  //, "Missing \"(\" on function declaration")?;

        let mut parameters: Vec<String> = vec![];
        while !self.matches_type(TokenKind::RParen) {
            parameters.push(match self.atom() {
                Ok(Expr::Ident(n)) => n,
                // _ => return Err("Expected named parameter for function declaration".into())
                _ => return parse_err!(self.peek().line_num, PEK::MissingIdentifier)
            });
            if self.matches_type(TokenKind::Comma) {
                self.advance();
            }
        }
        self.advance();  // RParen
        if parameters.len() > 255 {
            return parse_err!(self.peek().line_num, PEK::ExcessParameters)
        }
        if !self.matches_type(TokenKind::LSquirly) {
            return parse_err!(self.peek().line_num, PEK::MissingBlock)
        }
        let body: Stmt = self.block_statement()?;
        Ok(Stmt::FnDecl { name: name, parameters: parameters, body: Box::new(body) })
    }

    // Rule: varDecl -> "var" Identifier ( "=" statement )? ";"
    fn var_statement(&mut self) -> ParseResult<Stmt> {
        self.advance();  // var
        if !self.matches_type(TokenKind::Ident("".into())) {
            return parse_err!(self.peek().line_num, PEK::MissingIdentifier)
        }
        let ident: String = self.get_var_name();
        self.advance();
        let value = match self.peek().kind {
            TokenKind::Equal => {
                self.advance();  // =
                let deterministic: bool = self.matches_type(TokenKind::If);
                let v: Stmt = self.statement()?;
                match (deterministic, self.matches_type(TokenKind::EndLine)) {
                    (true, true)  => self.advance(),
                    (true, false) => return parse_err!(self.peek().line_num, PEK::MissingSemicolon),
                    _        => {}
                }
                Some(Box::new(v))
            },
            TokenKind::EndLine => {
                self.advance();  // ;
                None
            },
            _ => return parse_err!(self.peek().line_num, PEK::MissingSemicolon)
        };
        Ok(Stmt::VarDecl(ident.into(), value))
    }

    fn get_var_name(&self) -> String {
        // Assumes self.pos points at an Ident Token
        match &self.peek().kind {
            TokenKind::Ident(s) => s.to_string(),
            _ => panic!("bad get_var_name() call")
        }
    }

    // Rule: if -> "if(" expression ")" block ( else block )?
    fn if_statement(&mut self) -> ParseResult<Stmt> {
        self.advance();  // if
        self.consume(TokenKind::LParen)?;//, "Expected \"(\" after if")?;

        let cond: Expr = self.expression()?;
        
        self.consume(TokenKind::RParen)?;//, "Expected \")\" after if condition")?;

        let then: Stmt = self.statement()?;
        let els: Option<Box<Stmt>> = if self.matches_type(TokenKind::Else) {
            self.advance();  // else
            Some(Box::new(self.statement()?))
        } else {
            None
        };
        Ok(Stmt::If{ cond: cond, then: Box::new(then), els: els})
    }

    // Rule: while -> "while(" expression ")" block
    fn while_statement(&mut self) -> ParseResult<Stmt> {
        self.advance();  // while
        self.consume(TokenKind::LParen)?;

        let cond: Expr = self.expression()?;
        self.consume(TokenKind::RParen)?; //, "Expected \")\" after while condition")?;

        if self.peek().kind != TokenKind::LSquirly {
            return parse_err!(self.peek().line_num, PEK::MissingBlock)
        }
        let body = self.block_statement()?;
        Ok(Stmt::While { cond: cond, body: Box::new(body) })
    }

    // Rule: for -> "for" "(" ( varDeclr | expression )? ";" expression? ";" expression ")" block
    fn for_statement(&mut self) -> ParseResult<Stmt> {
        self.advance(); // for
        self.consume(TokenKind::LParen)?;

        let init = match self.peek().kind {
            TokenKind::EndLine => {
                self.advance();
                None
            },
            _ => Some(self.statement()?)
        };

        let cond = match self.peek().kind {
            TokenKind::EndLine => Expr::Boolean(true),
            _ => self.expression()?
        };
        self.consume(TokenKind::EndLine)?;

        let step = match self.peek().kind {
            TokenKind::RParen => None,
            _ => Some(self.expression()?)
        };

        // println!("{:?}, {:?}, {:?}", init, cond, step);
        self.consume(TokenKind::RParen)?; //, "Expected \"(\" to close loop condition")?;

        // assembling
        let mut loop_body = match self.block_statement() {
            Ok(Stmt::Block(stmts)) => stmts,
            _ => return parse_err!(self.peek().line_num, PEK::MissingBlock)
        };
        if let Some(step_expr) = step {
            loop_body.push(Stmt::Expr(step_expr));
        }
        let main_loop = Stmt::While {
            cond: cond,
            body: Box::new(Stmt::Block(loop_body))
        };
        match init {
            Some(init_stmt) => Ok(Stmt::Block(vec![init_stmt, main_loop])),
            None => Ok(main_loop)
        }
    }

    // Rule: block -> "{" statement* "}"
    fn block_statement(&mut self) -> ParseResult<Stmt> {
        self.advance();  // {
        let mut peek = self.peek();
        let mut statements:Vec<Stmt> = Vec::new();
        while peek.kind != TokenKind::EOF && peek.kind != TokenKind::RSquirly {
            statements.push(self.statement()?);
            peek = self.peek();
        }
        self.consume(TokenKind::RSquirly)?; // , "Expected \"}\" to close block")?;

        Ok(Stmt::Block(statements))
    }

    // Rule: expression -> comparator | term
    fn expression_statement(&mut self) -> ParseResult<Stmt> {
        let expr: Expr = self.expression()?;
        self.consume(TokenKind::EndLine)?; // , "Expected \";\" to end expression stmt")?;

        Ok(Stmt::Expr(expr))
    }
    
    fn eval_operator(&mut self) -> ParseResult<Operator> {
        let op = match self.peek().kind {
            TokenKind::Plus    => Operator::Plus,
            TokenKind::Minus   => Operator::Minus,
            TokenKind::Star    => Operator::Star,
            TokenKind::Slash   => Operator::Slash,
            TokenKind::Percent => Operator::Modulo,
            TokenKind::EqualTo => Operator::EqualTo,
            TokenKind::Negate  => {
                match self.peek_ahead() {
                    Some(Token {line_num: _, kind: TokenKind::Equal}) => {
                        self.advance();
                        Operator::NotEqualTo
                    },
                    _ => todo!("negation of variables")
                }
            },
            TokenKind::LessThan      => Operator::LessThan,
            TokenKind::LessEquals    => Operator::LessEquals,
            TokenKind::GreaterThan   => Operator::GreaterThan,
            TokenKind::GreaterEquals => Operator::GreaterEquals,
            TokenKind::Or      => Operator::LogicalOr,
            TokenKind::And     => Operator::LogicalAnd,
            _ => return parse_err!(self.peek().line_num, PEK::UnrecognisedSymbol(format!("{:?}", self.peek())))
        };
        self.advance();
        Ok(op)
    }

    fn peek(&self) -> &Token {
        &self.tokens.get(self.pos).unwrap()
    }

    fn peek_ahead(&self) -> Option<&Token> {
        self.tokens.get(self.pos + 1)
    }

    fn matches_type(&self, check: TokenKind) -> bool {
        discriminant(&check) == discriminant(&self.peek().kind)
    }

    fn matches_types(&self, types: Vec<TokenKind>) -> bool {
        if self.pos == self.tokens.len() {
            return false
        }
        for t in types.iter() {
            // Compare enum variant (ignoring any held data)
            if discriminant(t) == discriminant(&self.peek().kind) {
                return true
            }
        }
        false
    }

    fn illegal_token_seq_check(&self) -> ParseResult<()> {
        use tokens::TokenKind::*;
        for i in 1..self.tokens.len() {
            let prev = &self.tokens[i-1];
            let cur = &self.tokens[i];
            let mut bad_seq_flag = false;

            // Int and Ident token checking. Cannot seem to get working with pattern matching :(
            let int_discrim = discriminant(&TokenKind::Int(0));
            let ident_discrim = discriminant(&TokenKind::Ident("".into()));
            let prev_discrim = discriminant(&prev.kind);
            let cur_discrim = discriminant(&cur.kind);
            if prev_discrim == int_discrim && cur_discrim == int_discrim {
                // return Err("Two Int tokens in a row".into());
                bad_seq_flag = true;
            }
            if prev_discrim == ident_discrim && cur_discrim == ident_discrim {
                // return Err("Two Ident tokens in a row".into());
                bad_seq_flag = true;
            }
            if prev_discrim == int_discrim && cur_discrim == ident_discrim {
                // return Err("Int token immediately followed by Ident token".into())
                bad_seq_flag = true;
            }
            if prev_discrim == ident_discrim && cur_discrim == int_discrim {
                // return Err("Ident token immediately followed by Int token".into())
                bad_seq_flag = true;
            }

            // Remaining illegal token combinations
            match (&self.tokens[i-1].kind, &self.tokens[i].kind) {
                // (Function, Function) => return Err("Two Function tokens in a row".into()),
                // (Star, Star)   => return Err("Two Star tokens in a row".into()),
                // (Slash, Slash) => return Err("Two Slash tokens in a row".into()),
                // (Let, Let)     => return Err("Two Let tokens in a row".into()),
                (Function, Function)
                | (Star, Star)
                | (Slash, Slash)
                | (Let, Let) => bad_seq_flag = true,
                (_, _) => {}
            }
            if bad_seq_flag {
                return Err(ParseError{line_num: 0, kind: PEK::IllegalTokenSequence})
            }
        }
        Ok(())
    }

    fn brackets_balanced(&self) -> ParseResult<()> {
        let mut stack: Vec<(TokenKind, u32)> = Vec::new();

        for token in self.tokens.iter() {
            match token.kind {
                TokenKind::LParen   => stack.push((TokenKind::LParen, token.line_num)),
                TokenKind::LSquirly => stack.push((TokenKind::LSquirly, token.line_num)),
                TokenKind::RParen => {
                    match stack.pop() {
                        Some((TokenKind::LParen, _)) => {}
                        _ => return Err(ParseError {
                            line_num: token.line_num,
                            kind: PEK::MissingOpeningParenthesis
                        })
                    }
                }
                TokenKind::RSquirly => {
                    match stack.pop() {
                        Some((TokenKind::LSquirly, _)) => {}
                        _ => return Err(ParseError {
                            line_num: token.line_num,
                            kind: PEK::MissingOpeningCurlyBrace
                        })
                    }
                }
                _ => {}
            }
        }
        match stack.last() {
            Some((TokenKind::LParen, ln))   => parse_err!(*ln, PEK::MissingClosingParenthesis),
            Some((TokenKind::LSquirly, ln)) => parse_err!(*ln, PEK::MissingClosingCurlyBrace),
            _ => Ok(())
        }
        // match stack.is_empty() {
        //     true  => Ok(()),
        //     false => Err(ParseError { line_num: 0, kind: PEK::UnrecognisedSymbol("".into()) })
        // }
        // stack.is_empty()
    }

    fn advance(&mut self) -> () {
        self.pos += 1
    }

    /// Check the current token against the `expected` token, advance parser pos if matching.
    /// If Tokens don't match, return an appropriate Parse Error Result.
    fn consume(&mut self, expected: TokenKind) -> ParseResult<()> {
        if !self.matches_type(expected.clone()) {
            let err = match expected.clone() {
                TokenKind::EndLine => PEK::MissingSemicolon,
                TokenKind::LParen  => PEK::MissingOpeningParenthesis,
                TokenKind::RParen  => PEK::MissingClosingParenthesis,
                TokenKind::LSquirly => PEK::MissingOpeningCurlyBrace,
                TokenKind::RSquirly => PEK::MissingClosingCurlyBrace,
                _ => PEK::UnexpectedSymbol(expected, self.peek().kind.clone())

            };
            return parse_err!(self.peek().line_num, err)
        };
        self.advance();
        Ok(())
    }

    // Rule: expression -> assignment
    fn expression(&mut self) -> ParseResult<Expr> {
        // println!("expression called");
        Ok(self.assignment()?)
    }

    // Rule: assignment -> logic_or | Identifier "=" expression
    fn assignment(&mut self) -> ParseResult<Expr> {
        let mut expr: Expr = self.logic_or()?;

        expr = match (&expr, &self.peek().kind) {
            (Expr::Ident(v_name), TokenKind::Equal) => {
                self.advance();
                let value: Expr = self.assignment()?;
                Expr::Assign { var_name: v_name.into(), new_value: Box::new(value) }
            },
            _ => expr
        };

        Ok(expr)
    }

    // Rule: logic_or -> logic_and ( "||" logic_and )*
    fn logic_or(&mut self) -> ParseResult<Expr> {
        let mut expr: Expr = self.logic_and()?;

        if self.matches_type(TokenKind::Or) {
            let op: Operator = Operator::LogicalOr;
            self.advance();  // ||
            let right: Expr = self.logic_and()?;
            // expr = Expr::Logical { operator: op, left: Box::new(expr), right: Box::new(right) };
            expr = Expr::Dyadic { operator: op, left: Box::new(expr), right: Box::new(right) };
        }
    
        Ok(expr)
    }

    // Rule: logic_and -> term ( "&&" term )*
    fn logic_and(&mut self) -> ParseResult<Expr> {
        let mut expr: Expr = self.comparator()?;
        if self.matches_type(TokenKind::And) {
            let op: Operator = Operator::LogicalAnd;
            self.advance();
            let right: Expr = self.comparator()?;
            // expr = Expr::Logical { operator: op, left: Box::new(expr), right: Box::new(right) }
            expr = Expr::Dyadic { operator: op, left: Box::new(expr), right: Box::new(right) }
        }
        Ok(expr)
    }

    // Rule: comparator -> term ( ("==" | "!=" | "<" | "<=" | ">" | ">=" ) term )*
    fn comparator(&mut self) -> ParseResult<Expr> {
        let mut expr: Expr = self.term()?;
        let comparators = vec![TokenKind::EqualTo, TokenKind::Negate,
                                            TokenKind::LessThan, TokenKind::LessEquals,
                                            TokenKind::GreaterThan, TokenKind::GreaterEquals];
        
        if self.matches_types(comparators) {
            if let Ok(op) = self.eval_operator() {
                let right: Expr = self.term()?;
                expr = Expr::Dyadic {
                    operator: op,
                    left: Box::new(expr),
                    right: Box::new(right)
                }
            }
        }
        Ok(expr)
    }
    
    // Rule: term -> factor ( ( "+" | "-" ) factor)*
    fn term(&mut self) -> ParseResult<Expr> {
        let mut expr: Expr = self.factor()?;

        while self.matches_types(vec![TokenKind::Plus, TokenKind::Minus]) {
            let op: Operator = self.eval_operator()?;
            let right: Expr = self.factor()?;
            expr = Expr::Dyadic { operator: op, left: Box::new(expr), right: Box::new(right) }
        }
        Ok(expr)
    }

    // Rule: factor -> unary ( ( "*" | "/" | "%" ) unary)*
    fn factor(&mut self) -> ParseResult<Expr> {
        let mut expr: Expr = self.unary()?;
        // allowed tokens
        while self.matches_types(vec![TokenKind::Star, TokenKind::Slash, TokenKind::Percent]) {
            let op: Operator = self.eval_operator()?;
            let right: Expr = self.unary()?;
            expr = Expr::Dyadic { operator: op, left: Box::new(expr), right: Box::new(right) }
        }
        Ok(expr)
    }

    // Rule: unary -> fn_call | "+" fn_call | "-" fn_call
    fn unary(&mut self) -> ParseResult<Expr> {
        if self.matches_type(TokenKind::Plus) {
            self.advance();
            let operand: Expr = self.fn_call()?;
            return Ok(Expr::Monadic { operator: Operator::Plus, operand: Box::new(operand) })
        }
        if self.matches_type(TokenKind::Minus) {
            self.advance();
            let operand: Expr = self.fn_call()?;
            return Ok(Expr::Monadic { operator: Operator::Minus, operand: Box::new(operand) })
        }
        self.fn_call()
    }

    // fn_call -> atom | atom "(" expression ( "," expression )* ")"
    fn fn_call(&mut self) -> ParseResult<Expr> {
        let expr: Expr = self.atom()?;
        // println!("fn_call pre-match: {:?}, {:?}", expr, peek);
        match (&expr, &self.peek().kind) {
            (Expr::Ident(callee), TokenKind::LParen) => {
                self.advance();  // LParen
                let mut args: Vec<Expr> = Vec::new();
                while !self.matches_type(TokenKind::RParen) {
                    args.push(self.expression()?);
                    if self.matches_type(TokenKind::Comma) {
                        self.advance();
                    }
                }
                if args.len() > 255 {
                    // return Err(format!("Too many arguments for function {}", callee))
                    return Err(ParseError {
                        line_num: self.peek().line_num,
                        kind: PEK::ExcessParameters
                    })
                }
                self.advance();  // RParen
                Ok(Expr::Call { callee: callee.into(), args: args })
            },
            _ => Ok(expr)
        }
    }

    // Rule: atom -> "(" expression ")" | Integer | Identifier | Boolean
    fn atom(&mut self) -> ParseResult<Expr> {
        let expr: Expr = match &self.peek().kind {
            TokenKind::Int(n) => Expr::Int(*n),
            TokenKind::Ident(s) => Expr::Ident(s.into()),
            TokenKind::Boolean(b) => Expr::Boolean(*b),
            TokenKind::LParen => {
                self.advance();  // Move past the LParen "("
                let expr: Expr = self.expression()?;
                // Revisit: Is it still needed after parentheses balance checking?
                // if self.pos >= self.tokens.len() || self.peek() != &Token::RParen {
                //     return Err("Missing closing parenthesis".into())
                // }
                expr
            },
            _ => return parse_err!(self.peek().line_num, PEK::UnrecognisedSymbol(format!("{:?}", self.peek().kind)))
        };
        self.advance();
        Ok(expr)
    }

}

#[cfg(test)]
mod tests {
    use std::vec;

    use crate::errors::*;
    use crate::parser::Parser;
    use crate::tokens::Token;
    use crate::tokens::TokenKind as TK;
    use crate::ast::*;

    macro_rules! token {
        ($line_number:expr, $token_kind:expr) => {
            Token{line_num: $line_number, kind: $token_kind}
        };

        ($token_kind:expr) => {
            Token{line_num: 0, kind: $token_kind}
        };
    }

    macro_rules! tokens {
        ($token_vec:expr) => {
            $token_vec.iter().map(|tk| token!(tk.clone())).collect()
        };
    }

    #[test]
    fn token_type_check() {
        // let p: Parser = Parser::new(vec![TK::Int(5)]);
        let p: Parser = Parser::new(vec![token!(0, TK::Int(5))]);

        assert_eq!(p.matches_type(TK::Int(-9999)), true);
        assert_eq!(p.matches_type(TK::RSquirly), false);

        assert_eq!(p.matches_types(vec![TK::Int(0)]), true);
        assert_eq!(p.matches_types(vec![TK::EndLine, TK::Colon, TK::Int(9999)]), true);
        assert_eq!(p.matches_types(vec![TK::LParen, TK::Let, TK::Function]), false);
        assert_eq!(p.matches_types(vec![TK::Equal]), false);
    }

    #[test]
    fn simple() {
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Int(0))]),
            Parser::new(vec![token!(0, TK::Int(0)), token!(0, TK::EndLine), token!(0, TK::EOF)]).parse()
        );
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Int(5))]),
            Parser::new(vec![token!(0, TK::Int(5)), token!(0, TK::EndLine), token!(0, TK::EOF)]).parse()
        );
    }

    #[test]
    fn bad_token_combo_error_check() {
        assert_eq!(
            // Err("Two Int tokens in a row".into()),
            parse_err!(0, ParseErrorKind::IllegalTokenSequence),
            Parser::new(vec![token!(0, TK::Int(0)), token!(0, TK::Int(0)), token!(0, TK::EndLine), token!(0, TK::EOF)]).parse()
        );
        assert_eq!(
            // Err("Two Int tokens in a row".into()),
            parse_err!(0, ParseErrorKind::IllegalTokenSequence),
            Parser::new(vec![
                token!(0, TK::Int(0)), token!(0, TK::Plus), token!(0, TK::Int(0)), token!(0, TK::Int(0)),
                token!(0, TK::EndLine), token!(0, TK::EOF)
            ]).parse()
        );
        assert_eq!(
            // Err("Two Ident tokens in a row".into()),
            parse_err!(0, ParseErrorKind::IllegalTokenSequence),
            Parser::new(vec![
                token!(0, TK::Ident("a".into())), token!(0, TK::Ident("b".into())),
                token!(0, TK::EndLine), token!(0, TK::EOF)
            ]).parse()
        );
        assert_eq!(
            // Err("Ident token immediately followed by Int token".into()),
            parse_err!(0, ParseErrorKind::IllegalTokenSequence),
            Parser::new(vec![
                token!(0, TK::Ident("a".into())), token!(0, TK::Int(0)),
                token!(0, TK::EndLine), token!(0, TK::EOF)
            ]).parse()
        );
    }

    #[test]
    fn bracket_balance() {
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Int(0))]),
            Parser::new(tokens!(vec![TK::LParen, TK::LParen, TK::LParen, TK::Int(0),
                             TK::RParen, TK::RParen, TK::RParen, TK::EndLine, TK::EOF])).parse()
        );
        assert_eq!(
            parse_err!(0, ParseErrorKind::MissingOpeningParenthesis),
            Parser::new(tokens!(vec![TK::RParen, TK::EOF])).parse()
        );
        assert_eq!(
            parse_err!(0, ParseErrorKind::MissingClosingParenthesis),
            Parser::new(tokens!(vec![TK::LParen, TK::LParen, TK::RParen, TK::EndLine, TK::EOF])).parse()
        );
        assert_eq!(
            parse_err!(0, ParseErrorKind::MissingClosingParenthesis),
            Parser::new(tokens!(vec![TK::LParen, TK::RParen, TK::LParen,
                             TK::Ident("a".into()), TK::EndLine, TK::EOF])).parse()
        );
        assert_eq!(
            parse_err!(0, ParseErrorKind::MissingOpeningParenthesis),
            Parser::new(tokens!(vec![TK::LParen, TK::LParen, TK::LParen, TK::RParen,
                            TK::RParen, TK::RParen, TK::RParen, TK::EndLine, TK::EOF])).parse()
        );

    }

    #[test]        
    fn simple_monadic() {
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Monadic {
                operator: Operator::Plus,
                operand: Box::new(Expr::Int(256))
            })]),  // +256;
            Parser::new(vec![token!(TK::Plus), token!(TK::Int(256)), token!(TK::EndLine),
                            token!(TK::EOF)]).parse()
        );
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Monadic {
                operator: Operator::Minus,
                operand: Box::new(Expr::Int(15))
            })]),  // +256;
            Parser::new(vec![token!(TK::Minus), token!(TK::Int(15)), token!(TK::EndLine),
                            token!(TK::EOF)]).parse()
        );
    }

    #[test]
    fn simple_dyadic() {
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Dyadic {
                operator: Operator::Plus,
                left: Box::new(Expr::Int(5)),
                right: Box::new(Expr::Int(6))
            })]),  // 5 + 6;
            Parser::new(tokens!(vec![TK::Int(5), TK::Plus, TK::Int(6), TK::EndLine, TK::EOF])).parse()
        );
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Dyadic {
                operator: Operator::Minus,
                left: Box::new(Expr::Int(20)),
                right: Box::new(Expr::Int(5))
            })]),  // 20 - 5;
            Parser::new(tokens!(vec![TK::Int(20), TK::Minus, TK::Int(5), TK::EndLine, TK::EOF])).parse()
        );
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Dyadic {
                operator: Operator::Minus,
                left: Box::new(Expr::Int(256)),
                right: Box::new(Expr::Monadic {
                    operator: Operator::Minus,
                    operand: Box::new(Expr::Int(256))
                })
            })]),  // 256 - -256;
            Parser::new(tokens!(vec![TK::Int(256), TK::Minus, TK::Minus, TK::Int(256),
                                    TK::EndLine, TK::EOF])).parse()
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
            })]),  // 1 + 2 + 3;
            Parser::new(tokens!(vec![TK::Int(1), TK::Plus, TK::Int(2), TK::Plus,
                                    TK::Int(3), TK::EndLine, TK::EOF])).parse()
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
            })]),  // 1 + two - 4 + -8;
            Parser::new(tokens!(vec![TK::Int(1), TK::Plus, TK::Ident("two".into()), TK::Minus, TK::Int(4),
                             TK::Plus, TK::Minus, TK::Int(8), TK::EndLine, TK::EOF])).parse()
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
            })]),  // 10 + -20 + -30;
            Parser::new(tokens!(vec![TK::Int(10), TK::Plus, TK::Minus, TK::Int(20),
                             TK::Plus, TK::Minus, TK::Int(30), TK::EndLine, TK::EOF])).parse()
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
            Ok(&n_plus_n_minus_n),  // 512 + 256 - 128;
            Parser::new(tokens!(vec![TK::Int(512), TK::Plus, TK::Int(256),
                             TK::Minus, TK::Int(128), TK::EndLine, TK::EOF])).parse().as_ref()
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
            Ok(&n_plus_n_slash_n), // 512 + 256 / 128;
            Parser::new(tokens!(vec![TK::Int(512), TK::Plus, TK::Int(256),
                             TK::Slash, TK::Int(128), TK::EndLine, TK::EOF])).parse().as_ref()
        );
        // Parentheses precedence
        assert_eq!(
            Ok(&n_plus_n_minus_n), // ( 512 + 256 - 128);
            Parser::new(tokens!(vec![TK::LParen, TK::Int(512), TK::Plus, TK::Int(256),
                             TK::Minus, TK::Int(128), TK::RParen, TK::EndLine, TK::EOF])).parse().as_ref()
        );
        assert_eq!(
            Ok(&n_plus_n_slash_n),  // ( 512 + 256 / 128 );
            Parser::new(tokens!(vec![TK::LParen, TK::Int(512), TK::Plus, TK::Int(256),
                             TK::Slash, TK::Int(128), TK::RParen, TK::EndLine, TK::EOF])).parse().as_ref()
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
            })]),  // 512 + (256 + one_two_eight);
            Parser::new(tokens!(vec![TK::Int(512), TK::Plus, TK::LParen, TK::Int(256),
                             TK::Plus, TK::Ident("one_two_eight".into()),
                             TK::RParen, TK::EndLine, TK::EOF])).parse()
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
            })]),  // 16 / 4 * -1;
            Parser::new(tokens!(vec![TK::Int(16), TK::Slash, TK::Int(4), TK::Star,
                            TK::Minus, TK::Int(1), TK::EndLine, TK::EOF])).parse()
        );
    }

    #[test]
    fn control_flow() {
        assert_eq!(
            // if (true) {}
            Ok(vec![Stmt::If {cond: Expr::Boolean(true),
                              then: Box::new(Stmt::Block(vec![])),
                              els: None} ]),
                            //   els: Box::new(Stmt::Block(vec![]))} ]),
            Parser::new(tokens!(vec![TK::If, TK::LParen, TK::Boolean(true), TK::RParen,
                                TK::LSquirly, TK::RSquirly, TK::EOF])).parse()
        );
        assert_eq!(
            // if (a == 8) { print(0); }
            Ok(vec![Stmt::If{
                cond: Expr::Dyadic {
                    operator: Operator::EqualTo,
                    left: Box::new(Expr::Ident("a".into())),
                    right: Box::new(Expr::Int(8))
                },
                then: Box::new(Stmt::Block(vec![Stmt::Print(Expr::Int(0))])),
                els: None}]),
            Parser::new(tokens!(vec![TK::If, TK::LParen, TK::Ident("a".into()), TK::EqualTo,
                            TK::Int(8), TK::RParen, TK::LSquirly,
                                TK::Print, TK::LParen, TK::Int(0), TK::RParen, TK::EndLine,
                            TK::RSquirly, TK::EOF])
                        ).parse()
        );
        assert_eq!(
            // if (a != 8) { print(0); }
            Ok(vec![Stmt::If{
                cond: Expr::Dyadic {
                    operator: Operator::NotEqualTo,
                    left: Box::new(Expr::Ident("a".into())),
                    right: Box::new(Expr::Int(8))
                },
                then: Box::new(Stmt::Block(vec![Stmt::Print(Expr::Int(0))])),
                els: None}]),
            Parser::new(tokens!(vec![TK::If, TK::LParen, TK::Ident("a".into()), TK::Negate, TK::Equal,
                            TK::Int(8), TK::RParen, TK::LSquirly,
                                TK::Print, TK::LParen, TK::Int(0), TK::RParen, TK::EndLine,
                            TK::RSquirly, TK::EOF])).parse()
        );
        assert_eq!(
            // if (abc < -8) { print(0); } else { print(1); print(2); }
            Ok(vec![Stmt::If{
                cond: Expr::Dyadic {
                    operator: Operator::LessThan,
                    left: Box::new(Expr::Ident("abc".into())),
                    right: Box::new(Expr::Monadic {
                        operator: Operator::Minus,
                        operand: Box::new(Expr::Int(8))
                    })
                },
                then: Box::new(Stmt::Block(vec![Stmt::Print(Expr::Int(0))])),
                els: Some(Box::new(Stmt::Block(vec![Stmt::Print(Expr::Int(1)), Stmt::Print(Expr::Int(2))])))}]),
            Parser::new(tokens!(vec![TK::If, TK::LParen, TK::Ident("abc".into()), TK::LessThan, TK::Minus, TK::Int(8), TK::RParen,
                            TK::LSquirly,
                                TK::Print, TK::LParen, TK::Int(0), TK::RParen, TK::EndLine,
                            TK::RSquirly, TK::Else, TK::LSquirly,
                                TK::Print, TK::LParen, TK::Int(1), TK::RParen, TK::EndLine,
                                TK::Print, TK::LParen, TK::Int(2), TK::RParen, TK::EndLine,
                            TK::RSquirly, TK::EOF])).parse()
        );
        assert_eq!(
            // if (false || true) {}
            Ok(vec![Stmt::If {
                cond: Expr::Dyadic {
                    operator: Operator::LogicalOr,
                    left: Box::new(Expr::Boolean(false)),
                    right: Box::new(Expr::Boolean(true))
                },
                then: Box::new(Stmt::Block(vec![])),
                els: None
            }]),
            Parser::new(tokens!(vec![TK::If, TK::LParen, TK::Boolean(false), TK::Or, TK::Boolean(true),
                    TK::RParen, TK::LSquirly, TK::RSquirly, TK::EOF]))
                    .parse()
        );
        assert_eq!(
            // if (0 <= a && a <= 4) {}
            Ok(vec![Stmt::If {
                cond: Expr::Dyadic {
                    operator: Operator::LogicalAnd,
                    left: Box::new(Expr::Dyadic {
                        operator: Operator::LessEquals,
                        left: Box::new(Expr::Int(0)),
                        right: Box::new(Expr::Ident("a".into()))
                    }),
                    right: Box::new(Expr::Dyadic {
                        operator: Operator::LessEquals,
                        left: Box::new(Expr::Ident("a".into())),
                        right: Box::new(Expr::Int(4))
                    })
                },
                then: Box::new(Stmt::Block(vec![])),
                els: None
            }]),
            Parser::new(tokens!(vec![TK::If, TK::LParen,
                                TK::Int(0), TK::LessEquals, TK::Ident("a".into()), TK::And,
                                TK::Ident("a".into()), TK::LessEquals, TK::Int(4), TK::RParen,
                                TK::LSquirly, TK::RSquirly, TK::EOF])).parse()
        );
        // if (true || true && true) {}  // Logical And has higher precedence over Logical Or
        assert_eq!(
            Ok(vec![Stmt::If {
                cond: Expr::Dyadic {
                    operator: Operator::LogicalOr,
                    left: Box::new(Expr::Boolean(true)),
                    right: Box::new(Expr::Dyadic {
                        operator: Operator::LogicalAnd,
                        left: Box::new(Expr::Boolean(true)),
                        right: Box::new(Expr::Boolean(true))
                    })
                },
                then: Box::new(Stmt::Block(vec![])),
                els: None
            }]),
            Parser::new(tokens!(vec![TK::If, TK::LParen, TK::Boolean(true), TK::Or, TK::Boolean(true),
                                TK::And, TK::Boolean(true), TK::RParen, TK::LSquirly, TK::RSquirly,
                                TK::EOF]))
                             .parse()
        );
    }

    #[test]
    fn variables() {
        // var abcdefg;
        assert_eq!(
            Ok(vec![Stmt::VarDecl("abcdefg".into(), None)]),
            Parser::new(vec![TK::Var, TK::Ident("abcdefg".into()), TK::EndLine, TK::EOF]
                        .iter()
                        .map(|tk| token!(tk.clone()))
                        .collect())
                        .parse()
        );
        // var abc = 6;
        assert_eq!(
            Ok(vec![Stmt::VarDecl("abc".into(),
                            Some(Box::new(Stmt::Expr(Expr::Int(6)))))]),
            Parser::new(tokens!(vec![TK::Var, TK::Ident("abc".into()), TK::Equal,
                             TK::Int(6), TK::EndLine, TK::EOF]))
                            .parse()
        );
        // var abc = 3 + 2;
        assert_eq!(
            Ok(vec![Stmt::VarDecl("abc".into(),
                            Some(Box::new(Stmt::Expr(Expr::Dyadic {
                                                        operator: Operator::Plus,
                                                        left: Box::new(Expr::Int(3)),
                                                        right: Box::new(Expr::Int(2))
                                                    }))))]),
            Parser::new(tokens!(vec![TK::Var, TK::Ident("abc".into()), TK::Equal, TK::Int(3),
                             TK::Plus, TK::Int(2), TK::EndLine, TK::EOF]))
                             .parse()
        );
        // var b = if (false) { 0; } else { a; };
        assert_eq!(
            Ok(vec![Stmt::VarDecl("b".into(),
                            Some(Box::new(Stmt::If {
                                cond: Expr::Boolean(false),
                                then: Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Int(0))])),
                                els: Some(Box::new(Stmt::Block(vec![Stmt::Expr(Expr::Ident("a".into()))])))
                            })))]),
            Parser::new(tokens!(vec![TK::Var, TK::Ident("b".into()), TK::Equal,
                             TK::If,
                                TK::LParen, TK::Boolean(false), TK::RParen,
                                TK::LSquirly, TK::Int(0), TK::EndLine, TK::RSquirly,
                             TK::Else,
                                TK::LSquirly, TK::Ident("a".into()), TK::EndLine, TK::RSquirly,
                             TK::EndLine, TK::EOF]))
                             .parse()
        );
        // abc = 10;  // assumes declared abc variable
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Assign {
                var_name: "abc".into(), new_value: Box::new(Expr::Int(10))
            })]),
            Parser::new(tokens!(vec![TK::Ident("abc".into()), TK::Equal,
                             TK::Int(10), TK::EndLine, TK::EOF]))
                             .parse()
        );
    }

    #[test]
    fn variables_errors() {
        assert_eq!(  // var ;
            parse_err!(0, ParseErrorKind::MissingIdentifier),
            Parser::new(tokens!(vec![TK::Var, TK::EndLine, TK::EOF])).parse()
        );
        assert_eq!(  // var = 5;
            parse_err!(0, ParseErrorKind::MissingIdentifier),
            Parser::new(tokens!(vec![TK::Var, TK::Equal, TK::Int(5), TK::EndLine, TK::EOF])).parse()
        );
        // assert_eq!(
        //     parse_err!(0, ParseErrorKind::UnexpectedSymbol(TK::Equal, TK::Int(5))),
        //     Parser::new(tokens!(vec![TK::Var, TK::Ident("a".into()), TK::Int(5), TK::EndLine, TK::EOF])).parse()
        // );
        assert_eq!(
            // var n = 0;
            // var ;
            parse_err!(1, ParseErrorKind::MissingIdentifier),
            Parser::new(vec![
                token!(0, TK::Var), token!(0, TK::Ident("n".into())), token!(0, TK::EndLine),
                token!(1, TK::Var), token!(1, TK::EndLine), token!(0, TK::EOF)
            ]).parse()
        );
    }

    #[test]
    fn loops() {
        // while (true) {}
        assert_eq!(
            Ok(vec![Stmt::While { cond: Expr::Boolean(true), body: Box::new(Stmt::Block(vec![])) }]),
            Parser::new(tokens!(vec![TK::While, TK::LParen, TK::Boolean(true), TK::RParen,
                TK::LSquirly, TK::RSquirly, TK::EOF]))
                .parse()
        );

        // var i = 0;
        // while (i < 5) {
        //     i = i + 1;
        // }
        // i;
        assert_eq!(Ok(vec![
            Stmt::VarDecl("i".into(), Some(Box::new(Stmt::Expr(Expr::Int(0))))),
            Stmt::While {
                cond: Expr::Dyadic {
                    operator: Operator::LessThan,
                    left: Box::new(Expr::Ident("i".into())),
                    right: Box::new(Expr::Int(5))
                }, body: Box::new(Stmt::Block(vec![
                    Stmt::Expr(Expr::Assign {
                        var_name: "i".into(),
                        new_value: Box::new(Expr::Dyadic {
                            operator: Operator::Plus,
                            left: Box::new(Expr::Ident("i".into())),
                            right: Box::new(Expr::Int(1))
                        })
                    })
                ]))
            },
            Stmt::Expr(Expr::Ident("i".into()))
            ]),
            Parser::new(tokens!(vec![
                TK::Var, TK::Ident("i".into()), TK::Equal, TK::Int(0), TK::EndLine,
                TK::While, TK::LParen, TK::Ident("i".into()), TK::LessThan, TK::Int(5), TK::RParen, TK::LSquirly,
                    TK::Ident("i".into()), TK::Equal, TK::Ident("i".into()), TK::Plus, TK::Int(1), TK::EndLine,
                TK::RSquirly,
                TK::Ident("i".into()), TK::EndLine, TK::EOF
            ])).parse()
        );

        // for (;;) {}
        assert_eq!(
            Ok(vec![Stmt::While { cond: Expr::Boolean(true), body: Box::new(Stmt::Block(vec![])) }]),
            Parser::new(tokens!(vec![TK::For, TK::LParen, TK::EndLine, TK::EndLine, TK::RParen,
                TK::LSquirly, TK::RSquirly, TK::EOF]))
                .parse()
        );
        
        // for (var i = 0; i < 10; i = i + 1) {}
        assert_eq!(Ok(vec![
                Stmt::Block(vec![
                    Stmt::VarDecl("i".into(), Some(Box::new(Stmt::Expr(Expr::Int(0))))),
                    Stmt::While {
                        cond: Expr::Dyadic {
                            operator: Operator::LessThan,
                            left: Box::new(Expr::Ident("i".into())),
                            right: Box::new(Expr::Int(10))
                        },
                        body: Box::new(Stmt::Block(vec![
                            Stmt::Expr(Expr::Assign {
                                var_name: "i".into(),
                                new_value: Box::new(Expr::Dyadic {
                                    operator: Operator::Plus,
                                    left: Box::new(Expr::Ident("i".into())),
                                    right: Box::new(Expr::Int(1))
                                })
                            })
                        ]))
                    }
                ]),
            ]),
            Parser::new(tokens!(vec![
                TK::For, TK::LParen, TK::Var, TK::Ident("i".into()), TK::Equal, TK::Int(0), TK::EndLine,
                    TK::Ident("i".into()), TK::LessThan, TK::Int(10), TK::EndLine,
                    TK::Ident("i".into()), TK::Equal, TK::Ident("i".into()), TK::Plus, TK::Int(1),
                TK::RParen, TK::LSquirly, TK::RSquirly, TK::EOF
            ]))
            .parse()
        );

        // for (var i = 0; i < 10; i = i + 1) { print(i/2); }
        assert_eq!(Ok(vec![
            Stmt::Block(vec![
                Stmt::VarDecl("i".into(), Some(Box::new(Stmt::Expr(Expr::Int(0))))),
                Stmt::While {
                    cond: Expr::Dyadic {
                        operator: Operator::LessThan,
                        left: Box::new(Expr::Ident("i".into())),
                        right: Box::new(Expr::Int(10))
                    },
                    body: Box::new(Stmt::Block(vec![
                        Stmt::Print(Expr::Dyadic {
                            operator: Operator::Slash,
                            left: Box::new(Expr::Ident("i".into())),
                            right: Box::new(Expr::Int(2))
                        }),
                        Stmt::Expr(Expr::Assign {
                            var_name: "i".into(),
                            new_value: Box::new(Expr::Dyadic {
                                operator: Operator::Plus,
                                left: Box::new(Expr::Ident("i".into())),
                                right: Box::new(Expr::Int(1))
                            })
                        })
                    ]))
                }
            ]),
        ]),
        Parser::new(tokens!(vec![
            TK::For, TK::LParen, TK::Var, TK::Ident("i".into()), TK::Equal, TK::Int(0), TK::EndLine,
                TK::Ident("i".into()), TK::LessThan, TK::Int(10), TK::EndLine,
                TK::Ident("i".into()), TK::Equal, TK::Ident("i".into()), TK::Plus, TK::Int(1),
            TK::RParen, TK::LSquirly,
                TK::Print, TK::LParen, TK::Ident("i".into()), TK::Slash, TK::Int(2), TK::RParen, TK::EndLine,
            TK::RSquirly, TK::EOF
        ])).parse());
    }

    #[test]
    fn loops_errors() {
        // while loops
        assert_eq!(  // while (true);
            parse_err!(0, ParseErrorKind::MissingBlock),
            Parser::new(tokens!(vec![TK::While, TK::LParen, TK::Boolean(true), TK::RParen,
                                TK::EndLine, TK::EOF])).parse()
        );
        assert_eq!(  // while true) {}
            parse_err!(0, ParseErrorKind::MissingOpeningParenthesis),
            Parser::new(tokens!(vec![TK::While, TK::Boolean(true),
                                TK::LSquirly, TK::RSquirly, TK::EOF])).parse()
        );
        assert_eq!(  // while (true {}
            parse_err!(0, ParseErrorKind::MissingClosingParenthesis),
            Parser::new(tokens!(vec![TK::While, TK::LParen, TK::Boolean(true),
                                TK::LSquirly, TK::RSquirly, TK::EOF])).parse()
        );

        // TODO: for loops
        // assert_eq!(  // TODO: for (;;)
        //     parse_err!(0, ParseErrorKind::MissingBlock),
        //     Parser::new(tokens!(vec![TK::For, TK::LParen, TK::EndLine, TK::EndLine, TK::RParen,
        //                         TK::EOF])).parse()
        // )
    }

    #[test]
    fn function_calls() {
        // foo();
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Call { callee: "foo".into(), args: vec![] })]),
            Parser::new(tokens!(vec![TK::Ident("foo".into()), TK::LParen, TK::RParen,
                            TK::EndLine, TK::EOF])).parse()
        );

        // bar(1);
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Call {
                callee: "bar".into(),
                args: vec![Expr::Int(1)]
            })]),
            Parser::new(tokens!(vec![TK::Ident("bar".into()), TK::LParen, TK::Int(1), TK::RParen,
                                TK::EndLine, TK::EOF])).parse()
        );

        // max(0, foo());
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Call {
                callee: "max".into(),
                args: vec![Expr::Int(0), Expr::Call { callee: "foo".into(), args: vec![] }]
            })]),
            Parser::new(tokens!(vec![TK::Ident("max".into()), TK::LParen,
                                    TK::Int(0), TK::Comma,
                                    TK::Ident("foo".into()), TK::LParen,
                                TK::RParen, TK::RParen, TK::EndLine, TK::EOF]))
                                .parse()
        );

        // baz(1, 2, 3, 4);
        assert_eq!(
            Ok(vec![Stmt::Expr(Expr::Call {
                callee: "baz".into(),
                args: vec![Expr::Int(1), Expr::Int(2), Expr::Int(3), Expr::Int(4)]
            })]),
            Parser::new(tokens!(vec![TK::Ident("baz".into()), TK::LParen,
                                        TK::Int(1), TK::Comma,
                                        TK::Int(2), TK::Comma,
                                        TK::Int(3), TK::Comma,
                                        TK::Int(4),
                                    TK::RParen, TK::EndLine, TK::EOF]))
                             .parse()
        );
    }

    #[test]
    fn function_declarations() {
        // fn no_params() {
        //     print(1);
        // }
        assert_eq!(
            Ok(vec![Stmt::FnDecl {
                name: "no_params".into(),
                parameters: vec![],
                body: Box::new(Stmt::Block(vec![Stmt::Print(Expr::Int(1))]))
            }]),
            Parser::new(vec![TK::Function, TK::Ident("no_params".into()), TK::LParen, TK::RParen, TK::LSquirly,
                            TK::Print, TK::LParen, TK::Int(1), TK::RParen, TK::EndLine, TK::RSquirly, TK::EOF]
                            .iter()
                            .map(|tk| token!(tk.clone()))
                            .collect())
                            .parse()
        );

        // fn one_param(abc) {
        //     print(abc);
        // }
        assert_eq!(
            Ok(vec![Stmt::FnDecl {
                name: "one_param".into(),
                parameters: vec!["abc".into()],
                body: Box::new(Stmt::Block(vec![Stmt::Print(Expr::Ident("abc".into()))]))
            }]),
            Parser::new(vec![TK::Function, TK::Ident("one_param".into()), TK::LParen, TK::Ident("abc".into()), TK::RParen,
                             TK::LSquirly, TK::Print, TK::LParen, TK::Ident("abc".into()), TK::RParen, TK::EndLine,
                             TK::RSquirly, TK::EOF]
                             .iter()
                             .map(|tk| token!(tk.clone()))
                             .collect())
                             .parse()
        );

        // fn multi_param(abc, def) {
        //     abc = abc + def;
        //     print(abc);
        // }
        assert_eq!(
            Ok(vec![Stmt::FnDecl {
                name: "multi_param".into(),
                parameters: vec!["abc".into(), "def".into()],
                body: Box::new(Stmt::Block(vec![
                    Stmt::Expr(Expr::Assign {
                        var_name: "abc".into(),
                        new_value: Box::new(Expr::Dyadic {
                            operator: Operator::Plus,
                            left: Box::new(Expr::Ident("abc".into())),
                            right: Box::new(Expr::Ident("def".into()))
                        })
                    }),
                    Stmt::Print(Expr::Ident("abc".into()))
                ]))
            }]),
            Parser::new(vec![TK::Function, TK::Ident("multi_param".into()), TK::LParen,
                                TK::Ident("abc".into()), TK::Comma, TK::Ident("def".into()),
                             TK::RParen, TK::LSquirly,
                                TK::Ident("abc".into()), TK::Equal, TK::Ident("abc".into()), TK::Plus,
                                        TK::Ident("def".into()), TK::EndLine,
                                TK::Print, TK::LParen, TK::Ident("abc".into()), TK::RParen, TK::EndLine,
                             TK::RSquirly, TK::EOF]
                             .iter()
                             .map(|tk| token!(tk.clone()))
                             .collect())
                             .parse()
        );
    }

}