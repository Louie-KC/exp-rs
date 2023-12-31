use std::env;
use std::fs;

use crate::parser::Parser;
use crate::interpreter::Interpreter;

mod tokens;
mod ast;
mod lexer;
mod parser;
mod interpreter;
mod errors;

fn main() {
    let args: Vec<String> = env::args().collect();
    let source = match args.get(1) {
        Some(path) => fs::read_to_string(path).expect("Source file readable"),
        None => "".to_string()
    };
    
    println!("{}", source);

    let tokens = lexer::tokenise(source);
    let parsed = match Parser::new(tokens).parse() {
        Ok(p) => p,
        Err(e) => {
            println!("{}", e);
            return;  // Early exit main
        },
    };
    println!("{:?}", &parsed);

    let mut interpreter = Interpreter::new();
    // let result = interpreter.interpret(&parsed);
    let result = match interpreter.interpret(&parsed) {
        Ok(out) => out,
        Err(e) => {
            println!("{}", e);
            return  // Early exit main
        },
    };
    println!("Interpreter result: {}", result);
}
