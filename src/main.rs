use std::env;
use std::fs;

use crate::parser::Parser;
use crate::interpreter::Interpreter;

mod tokens;
mod ast;
mod lexer;
mod parser;
mod interpreter;

fn main() {
    let args: Vec<String> = env::args().collect();
    let source = match args.get(1) {
        Some(path) => fs::read_to_string(path).expect("Source file readable"),
        None => "".to_string()
    };
    
    println!("{}", source);

    let tokens = lexer::tokenise(source);
    // for token in &tokens {
    //     println!("{:?}", token);
    // }
    let parsed = Parser::new(tokens).parse().unwrap();
    println!("{:?}", &parsed);

    let mut interpreter = Interpreter::new();
    let result = interpreter.interpret(&parsed);
    println!("Interpret result: {}", result.unwrap());
}
