use crate::parser::Parser;
use crate::interpreter::Interpreter;

mod tokens;
mod ast;
mod lexer;
mod parser;
mod interpreter;

fn main() {
    // let source = String::from("1 + 2 + -3");
    // let source = String::from("1 + -2 + 3 + 4");
    // let source = String::from("1 * 2 * 4");
    let source = String::from("1 + 2 + 4 + 5");
    let tokens = lexer::tokenise(source);
    for token in &tokens {
        println!("{:?}", token);
    }
    let parsed = Parser::new(tokens).parse();
    println!("{:?}", &parsed);

    let mut interpreter = Interpreter::new();
    let result = interpreter.interpret(parsed.unwrap());
    println!("{:?}", result);
}
