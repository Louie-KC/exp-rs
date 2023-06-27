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
    // let source = String::from("1 + 2 + 4 + 5;");
    // let source = String::from("3 + 5; print(1 + 2); 2 - 1;");
    let source = r#"
        print(6);
        if (true || false) {
            print(5);
            print(50);
        } else {
            print(3333333);
        }
        if (true) {}
        print(4);
        "#.to_string();
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
