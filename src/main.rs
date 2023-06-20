mod tokens;
mod ast;
mod lexer;
mod parser;
// mod parseraaa;

fn main() {
    // let source = String::from("1 + 2 + -3");
    // let source = String::from("1 + 2 + 3");
    // let source = String::from("1 + 2 - 4 + -8");
    // let source = String::from("1 + -2 + 3 + 4");
    // let source = String::from("1 * 2 * 4");
    let source = String::from("1 + 2 + 4 + 5");
    let tokens = lexer::tokenise(source);
    for token in &tokens {
        println!("{:?}", token);
    }
    let parsed = parser::Parser::new(tokens).parse();
    println!("{:?}", parsed);
}
