mod tokens;
mod ast;
mod lexer;
// mod parser;

fn main() {
    let source = String::from("1 + 2 + -3");

    let tokens = lexer::tokenise(source);

    for token in &tokens {
        println!("{:?}", token);
    }
}
