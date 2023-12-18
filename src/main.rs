use cnn::conv_layer::ActivationFunction::{LeakyReLU, Mish, ReLU, Swish, ELU};
use cnn::*;

fn main() {
    run::run(1, ELU, ELU);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run() {
        let activation_functions = vec![ReLU, LeakyReLU, ELU, Mish, Swish];

        for activation_function in &activation_functions {
            println!("Testing activation function: {:?}", activation_function);
            run::run(30, *activation_function, *activation_function);
        }
    }
}
