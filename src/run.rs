use csv::Writer;
use std::path::Path;
use std::time::Instant;
use std::{fs::File, io::Write};

use mnist::{Mnist, MnistBuilder};

use crate::cnn_struct::CNN;
use crate::conv_layer::ActivationFunction;

pub fn run(
    num_epochs: usize,
    activation_function_large: ActivationFunction,
    activation_function_small: ActivationFunction,
) {
    // Load the MNIST dataset
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let train_data: Vec<Vec<Vec<f32>>> = format_images(trn_img, 50_000);
    let train_labels: Vec<u8> = trn_lbl;

    let _test_data: Vec<Vec<Vec<f32>>> = format_images(tst_img, 10_000);
    let _test_labels: Vec<u8> = tst_lbl;
    /*
    let label_mapping = vec![
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ];
    */

    // Create a new CNN and specify its layers
    let mut cnn: CNN = CNN::new();
    cnn.add_conv_layer(28, 1, 6, 5, 1, activation_function_large);
    cnn.add_mxpl_layer(24, 6, 2, 2);
    cnn.add_conv_layer(12, 6, 9, 3, 1, activation_function_small);
    cnn.add_mxpl_layer(10, 9, 2, 2);
    cnn.add_fcl_layer(5, 9, 10);
    let mut prev: Vec<bool> = vec![false; 100];

    let log_file_name = if Path::new("log.csv").exists() {
        format!("{:?}_log.csv", activation_function_large)
    } else {
        "log.csv".to_string()
    };

    let output_file_name = if Path::new("output.csv").exists() {
        format!("{:?}_output.csv", activation_function_large)
    } else {
        "output.csv".to_string()
    };

    let mut log_file = File::create(&log_file_name).expect("create failed");
    log_file
        .write_all(b"Epoch,Activation Function,Accuracy\n")
        .expect("write failed");

    // Create the CSV file and write the header
    let mut file = File::create(&output_file_name).expect("create failed");
    file.write_all(b"Epoch,Accuracy,Total Predictions,Wrong Predictions\n")
        .expect("write failed");

    // Initialize a 10x10 confusion matrix
    let mut confusion_matrix: [[u16; 10]; 10] = [[0; 10]; 10];

    // Create the CSV file and write the header
    let mut file = File::create(&output_file_name).expect("create failed");
    file.write_all(b"Epoch,Accuracy,Total Predictions,Wrong Predictions\n")
        .expect("write failed");

    // Create a new instance of Instant before the training starts
    let start_time = Instant::now();

    // Initialize counters at the start of each epoch
    let mut total_predictions: u128 = 0;
    let mut correct_predictions: u128 = 0;
    let mut wrong_predictions: u128 = 0;

    // Calculate the accuracy
    let accuracy = correct_predictions as f64 / total_predictions as f64;

    for _epoch in 0..num_epochs {
        for index in 0..train_data.len() {
            // Train the CNN on the image
            let output: Vec<f32> = cnn.forward_propagate(vec![train_data[index].clone()]);
            let result: bool = highest_index(output.clone()) == train_labels[index];

            // Get the predicted class and the actual class
            let predicted_class = highest_index(output.clone());
            let actual_class = train_labels[index] as usize;

            // Increment total_predictions
            total_predictions += 1;

            // If the prediction is correct, increment correct_predictions
            if result {
                correct_predictions += 1;
            } else {
                wrong_predictions += 1;
            }

            // Update confusion matrix
            confusion_matrix[actual_class][predicted_class as usize] += 1;

            cnn.back_propagate(train_labels[index] as usize);

            // Keep track of the last 100 results
            prev.pop();
            prev.insert(0, result);

            let accuracy = if total_predictions > 0 {
                correct_predictions as f64 / total_predictions as f64
            } else {
                0.0
            };

            // print the results every image
            if index % 1 == 0 {
                let percentage_done = (index as f32 / train_data.len() as f32) * 100.0;
                println!(
                    "Epoch: {}, Image: {}, Activation Function: {:?}, Accuracy: {:.2}%, Time: {:.2} seconds, Done: {:.2}%",
                    _epoch,
                    index,
                    activation_function_large,
                    success(&prev) * 100.0,
                    start_time.elapsed().as_secs_f32(),
                    percentage_done
                );
            }

            file.write_all(
                format!(
                    "{},{},{},{}\n",
                    _epoch, accuracy, total_predictions, wrong_predictions
                )
                .as_bytes(),
            )
            .expect("write failed");
        }
        // At the end of each epoch, write the epoch, activation function, and accuracy to the log file
        log_file
            .write_all(
                format!(
                    "{},{:?},{:.2}\n",
                    _epoch + 1,
                    activation_function_large,
                    success(&prev)
                )
                .as_bytes(),
            )
            .expect("write failed");
    }

    // After all epochs have been processed, write the confusion matrix to a CSV file
    let mut writer = Writer::from_path("confusion_matrix.csv").expect("Unable to create file");
    for row in &confusion_matrix {
        writer.serialize(row).expect("Unable to write row");
    }
}

/// Formats the dataset into a 3D vector
fn format_images(data: Vec<u8>, num_images: usize) -> Vec<Vec<Vec<f32>>> {
    let img_width: usize = 28;
    let img_height: usize = 28;

    let mut images: Vec<Vec<Vec<f32>>> = vec![];
    for image_count in 0..num_images {
        let mut image: Vec<Vec<f32>> = vec![];
        for h in 0..img_height {
            let mut row: Vec<f32> = vec![];
            for w in 0..img_width {
                let i: usize = (image_count * 28 * 28) + (h * 28) + w;
                row.push(data[i] as f32 / 256.0);
            }
            image.push(row);
        }
        images.push(image);
    }

    images
}

/// Returns the percentage of the results that were correct
fn success(prev: &Vec<bool>) -> f32 {
    let mut num_true: u16 = 0;
    for i in 0..prev.len() {
        num_true += prev[i] as u16;
    }

    num_true as f32 / prev.len() as f32
}

/// Returns the index of the highest value in the output vector
fn highest_index(output: Vec<f32>) -> u8 {
    let mut highest_index: u8 = 127;
    let mut highest_value: f32 = 0.0;

    for i in 0..output.len() {
        if output[i] > highest_value {
            highest_value = output[i];
            highest_index = i as u8;
        }
    }

    return highest_index;
}
