use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use cnn::conv_layer::ActivationFunction::{LeakyReLU, Mish, ReLU, Swish, ELU};
use cnn::*;

fn rel_benches(c: &mut Criterion) {
    // vector of activation functions to benchmark
    let tests = vec![ReLU, LeakyReLU, ELU, Mish, Swish];

    let mut group = c.benchmark_group("relu");
    for act in &tests {
        group.bench_function(
            BenchmarkId::new("run", format!("{:?}_epochs_10", act)),
            move |b| b.iter(|| run::run(10, *act, *act)),
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(1).warm_up_time(Duration::from_secs(3)).measurement_time(Duration::from_secs(5810)).with_output_color(true);
    targets = rel_benches
}

criterion_main!(benches);
