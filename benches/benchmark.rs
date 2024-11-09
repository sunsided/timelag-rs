use criterion::{black_box, criterion_group, criterion_main, Criterion};
use timelag::{lag_matrix, lag_matrix_2d, MatrixLayout};

pub fn benchmark_lag_matrix(c: &mut Criterion) {
    let data = [42.0, 40.0, 38.0, 36.0];
    let lag = f64::INFINITY;

    c.bench_function("lag_matrix_0_to_3", |b| {
        b.iter(|| {
            let _ = lag_matrix(
                black_box(&data),
                black_box(0..=3),
                black_box(lag),
                black_box(0),
            )
            .unwrap();
        })
    });

    c.bench_function("lag_matrix_stride_5", |b| {
        b.iter(|| {
            let _ = lag_matrix(
                black_box(&data),
                black_box(0..=3),
                black_box(lag),
                black_box(5),
            )
            .unwrap();
        })
    });

    c.bench_function("lag_matrix_stride_8", |b| {
        b.iter(|| {
            let _ = lag_matrix(
                black_box(&data),
                black_box(0..=3),
                black_box(lag),
                black_box(8),
            )
            .unwrap();
        })
    });
}

pub fn benchmark_lag_matrix_2d(c: &mut Criterion) {
    let data_rowwise = [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
    let data_columnwise = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0];
    let lag = f64::INFINITY;

    c.bench_function("lag_matrix_2d_rowwise_stride_5", |b| {
        b.iter(|| {
            let _ = lag_matrix_2d(
                black_box(&data_rowwise),
                black_box(MatrixLayout::RowMajor(4)),
                black_box(0..=3),
                black_box(lag),
                black_box(5),
            )
            .unwrap();
        })
    });

    c.bench_function("lag_matrix_2d_rowwise_stride_7", |b| {
        b.iter(|| {
            let _ = lag_matrix_2d(
                black_box(&data_rowwise),
                black_box(MatrixLayout::RowMajor(4)),
                black_box([1, 3, 2]),
                black_box(lag),
                black_box(5),
            )
            .unwrap();
        })
    });

    c.bench_function("lag_matrix_2d_columnwise_stride_9", |b| {
        b.iter(|| {
            let _ = lag_matrix_2d(
                black_box(&data_columnwise),
                black_box(MatrixLayout::ColumnMajor(4)),
                black_box(0..=3),
                black_box(lag),
                black_box(9),
            )
            .unwrap();
        })
    });

    c.bench_function("lag_matrix_2d_columnwise_stride_7", |b| {
        b.iter(|| {
            let _ = lag_matrix_2d(
                black_box(&data_columnwise),
                black_box(MatrixLayout::ColumnMajor(4)),
                black_box([1, 3, 2]),
                black_box(lag),
                black_box(7),
            )
            .unwrap();
        })
    });

    // Benchmark with a significantly longer series for lag_matrix_2d
    let long_data_rowwise: Vec<f64> = (0..20_000).map(|i| i as f64).collect();

    c.bench_function("lag_matrix_2d_long_series_rowwise", |b| {
        b.iter(|| {
            let _ = lag_matrix_2d(
                black_box(&long_data_rowwise),
                black_box(MatrixLayout::RowMajor(20_000)),
                black_box(0..=999),
                black_box(lag),
                black_box(20_000),
            )
            .unwrap();
        })
    });
}

criterion_group!(benches, benchmark_lag_matrix, benchmark_lag_matrix_2d);
criterion_main!(benches);
