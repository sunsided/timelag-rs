# timelag — creating time lagged data series

This crate provides `lag_matrix` and related functions to create time-lagged
versions of time series.


```rust
use timelag::lag_matrix;

fn it_works() {
    let data = [1.0, 2.0, 3.0, 4.0];
    
    // Using infinity for padding because NaN doesn't equal itself.
    let lag = f64::INFINITY;
    let padding = f64::INFINITY;
    
    // Create three lagged versions.
    // Use a stride of 5 for the rows, i.e. pad with one extra entry.
    let direct = lag_matrix(&data, 3, lag, 5).unwrap();
    
    assert_eq!(
        direct,
        &[
            1.0, 2.0, 3.0, 4.0, padding, // original data
            lag, 1.0, 2.0, 3.0, padding, // first lag
            lag, lag, 1.0, 2.0, padding, // second lag
            lag, lag, lag, 1.0, padding, // third lag
        ]
    );
}
```
