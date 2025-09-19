# Kalman Crate

A simple Rust crate that implements a Kalman filter.

## Example

```rust
use kalman_crate::KalmanFilter;
use nalgebra::{SVector, SMatrix};

let mut kf = KalmanFilter::new(
    SVector::from_row_slice(&[0., 0., 0.]),
    &SVector::from_row_slice(&[1e12, 0., 0.]),
    |dt| SMatrix::from_row_slice(&[1.0, dt, 0.5 * dt * dt, 0.0, 1.0, dt, 0.0, 0.0, 1.0]),
    |_dt| SMatrix::identity(),
    1e-10,
);

// You usually do this on a loop
kf.predict(meas, meas_sd, dt);
let state = kf.state();
```

For a complete example check the tests on `src/lib.rs`
