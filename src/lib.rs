use core::f64;
use nalgebra::{SMatrix, SVector};

#[derive(Debug, Clone)]
pub struct KalmanFilter<const X_DIM: usize, const MEAS_DIM: usize, F, G>
where
    F: Fn(f64) -> SMatrix<f64, X_DIM, X_DIM>,
    G: Fn(f64) -> SMatrix<f64, MEAS_DIM, X_DIM>,
{
    x: SVector<f64, X_DIM>,             // State vector
    p: SMatrix<f64, X_DIM, X_DIM>,      // Covariance matrix
    q: ProcessNoiseFn<X_DIM, MEAS_DIM>, // Process noise covariance
    g: G,                               // Measurement model
    f: F,                               // Prediction model
}

impl<const X_DIM: usize, const MEAS_DIM: usize, F, G> KalmanFilter<X_DIM, MEAS_DIM, F, G>
where
    F: Fn(f64) -> SMatrix<f64, X_DIM, X_DIM>,
    G: Fn(f64) -> SMatrix<f64, MEAS_DIM, X_DIM>,
{
    pub fn new(
        initial: SVector<f64, X_DIM>,
        initial_uncertainty: &SVector<f64, X_DIM>,
        prediction_model: F,
        measurement_model: G,
        variance: f64,
    ) -> Self {
        Self {
            x: initial,
            p: SMatrix::from_diagonal(initial_uncertainty),
            q: ProcessNoiseFn { variance },
            g: measurement_model,
            f: prediction_model,
        }
    }

    pub fn predict(&mut self, z: SVector<f64, MEAS_DIM>, z_sd: SVector<f64, MEAS_DIM>, dt: f64) {
        let f = (self.f)(dt);
        let g = (self.g)(dt);
        let q = self.q.call(&g);

        // Predict state
        self.x = f * self.x;
        self.p = f * self.p * f.transpose() + q;

        // Measurement noise covariance from sd
        let r = SMatrix::from_diagonal(&z_sd.component_mul(&z_sd));
        // Kalman gain
        let s = g * self.p * g.transpose() + r;
        let k = self.p * g.transpose() * s.try_inverse().unwrap_or(SMatrix::zeros());

        // Update state
        let y = z - g * self.x;
        self.x = self.x + k * y;
        self.p = (SMatrix::identity() - k * g) * self.p;
    }

    pub fn state(&self) -> &SVector<f64, X_DIM> {
        &self.x
    }
}

#[derive(Debug, Clone)]
struct ProcessNoiseFn<const X_DIM: usize, const MEAS_DIM: usize> {
    variance: f64,
}

impl<const X_DIM: usize, const MEAS_DIM: usize> ProcessNoiseFn<X_DIM, MEAS_DIM> {
    fn call(
        &self,
        measurement_model: &SMatrix<f64, MEAS_DIM, X_DIM>,
    ) -> SMatrix<f64, X_DIM, X_DIM> {
        // https://en.wikipedia.org/wiki/Kalman_filter#Example_application,_technical
        // Q = G * Gt * variance formula
        measurement_model.tr_mul(measurement_model) * self.variance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::{Distribution, Normal};

    #[test]
    fn it_works() {
        let mut diff = f64::MAX;
        let dt = 0.1;
        let steps = 1_000_000;

        // True motion: constant accel 1.0
        let true_acc = 1.;

        let mut kf = KalmanFilter::new(
            SVector::from_row_slice(&[0., 0., 0.]),
            &SVector::from_row_slice(&[1e12, 0., 0.]),
            |dt| SMatrix::from_row_slice(&[1.0, dt, 0.5 * dt * dt, 0.0, 1.0, dt, 0.0, 0.0, 1.0]),
            |_dt| SMatrix::identity(),
            1e-10,
        );

        // Noise distributions for measurements
        let pos_noise = Normal::new(0.0, 2.).unwrap();
        let vel_noise = Normal::new(0.0, 3.).unwrap();
        let acc_noise = Normal::new(0.0, 0.05).unwrap();

        let mut rng = rand::rng();

        for i in 0..steps {
            let true_pos = 0.5 * true_acc * ((i as f64) * dt) * ((i as f64) * dt);
            let true_velocity = true_acc * (i as f64) * dt;

            let meas: SVector<f64, 3> = SVector::from_row_slice(&[
                true_pos + pos_noise.sample(&mut rng),
                true_velocity + vel_noise.sample(&mut rng),
                true_acc + acc_noise.sample(&mut rng),
            ]);

            let meas_sd = SVector::from_row_slice(&[2., 3., 0.05]);
            kf.predict(meas, meas_sd, dt);

            let state: &SVector<f64, 3> = kf.state();
            diff = (state[0] - true_pos).abs();
        }

        assert!(diff < 0.5);
    }
}
