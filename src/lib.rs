use core::f64;
use nalgebra::{SMatrix, SVector};

#[derive(Debug, Clone)]
pub struct KalmanFilter<const X_DIM: usize, const MEAS_DIM: usize, F, G>
where
    F: Fn(f64, &SVector<f64, X_DIM>) -> SMatrix<f64, X_DIM, X_DIM>,
    G: Fn(f64, &SVector<f64, X_DIM>) -> SMatrix<f64, X_DIM, MEAS_DIM>,
{
    x: SVector<f64, X_DIM>,             // State vector
    p: SMatrix<f64, X_DIM, X_DIM>,      // Covariance matrix
    q: ProcessNoiseFn<X_DIM, MEAS_DIM>, // Process noise covariance
    g: G,                               // Measurement model
    f: F,                               // Prediction model
}

impl<const X_DIM: usize, const MEAS_DIM: usize, F, G> KalmanFilter<X_DIM, MEAS_DIM, F, G>
where
    F: Fn(f64, &SVector<f64, X_DIM>) -> SMatrix<f64, X_DIM, X_DIM>,
    G: Fn(f64, &SVector<f64, X_DIM>) -> SMatrix<f64, X_DIM, MEAS_DIM>,
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
        let f = (self.f)(dt, &self.x);
        let g = (self.g)(dt, &self.x);
        let q = self.q.call(&g);

        // Predict state
        self.x = f * self.x;
        self.p = f * self.p * f.transpose() + q;

        // Measurement noise covariance from sd
        let r = SMatrix::from_diagonal(&z_sd.component_mul(&z_sd));
        // Kalman gain
        let s = g.transpose() * self.p * g + r;
        let k = self.p * g * s.try_inverse().unwrap_or(SMatrix::zeros());

        // Update state
        let y = z - g.tr_mul(&self.x);
        self.x = self.x + k * y;
        self.p = (SMatrix::identity() - k * g.transpose()) * self.p;
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
        measurement_model: &SMatrix<f64, X_DIM, MEAS_DIM>,
    ) -> SMatrix<f64, X_DIM, X_DIM> {
        // https://en.wikipedia.org/wiki/Kalman_filter#Example_application,_technical
        // Q = G * Gt * variance formula
        measurement_model * measurement_model.transpose() * self.variance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::{Distribution, Normal};

    #[test]
    fn test_mrua() {
        // test agains linear acceleration
        let mut diff = f64::MAX;
        let dt = 0.1;
        let steps = 1_000_000;

        // True motion: constant accel 1.0
        let true_acc = 1.;

        let mut kf = KalmanFilter::new(
            SVector::from_row_slice(&[0., 0., 0.]),
            &SVector::from_row_slice(&[1e12, 0., 0.]),
            |dt, _state| {
                SMatrix::from_row_slice(&[1.0, dt, 0.5 * dt * dt, 0.0, 1.0, dt, 0.0, 0.0, 1.0])
            },
            |_dt, _state| SMatrix::identity(),
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

    #[test]
    fn test_mcu() {
        // Test against circular motion
        let mut dist = f64::MAX;
        let mut rng = rand::rng();

        let dt = 0.1;
        let iterations = 1_000_000;
        let sd = 2.;
        let noise = Normal::new(0., sd).unwrap();

        let radious = 1.;

        let rotational_velocity = 1.;
        let mut rotation = 0.;

        let mut pos_x = 0.;
        let mut pos_y = 0.;
        let velocity: f64 = 1.;
        let acceleration = velocity.powi(2) / radious;

        let mut kf: KalmanFilter<6, 5, _, _> = KalmanFilter::new(
            SVector::from_row_slice(&[
                pos_x,
                pos_y,
                rotation,
                rotational_velocity,
                velocity,
                acceleration,
            ]),
            &SVector::from_row_slice(&[0., 0., 0., 0., 0., 0.]),
            |dt, state| {
                SMatrix::from_row_slice(&[
                    1.,
                    0.,
                    0.,
                    0.,
                    state[2].cos() * dt,
                    state[2].cos() * dt.powi(2) * 0.5,
                    0.,
                    1.,
                    0.,
                    0.,
                    state[2].sin() * dt,
                    state[2].sin() * dt.powi(2) * 0.5,
                    0.,
                    0.,
                    1.,
                    dt,
                    0.,
                    0.,
                    0.,
                    0.,
                    0.,
                    1.,
                    0.,
                    0.,
                    0.,
                    0.,
                    0.,
                    0.,
                    1.,
                    dt,
                    0.,
                    0.,
                    0.,
                    0.,
                    0.,
                    1.,
                ])
            },
            |dt, _state| {
                SMatrix::from_row_slice(&[
                    1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., dt, 0., 0., 0., 0., 1., 0.,
                    0., 0., 0., 0., dt, 0., 0., 0., 0., 1.,
                ])
            },
            1e-10,
        );

        for i in 0..=iterations {
            let t = dt * (i as f64);
            rotation = rotational_velocity * t;
            pos_x = rotation.cos() * radious;
            pos_y = rotation.sin() * radious;

            let meas = SVector::from_row_slice(&[
                pos_x + noise.sample(&mut rng),
                pos_y + noise.sample(&mut rng),
                rotation + noise.sample(&mut rng),
                rotational_velocity + noise.sample(&mut rng),
                acceleration + noise.sample(&mut rng),
            ]);
            let meas_sd = SVector::from_row_slice(&[sd, sd, sd, sd, sd]);
            kf.predict(meas, meas_sd, dt);

            let state = kf.state();
            dist = ((state[0] - pos_x).powi(2) + (state[1] - pos_y).powi(2)).sqrt();
        }

        assert!(dist < 1.5);
    }
}
