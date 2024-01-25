use rand::distributions::Distribution;

#[derive(Clone, Copy, Debug)]
pub enum Random {
    Uniform(rand::distributions::Uniform<f64>),
    Gaussian((f64, f64, rand_distr::Normal<f64>)), // min_val, max_val, GaussianDistr
}

impl Random {
    pub fn new_uniform(min_val: f64, max_val: f64) -> Self {
        Self::Uniform(rand::distributions::Uniform::new_inclusive(
            min_val, max_val,
        ))
    }

    pub fn new_gaussian(min_val: f64, max_val: f64) -> Self {
        let mean = (min_val + max_val) / 2.0;
        let sigma = (max_val - min_val) / 6.0;

        Self::Gaussian((
            min_val,
            max_val,
            rand_distr::Normal::new(mean, sigma).expect("fail to create gaussian distribution"),
        ))
    }

    pub fn sample(&self) -> f64 {
        match self {
            Random::Uniform(s) => s.sample(&mut rand::thread_rng()),
            Random::Gaussian((min_val, max_val, s)) => {
                let mut val = s.sample(&mut rand::thread_rng());
                if val < *min_val {
                    val = *min_val
                }
                if val > *max_val {
                    val = *max_val
                }

                val
            }
        }
    }
}
