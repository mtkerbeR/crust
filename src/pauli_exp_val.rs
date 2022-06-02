use std::convert::TryInto;

use num_complex::Complex64;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;

// use crate::getenv_use_multiple_threads;

const LANES: usize = 8;
const PARALLEL_THRESHOLD: usize = 19;

// Based on the sum implementation in:
// https://stackoverflow.com/a/67191480/14033130
// and adjust for f64 usage
#[inline]
fn fast_sum(values: &[f64]) -> f64 {
    let chunks = values.chunks_exact(LANES);
    let remainder = chunks.remainder();

    let sum = chunks.fold([0.; LANES], |mut acc, chunk| {
        let chunk: [f64; LANES] = chunk.try_into().unwrap();
        for i in 0..LANES {
            acc[i] += chunk[i];
        }
        acc
    });
    let remainder: f64 = remainder.iter().copied().sum();

    let mut reduced = 0.;
    for val in sum {
        reduced += val;
    }
    reduced + remainder
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_sum() {
        let values = (0..4).map(|x| x as f64).collect::<Vec<_>>();
        assert_eq!(fast_sum(&values), 6.0_f64);
    }
}
