// SPDX-FileCopyrightText: 2023 Markus Mayer
// SPDX-License-Identifier: EUPL-1.2

// only enables the `doc_cfg` feature when
// the `docsrs` configuration attribute is defined
#![cfg_attr(docsrs, feature(doc_cfg))]

use std::borrow::Borrow;
use std::fmt::{Display, Formatter};

pub trait CreateLagMatrix<T> {
    fn lag_matrix(&self, lags: usize, fill: T) -> Result<Vec<T>, LagError>;
}

impl<S, T> CreateLagMatrix<T> for S
where
    S: Borrow<[T]>,
    T: Copy,
{
    fn lag_matrix(&self, lags: usize, fill: T) -> Result<Vec<T>, LagError> {
        lag_matrix(self.borrow(), lags, fill)
    }
}

pub fn lag_matrix<T: Copy>(data: &[T], lags: usize, fill: T) -> Result<Vec<T>, LagError> {
    if lags == 0 || data.is_empty() {
        return Ok(data.to_vec());
    }

    let data_rows = data.len();
    if lags > data_rows {
        return Err(LagError::LagExceedsValueCount);
    }

    let mut lagged = vec![fill; data_rows * (lags + 1)];
    lagged[..data.len()].copy_from_slice(data);

    for lag in 1..=lags {
        let lagged_offset = lag * data_rows + lag;
        let lagged_rows = data_rows - lag;
        let lagged_end = lagged_offset + lagged_rows;
        lagged[lagged_offset..lagged_end].copy_from_slice(&data[0..lagged_rows]);
    }

    Ok(lagged)
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum LagError {
    LagExceedsValueCount,
}

impl std::error::Error for LagError {}

impl Display for LagError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            LagError::LagExceedsValueCount => {
                write!(
                    f,
                    "The specified lag exceeds the number of values in the time series"
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let data = [42.0, 40.0, 38.0, 36.0];

        let direct = lag_matrix(&data, 3, f64::INFINITY).unwrap();
        let implicit = data.lag_matrix(3, f64::INFINITY).unwrap();

        assert_eq!(
            direct,
            &[
                // original data
                42.0,
                40.0,
                38.0,
                36.0,
                // first lag
                f64::INFINITY,
                42.0,
                40.0,
                38.0,
                // second lag
                f64::INFINITY,
                f64::INFINITY,
                42.0,
                40.0,
                // third lag
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                42.0
            ]
        );
        assert_eq!(direct, implicit);
    }
}
