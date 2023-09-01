//! timelag — creating time lagged data series
//!
//! Functions for creating time-lagged versions of time series data.
//!
//! ## Example
//! ```
//! # use timelag::lag_matrix;
//! let data = [1.0, 2.0, 3.0, 4.0];
//!
//! // Using infinity for padding because NaN doesn't equal itself.
//! let lag = f64::INFINITY;
//! let padding = f64::INFINITY;
//!
//! // Create three lagged versions.
//! // Use a stride of 5 for the rows, i.e. pad with one extra entry.
//! let direct = lag_matrix(&data, 3, lag, 5).unwrap();
//!
//! assert_eq!(
//!     direct,
//!     &[
//!         1.0, 2.0, 3.0, 4.0, padding, // original data
//!         lag, 1.0, 2.0, 3.0, padding, // first lag
//!         lag, lag, 1.0, 2.0, padding, // second lag
//!         lag, lag, lag, 1.0, padding, // third lag
//!     ]
//! );
//! ```

// SPDX-FileCopyrightText: 2023 Markus Mayer
// SPDX-License-Identifier: EUPL-1.2

// only enables the `doc_cfg` feature when
// the `docsrs` configuration attribute is defined
#![cfg_attr(docsrs, feature(doc_cfg))]

use std::borrow::Borrow;
use std::fmt::{Display, Formatter};

/// The prelude.
pub mod prelude {
    pub use crate::CreateLagMatrix;
}

pub trait CreateLagMatrix<T> {
    /// Create a time-lagged matrix of time series values.
    ///
    /// This function creates lagged copies of the provided data and pads them with a placeholder value.
    /// The source data is interpreted as increasing time steps with every subsequent element; as a
    /// result, earlier (lower index) elements of the source array will be retained while later (higher
    /// index) elements will be dropped with each lag. Lagged versions are prepended with the placeholder.
    ///
    /// ## Arguments
    /// * `lags` - The number of lagged versions to create.
    /// * `fill` - The value to use to fill in lagged gaps.
    /// * `stride` - The number of elements between lagged versions in the resulting vector.
    ///            If set to `0` or `data.len()`, no padding is introduced. Values larger than
    ///            `data.len()` creates padding entries set to the `fill` value.
    ///
    /// ## Returns
    /// A vector containing lagged copies of the original data, or an error.
    ///
    /// For `N` datapoints and `M` lags, the result can be interpreted as an `M×N` matrix with
    /// lagged versions along the rows. With strides `S >= N`, the resulting matrix is of shape `M×S`
    /// with an `M×N` submatrix at `0×0` and padding to the right.
    ///
    /// ## Example
    /// ```
    /// use timelag::prelude::*;
    ///
    /// let data = [1.0, 2.0, 3.0, 4.0];
    ///
    /// // Using infinity for padding because NaN doesn't equal itself.
    /// let lag = f64::INFINITY;
    /// let padding = f64::INFINITY;
    ///
    /// // Create three lagged versions.
    /// // Use a stride of 5 for the rows, i.e. pad with one extra entry.
    /// let direct = data.lag_matrix(3, lag, 5).unwrap();
    ///
    /// assert_eq!(
    ///     direct,
    ///     &[
    ///         1.0, 2.0, 3.0, 4.0, padding, // original data
    ///         lag, 1.0, 2.0, 3.0, padding, // first lag
    ///         lag, lag, 1.0, 2.0, padding, // second lag
    ///         lag, lag, lag, 1.0, padding, // third lag
    ///     ]
    /// );
    /// ```
    fn lag_matrix(&self, lags: usize, fill: T, stride: usize) -> Result<Vec<T>, LagError>;
}

impl<S, T> CreateLagMatrix<T> for S
where
    S: Borrow<[T]>,
    T: Copy,
{
    fn lag_matrix(&self, lags: usize, fill: T, stride: usize) -> Result<Vec<T>, LagError> {
        lag_matrix(self.borrow(), lags, fill, stride)
    }
}

/// Create a time-lagged matrix of time series values.
///
/// This function creates lagged copies of the provided data and pads them with a placeholder value.
/// The source data is interpreted as increasing time steps with every subsequent element; as a
/// result, earlier (lower index) elements of the source array will be retained while later (higher
/// index) elements will be dropped with each lag. Lagged versions are prepended with the placeholder.
///
/// ## Arguments
/// * `data` - The time series data to create lagged versions of.
/// * `lags` - The number of lagged versions to create.
/// * `fill` - The value to use to fill in lagged gaps.
/// * `stride` - The number of elements between lagged versions in the resulting vector.
///            If set to `0` or `data.len()`, no padding is introduced. Values larger than
///            `data.len()` creates padding entries set to the `fill` value.
///
/// ## Returns
/// A vector containing lagged copies of the original data, or an error.
///
/// For `N` datapoints and `M` lags, the result can be interpreted as an `M×N` matrix with
/// lagged versions along the rows. With strides `S >= N`, the resulting matrix is of shape `M×S`
/// with an `M×N` submatrix at `0×0` and padding to the right.
///
/// ## Example
/// ```
/// # use timelag::lag_matrix;
/// let data = [1.0, 2.0, 3.0, 4.0];
///
/// // Using infinity for padding because NaN doesn't equal itself.
/// let lag = f64::INFINITY;
/// let padding = f64::INFINITY;
///
/// // Create three lagged versions.
/// // Use a stride of 5 for the rows, i.e. pad with one extra entry.
/// let direct = lag_matrix(&data, 3, lag, 5).unwrap();
///
/// assert_eq!(
///     direct,
///     &[
///         1.0, 2.0, 3.0, 4.0, padding, // original data
///         lag, 1.0, 2.0, 3.0, padding, // first lag
///         lag, lag, 1.0, 2.0, padding, // second lag
///         lag, lag, lag, 1.0, padding, // third lag
///     ]
/// );
/// ```
pub fn lag_matrix<T: Copy>(
    data: &[T],
    lags: usize,
    fill: T,
    mut stride: usize,
) -> Result<Vec<T>, LagError> {
    if lags == 0 || data.is_empty() {
        return Ok(data.to_vec());
    }

    let data_rows = data.len();
    if lags > data_rows {
        return Err(LagError::LagExceedsValueCount);
    }

    if stride == 0 {
        stride = data_rows;
    }

    if stride < data_rows {
        return Err(LagError::InvalidStride);
    }

    let mut lagged = vec![fill; stride * (lags + 1)];
    lagged[..data.len()].copy_from_slice(data);

    for lag in 1..=lags {
        let lagged_offset = lag * stride + lag;
        let lagged_rows = data_rows - lag;
        let lagged_end = lagged_offset + lagged_rows;
        lagged[lagged_offset..lagged_end].copy_from_slice(&data[0..lagged_rows]);
    }

    Ok(lagged)
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum LagError {
    LagExceedsValueCount,
    InvalidStride,
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
            LagError::InvalidStride => {
                write!(
                    f,
                    "The specified stride value must be greater than or equal to the number of elements in the data"
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lag() {
        let data = [42.0, 40.0, 38.0, 36.0];

        let direct = lag_matrix(&data, 3, f64::INFINITY, 0).unwrap();
        let implicit = data.lag_matrix(3, f64::INFINITY, 0).unwrap();

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

    #[test]
    fn test_strided_lag_1() {
        let data = [42.0, 40.0, 38.0, 36.0];

        let direct = lag_matrix(&data, 3, f64::INFINITY, 5).unwrap();

        assert_eq!(
            direct,
            &[
                // original data
                42.0,
                40.0,
                38.0,
                36.0,
                f64::INFINITY, // padding from stride
                // first lag
                f64::INFINITY, // lag
                42.0,
                40.0,
                38.0,
                f64::INFINITY, // padding from stride
                // second lag
                f64::INFINITY, // lag
                f64::INFINITY, // lag
                42.0,
                40.0,
                f64::INFINITY, // padding from stride
                // third lag
                f64::INFINITY, // lag
                f64::INFINITY, // lag
                f64::INFINITY, // lag
                42.0,
                f64::INFINITY, // padding from stride
            ]
        );
    }

    #[test]
    fn test_strided_lag_2() {
        let data = [42.0, 40.0, 38.0, 36.0];

        let direct = lag_matrix(&data, 3, f64::INFINITY, 8).unwrap();

        assert_eq!(
            direct,
            &[
                // original data
                42.0,
                40.0,
                38.0,
                36.0,
                f64::INFINITY, // padding from stride (1st)
                f64::INFINITY, // padding from stride (2nd)
                f64::INFINITY, // padding from stride (3rd)
                f64::INFINITY, // padding from stride (4th)
                // first lag
                f64::INFINITY, // lag
                42.0,
                40.0,
                38.0,
                f64::INFINITY, // padding from stride (1st)
                f64::INFINITY, // padding from stride (2nd)
                f64::INFINITY, // padding from stride (3rd)
                f64::INFINITY, // padding from stride (4th)
                // second lag
                f64::INFINITY, // lag
                f64::INFINITY, // lag
                42.0,
                40.0,
                f64::INFINITY, // padding from stride (1st)
                f64::INFINITY, // padding from stride (2nd)
                f64::INFINITY, // padding from stride (3rd)
                f64::INFINITY, // padding from stride (4th)
                // third lag
                f64::INFINITY, // lag
                f64::INFINITY, // lag
                f64::INFINITY, // lag
                42.0,
                f64::INFINITY, // padding from stride (1st)
                f64::INFINITY, // padding from stride (2nd)
                f64::INFINITY, // padding from stride (3rd)
                f64::INFINITY, // padding from stride (4th)
            ]
        );
    }
}
