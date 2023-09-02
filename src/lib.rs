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
/// For `D` datapoints and `L` lags, the result can be interpreted as an `L×D` matrix with
/// lagged versions along the rows. With strides `S >= D`, the resulting matrix is of shape `L×S`
/// with an `L×D` submatrix at `0×0` and padding to the right.
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

/// Describes the layout of the data matrix.
pub enum MatrixLayout {
    /// Data is laid out row-wise, i.e. reach row of the matrix contains a time series
    /// and the columns represent points in time.
    ///
    /// The values represents the number of elements per row, i.e. the length of each time series.
    RowWise(usize),
    /// Data is laid out column-wise, i.e. reach column of the matrix contains a time series
    /// and the rows represent points in time.
    ///
    /// The values represents the number of elements per column, i.e. the length of each time series.
    ColumnWise(usize),
}

impl MatrixLayout {
    pub fn len(&self) -> usize {
        match self {
            MatrixLayout::RowWise(len) => *len,
            MatrixLayout::ColumnWise(len) => *len,
        }
    }
}

/// Create a time-lagged matrix of multiple time series.
///
/// This function creates lagged copies of the provided data and pads them with a placeholder value.
/// The source data is interpreted as multiple time series with increasing time steps for every
/// subsequent element along either a matrix row or colum; as a result, earlier (lower index)
/// elements of the source array will be retained while later (higher index) elements will be
/// dropped with each lag. Lagged versions are prepended with the placeholder.
///
/// ## Arguments
/// * `data_matrix` - The matrix of multiple time series data to create lagged versions of.
/// * `lags` - The number of lagged versions to create.
/// * `layout` - The matrix layout, specifying column- or row-major order and the series length.
/// * `fill` - The value to use to fill in lagged gaps.
/// * `row_stride` - The number of elements along a row of the matrix.
///            If set to `0` or `data.len()`, no padding is introduced. Values larger than
///            `data.len()` creates padding entries set to the `fill` value.
///
/// ## Returns
/// A vector containing lagged copies of the original data, or an error.
///
/// For `D` datapoints of `S` series and `L` lags in column-major order, the result can be
/// interpreted as an `D×(S·L)` matrix with different time series along the columns and
/// subsequent lags in subsequent columns. With row strides `M >= (S·L)`, the
/// resulting matrix is of shape `D×M`.
///
/// For `D` datapoints of `S` series and `L` lags in row-major order, the result can be
/// interpreted as an `(S·L)×D` matrix with different time series along the rows and
/// subsequent lags in subsequent rows. With row strides `M >= D`, the
/// resulting matrix is of shape `(S·L)×M`.
///
/// ## Example
///
/// For matrices with time series along their rows:
///
/// ```
/// # use timelag::{lag_matrix_2d, MatrixLayout};
/// let data = [
///      1.0,  2.0,  3.0,  4.0,
///     -1.0, -2.0, -3.0, -4.0
/// ];
///
/// // Using infinity for padding because NaN doesn't equal itself.
/// let lag = f64::INFINITY;
/// let padding = f64::INFINITY;
///
/// let direct = lag_matrix_2d(&data, MatrixLayout::RowWise(4), 3, lag, 5).unwrap();
///
/// assert_eq!(
///     direct,
///     &[
///          1.0,  2.0,  3.0,  4.0, padding, // original data
///         -1.0, -2.0, -3.0, -4.0, padding,
///          lag,  1.0,  2.0,  3.0, padding, // first lag
///          lag, -1.0, -2.0, -3.0, padding,
///          lag,  lag,  1.0,  2.0, padding, // second lag
///          lag,  lag, -1.0, -2.0, padding,
///          lag,  lag,  lag,  1.0, padding, // third lag
///          lag,  lag,  lag, -1.0, padding,
///     ]
/// );
/// ```
///
/// For matrices with time series along their columns:
///
/// ```
/// # use timelag::{lag_matrix_2d, MatrixLayout};
/// let data = [
///     1.0, -1.0,
///     2.0, -2.0,
///     3.0, -3.0,
///     4.0, -4.0
/// ];
///
/// // Using infinity for padding because NaN doesn't equal itself.
/// let lag = f64::INFINITY;
/// let padding = f64::INFINITY;
///
/// // Example row stride of nine: 2 time series × (1 original + 3 lags) + 1 extra padding.
/// let direct = lag_matrix_2d(&data, MatrixLayout::ColumnWise(4), 3, lag, 9).unwrap();
///
/// assert_eq!(
///     direct,
///     &[
///     //   original
///     //   |-----|    first lag
///     //   |     |     |-----|    second lag
///     //   |     |     |     |     |-----|    third lag
///     //   |     |     |     |     |     |     |-----|
///     //   ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓
///         1.0, -1.0,  lag,  lag,  lag,  lag,  lag,  lag, padding,
///         2.0, -2.0,  1.0, -1.0,  lag,  lag,  lag,  lag, padding,
///         3.0, -3.0,  2.0, -2.0,  1.0, -1.0,  lag,  lag, padding,
///         4.0, -4.0,  3.0, -3.0,  2.0, -2.0,  1.0, -1.0, padding
///     ]
/// );
/// ```
pub fn lag_matrix_2d<T: Copy>(
    data_matrix: &[T],
    layout: MatrixLayout,
    lags: usize,
    fill: T,
    mut row_stride: usize,
) -> Result<Vec<T>, LagError> {
    if lags == 0 || data_matrix.is_empty() {
        return Ok(data_matrix.to_vec());
    }

    let series_length = layout.len();
    if lags > series_length {
        return Err(LagError::LagExceedsValueCount);
    }

    let num_series = data_matrix.len() / series_length;
    if num_series * series_length != data_matrix.len() {
        return Err(LagError::InvalidLength);
    }

    if row_stride == 0 {
        row_stride = num_series * lags;
    }

    Ok(match layout {
        MatrixLayout::RowWise(_) => {
            if row_stride < series_length {
                return Err(LagError::InvalidStride);
            }

            let mut lagged = vec![fill; num_series * row_stride * (lags + 1)];
            for lag in 0..=lags {
                for s in 0..num_series {
                    let lagged_offset = lag * num_series * row_stride + s * row_stride + lag;
                    let lagged_rows = series_length - lag;
                    let lagged_end = lagged_offset + lagged_rows;

                    let data_start = s * series_length;
                    let data_end = data_start + lagged_rows;

                    lagged[lagged_offset..lagged_end]
                        .copy_from_slice(&data_matrix[data_start..data_end]);
                }
            }
            lagged
        }
        MatrixLayout::ColumnWise(_) => {
            if row_stride < num_series * lags {
                return Err(LagError::InvalidStride);
            }

            let mut lagged = vec![fill; row_stride * series_length];

            // Prepare the last valid row.
            // This row contains the last values of the original series first,
            // followed by the values before that (due to time-lagging), etc.
            // If we create a complete set of lags, the row will therefore contain
            // all original data in reverse order, followed by optional padding (due to stride).
            for lag in 0..=lags {
                let lagged_offset = (series_length - 1) * row_stride + lag * num_series;
                let lagged_end = lagged_offset + num_series;

                // Access the data in reverse order.
                let data_start = (lags - lag) * num_series;
                let data_end = (lags - lag + 1) * num_series;

                lagged[lagged_offset..lagged_end]
                    .copy_from_slice(&data_matrix[data_start..data_end]);
            }

            // For each row above, left-shift the row below by the number of series.
            for lag in 1..=lags {
                let data_start = (series_length - 1) * row_stride + lag * num_series;
                let data_end = data_start + (lags - lag + 1) * num_series;
                let lagged_offset = (series_length - lag - 1) * row_stride;
                lagged.copy_within(data_start..data_end, lagged_offset);
            }

            lagged
        }
    })
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum LagError {
    /// The number of lags is greater than the number of data points.
    LagExceedsValueCount,
    /// The row/column stride is less than the number of elements.
    InvalidStride,
    /// The number of data points does not match the row/column length specified.
    InvalidLength,
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
            LagError::InvalidLength => write!(
                f,
                "The number of data points does not match the row/column length specified"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[rustfmt::skip]
    fn test_lag() {
        let data = [42.0, 40.0, 38.0, 36.0];
        let lag = f64::INFINITY;

        let direct = lag_matrix(&data, 3, lag, 0).unwrap();
        let implicit = data.lag_matrix(3, lag, 0).unwrap();

        assert_eq!(
            direct,
            &[
                42.0, 40.0, 38.0, 36.0, // original data
                 lag, 42.0, 40.0, 38.0, // first lag
                 lag,  lag, 42.0, 40.0, // second lag
                 lag,  lag,  lag, 42.0  // third lag
            ]
        );
        assert_eq!(direct, implicit);
    }

    #[test]
    #[rustfmt::skip]
    fn test_strided_lag_1() {
        let data = [42.0, 40.0, 38.0, 36.0];
        let lag = f64::INFINITY;
        let padding = f64::INFINITY;

        let direct = lag_matrix(&data, 3, lag, 5).unwrap();

        assert_eq!(
            direct,
            &[
                42.0, 40.0, 38.0, 36.0, padding, // original data
                 lag, 42.0, 40.0, 38.0, padding, // first lag
                 lag,  lag, 42.0, 40.0, padding, // second lag
                 lag,  lag,  lag, 42.0, padding  // third lag
            ]
        );
    }

    #[test]
    #[rustfmt::skip]
    fn test_strided_lag_2() {
        let data = [42.0, 40.0, 38.0, 36.0];
        let lag = f64::INFINITY;
        let padding = f64::INFINITY;

        let direct = lag_matrix(&data, 3, lag, 8).unwrap();

        assert_eq!(
            direct,
            &[
                42.0, 40.0, 38.0, 36.0, padding, padding, padding, padding, // original data
                 lag, 42.0, 40.0, 38.0, padding, padding, padding, padding, // first lag
                 lag,  lag, 42.0, 40.0, padding, padding, padding, padding, // second lag
                 lag,  lag,  lag, 42.0, padding, padding, padding, padding  // third lag
            ]
        );
    }

    #[test]
    #[rustfmt::skip]
    fn test_lag_2d_rowwise() {
        let data = [
             1.0,  2.0,  3.0,  4.0,
            -1.0, -2.0, -3.0, -4.0
        ];

        // Using infinity for padding because NaN doesn't equal itself.
        let lag = f64::INFINITY;
        let padding = f64::INFINITY;

        let direct = lag_matrix_2d(&data, MatrixLayout::RowWise(4), 3, lag, 5).unwrap();

        assert_eq!(
            direct,
            &[
                 1.0,  2.0,  3.0,  4.0, padding, // original data
                -1.0, -2.0, -3.0, -4.0, padding,
                 lag,  1.0,  2.0,  3.0, padding, // first lag
                 lag, -1.0, -2.0, -3.0, padding,
                 lag,  lag,  1.0,  2.0, padding, // second lag
                 lag,  lag, -1.0, -2.0, padding,
                 lag,  lag,  lag,  1.0, padding, // third lag
                 lag,  lag,  lag, -1.0, padding,
            ]
        );
    }

    #[test]
    #[rustfmt::skip]
    fn test_lag_2d_columnwise() {
        let data = [
            1.0, -1.0,
            2.0, -2.0,
            3.0, -3.0,
            4.0, -4.0
        ];

        // Using infinity for padding because NaN doesn't equal itself.
        let lag = f64::INFINITY;
        let padding = f64::INFINITY;

        let direct = lag_matrix_2d(&data, MatrixLayout::ColumnWise(4), 3, lag, 9).unwrap();

        assert_eq!(
            direct,
            &[
            //   original
            //   |-----|    first lag
            //   |     |     |-----|    second lag
            //   |     |     |     |     |-----|    third lag
            //   |     |     |     |     |     |     |-----|
            //   ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓
                1.0, -1.0,  lag,  lag,  lag,  lag,  lag,  lag, padding,
                2.0, -2.0,  1.0, -1.0,  lag,  lag,  lag,  lag, padding,
                3.0, -3.0,  2.0, -2.0,  1.0, -1.0,  lag,  lag, padding,
                4.0, -4.0,  3.0, -3.0,  2.0, -2.0,  1.0, -1.0, padding
            ]
        );
    }
}
