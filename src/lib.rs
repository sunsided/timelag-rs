//! timelag — creating time-lagged time series data
//!
//! This crate provides the `lag_matrix` and related functions to create time-lagged versions of time series similar
//! to MATLAB's [`lagmatrix`](https://mathworks.com/help/econ/lagmatrix.html) for time series analysis.
//!
//! ## Crate Features
//!
//! * `ndarray` - Enables support for [ndarray](https://crates.io/crates/ndarray)'s `Array1` and `Array2` traits.
//!
//! ## Example
//!
//! For singular time series:
//!
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
//! let lagged = lag_matrix(&data, 3, lag, 5).unwrap();
//!
//! assert_eq!(
//!     lagged,
//!     &[
//!         1.0, 2.0, 3.0, 4.0, padding, // original data
//!         lag, 1.0, 2.0, 3.0, padding, // first lag
//!         lag, lag, 1.0, 2.0, padding, // second lag
//!         lag, lag, lag, 1.0, padding, // third lag
//!     ]
//! );
//! ```
//!
//! For matrices with time series along their rows:
//!
//! ```
//! # use timelag::{lag_matrix_2d, MatrixLayout};
//! let data = [
//!      1.0,  2.0,  3.0,  4.0,
//!     -1.0, -2.0, -3.0, -4.0
//! ];
//!
//! // Using infinity for padding because NaN doesn't equal itself.
//! let lag = f64::INFINITY;
//! let padding = f64::INFINITY;
//!
//! let lagged = lag_matrix_2d(&data, MatrixLayout::RowMajor(4), 3, lag, 5).unwrap();
//!
//! assert_eq!(
//!     lagged,
//!     &[
//!          1.0,  2.0,  3.0,  4.0, padding, // original data
//!         -1.0, -2.0, -3.0, -4.0, padding,
//!          lag,  1.0,  2.0,  3.0, padding, // first lag
//!          lag, -1.0, -2.0, -3.0, padding,
//!          lag,  lag,  1.0,  2.0, padding, // second lag
//!          lag,  lag, -1.0, -2.0, padding,
//!          lag,  lag,  lag,  1.0, padding, // third lag
//!          lag,  lag,  lag, -1.0, padding,
//!     ]
//! );
//! ```
//!
//! For matrices with time series along their columns:
//!
//! ```
//! # use timelag::{lag_matrix_2d, MatrixLayout};
//! let data = [
//!     1.0, -1.0,
//!     2.0, -2.0,
//!     3.0, -3.0,
//!     4.0, -4.0
//! ];
//!
//! // Using infinity for padding because NaN doesn't equal itself.
//! let lag = f64::INFINITY;
//! let padding = f64::INFINITY;
//!
//! // Example row stride of nine: 2 time series × (1 original + 3 lags) + 1 extra padding.
//! let lagged = lag_matrix_2d(&data, MatrixLayout::ColumnMajor(4), 3, lag, 9).unwrap();
//!
//! assert_eq!(
//!     lagged,
//!     &[
//!     //   original
//!     //   |-----|    first lag
//!     //   |     |     |-----|    second lag
//!     //   |     |     |     |     |-----|    third lag
//!     //   |     |     |     |     |     |     |-----|
//!     //   ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓
//!         1.0, -1.0,  lag,  lag,  lag,  lag,  lag,  lag, padding,
//!         2.0, -2.0,  1.0, -1.0,  lag,  lag,  lag,  lag, padding,
//!         3.0, -3.0,  2.0, -2.0,  1.0, -1.0,  lag,  lag, padding,
//!         4.0, -4.0,  3.0, -3.0,  2.0, -2.0,  1.0, -1.0, padding
//!     ]
//! );
//! ```

// SPDX-FileCopyrightText: 2023 Markus Mayer
// SPDX-License-Identifier: EUPL-1.2

// only enables the `doc_cfg` feature when
// the `docsrs` configuration attribute is defined
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
mod ndarray_support;

use std::borrow::Borrow;
use std::fmt::{Display, Formatter};
use std::ops::Deref;

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
pub use ndarray_support::LagMatrixFromArray;

/// The prelude.
pub mod prelude {
    pub use crate::CreateLagMatrix;

    #[cfg(feature = "ndarray")]
    #[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
    pub use crate::ndarray_support::LagMatrixFromArray;
}

/// A matrix of time-lagged values.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LagMatrix<T> {
    data: Vec<T>,
    num_rows: usize,
    num_cols: usize,
    series_length: usize,
    series_count: usize,
    num_lags: usize,
    row_stride: usize,
    row_major: bool,
}

impl<T> LagMatrix<T> {
    /// The number of logical rows in the matrix.
    /// This value is less than or equal to [`row_stride`].
    ///
    /// Note that the physical row length, i.e. the number of elements
    /// to skip in order to go the next row, is determined by by the [`row_stride`].
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// The number of columns in the matrix.
    pub fn num_cols(&self) -> usize {
        self.num_cols
    }

    /// The number of time series captured by the matrix.
    pub fn series_count(&self) -> usize {
        self.series_count
    }

    /// The length of each time series captured by the matrix.
    pub fn series_length(&self) -> usize {
        self.series_length
    }

    /// The number of lags represented in the matrix.
    pub fn num_lags(&self) -> usize {
        self.num_lags
    }

    /// The number of elements to skip in order to go from one
    /// row to another. This value is greater than or equal to [`num_rows`].
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// Determines whether the matrix is row-major, i.e. time series data
    /// lies along the columns and lags are produced along the columns.
    pub fn is_row_major(&self) -> bool {
        self.row_major
    }

    /// Determines whether the matrix is column-major, i.e. time series data
    /// lies along the columns and lags are produced along the rows.
    pub fn is_column_major(&self) -> bool {
        !self.row_major
    }

    /// Obtains the matrix layout.
    pub fn matrix_layout(&self) -> MatrixLayout {
        if self.row_major {
            MatrixLayout::RowMajor(self.series_length)
        } else {
            MatrixLayout::ColumnMajor(self.series_length)
        }
    }
}

impl<T> Deref for LagMatrix<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Provides the [`lag_matrix`](CreateLagMatrix::lag_matrix) and [`lag_matrix_2d`](CreateLagMatrix::lag_matrix_2d)
/// functions for slice-able copy-able types.
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
    /// let lagged = data.lag_matrix(3, lag, 5).unwrap();
    ///
    /// assert_eq!(
    ///     lagged,
    ///     &[
    ///         1.0, 2.0, 3.0, 4.0, padding, // original data
    ///         lag, 1.0, 2.0, 3.0, padding, // first lag
    ///         lag, lag, 1.0, 2.0, padding, // second lag
    ///         lag, lag, lag, 1.0, padding, // third lag
    ///     ]
    /// );
    /// ```
    fn lag_matrix(&self, lags: usize, fill: T, stride: usize) -> Result<LagMatrix<T>, LagError>;

    /// Create a time-lagged matrix of multiple time series.
    ///
    /// This function creates lagged copies of the provided data and pads them with a placeholder value.
    /// The source data is interpreted as multiple time series with increasing time steps for every
    /// subsequent element along either a matrix row or colum; as a result, earlier (lower index)
    /// elements of the source array will be retained while later (higher index) elements will be
    /// dropped with each lag. Lagged versions are prepended with the placeholder.
    ///
    /// ## Arguments
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
    /// let lagged = lag_matrix_2d(&data, MatrixLayout::RowMajor(4), 3, lag, 5).unwrap();
    ///
    /// assert_eq!(
    ///     lagged,
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
    /// let lagged = lag_matrix_2d(&data, MatrixLayout::ColumnMajor(4), 3, lag, 9).unwrap();
    ///
    /// assert_eq!(
    ///     lagged,
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
    fn lag_matrix_2d(
        &self,
        layout: MatrixLayout,
        lags: usize,
        fill: T,
        row_stride: usize,
    ) -> Result<LagMatrix<T>, LagError>;
}

impl<S, T> CreateLagMatrix<T> for S
where
    S: Borrow<[T]>,
    T: Copy,
{
    #[inline(always)]
    fn lag_matrix(&self, lags: usize, fill: T, stride: usize) -> Result<LagMatrix<T>, LagError> {
        lag_matrix(self.borrow(), lags, fill, stride)
    }

    #[inline(always)]
    fn lag_matrix_2d(
        &self,
        layout: MatrixLayout,
        lags: usize,
        fill: T,
        row_stride: usize,
    ) -> Result<LagMatrix<T>, LagError> {
        lag_matrix_2d(self.borrow(), layout, lags, fill, row_stride)
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
/// let lagged = lag_matrix(&data, 3, lag, 5).unwrap();
///
/// assert_eq!(
///     lagged,
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
) -> Result<LagMatrix<T>, LagError> {
    if lags == 0 {
        return Err(LagError::InvalidLags);
    }

    if data.is_empty() {
        return Err(LagError::EmptyData);
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

    let mut num_lags = 0;
    for lag in 1..=lags {
        num_lags += 1;
        let lagged_offset = lag * stride + lag;
        let lagged_rows = data_rows - lag;
        let lagged_end = lagged_offset + lagged_rows;
        lagged[lagged_offset..lagged_end].copy_from_slice(&data[0..lagged_rows]);
    }

    let matrix = LagMatrix {
        data: lagged,
        num_rows: data_rows,
        num_cols: num_lags + 1,
        series_length: data_rows,
        row_stride: stride,
        series_count: 1,
        num_lags: num_lags + 1, // including zero lag
        row_major: true,
    };

    Ok(matrix)
}

/// Describes the layout of the data matrix.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum MatrixLayout {
    /// Data is laid out row-wise, i.e. reach row of the matrix contains a time series
    /// and the columns represent points in time.
    ///
    /// The values represents the number of elements per row, i.e. the length of each time series.
    RowMajor(usize),
    /// Data is laid out column-wise, i.e. reach column of the matrix contains a time series
    /// and the rows represent points in time.
    ///
    /// The values represents the number of elements per column, i.e. the length of each time series.
    ColumnMajor(usize),
}

impl MatrixLayout {
    pub fn len(&self) -> usize {
        match self {
            MatrixLayout::RowMajor(len) => *len,
            MatrixLayout::ColumnMajor(len) => *len,
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
/// let lagged = lag_matrix_2d(&data, MatrixLayout::RowMajor(4), 3, lag, 5).unwrap();
///
/// assert_eq!(
///     lagged,
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
/// let lagged = lag_matrix_2d(&data, MatrixLayout::ColumnMajor(4), 3, lag, 9).unwrap();
///
/// assert_eq!(
///     lagged,
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
) -> Result<LagMatrix<T>, LagError> {
    if lags == 0 {
        return Err(LagError::InvalidLags);
    }

    if data_matrix.is_empty() {
        return Err(LagError::EmptyData);
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
        MatrixLayout::RowMajor(_) => {
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

            LagMatrix {
                data: lagged,
                num_rows: series_length,
                num_cols: num_series * (lags + 1),
                series_length,
                series_count: num_series,
                num_lags: lags + 1, // including zero-lag
                row_stride,
                row_major: true,
            }
        }
        MatrixLayout::ColumnMajor(_) => {
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

            LagMatrix {
                data: lagged,
                num_cols: num_series * (lags + 1),
                num_rows: series_length,
                series_length,
                series_count: num_series,
                num_lags: lags + 1, // including zero-lag
                row_stride,
                row_major: false,
            }
        }
    })
}

/// An error during creation of a lagged data matrix.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum LagError {
    /// Invalid or no lags were specified.
    InvalidLags,
    /// The data slice was empty.
    EmptyData,
    /// The number of lags is greater than the number of data points.
    LagExceedsValueCount,
    /// The row/column stride is less than the number of elements.
    InvalidStride,
    /// The number of data points does not match the row/column length specified.
    InvalidLength,
    /// The data is in an invalid (e.g. non-contiguous) memory layout.
    InvalidMemoryLayout,
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
            LagError::InvalidMemoryLayout => write!(
                f,
                "The data is in an invalid (e.g. non-contiguous) memory layout"
            ),
            LagError::InvalidLags => write!(f, "Invalid or no lags were specified"),
            LagError::EmptyData => write!(f, "TThe data slice was emptyt"),
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

        assert_eq!(direct.num_lags(), 4);
        assert_eq!(direct.num_rows(), 4);
        assert_eq!(direct.num_cols(), 4);
        assert_eq!(direct.row_stride(), 4);
        assert_eq!(direct.series_count(), 1);
        assert_eq!(direct.series_length(), 4);
        assert_eq!(direct.matrix_layout(), MatrixLayout::RowMajor(4));
        assert!(direct.is_row_major());

        assert_eq!(
            direct.as_ref(),
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

        assert_eq!(direct.num_lags(), 4);
        assert_eq!(direct.num_rows(), 4);
        assert_eq!(direct.num_cols(), 4);
        assert_eq!(direct.row_stride(), 5);
        assert_eq!(direct.series_count(), 1);
        assert_eq!(direct.series_length(), 4);
        assert_eq!(direct.matrix_layout(), MatrixLayout::RowMajor(4));
        assert!(direct.is_row_major());

        assert_eq!(
            direct.as_ref(),
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

        assert_eq!(direct.num_lags(), 4);
        assert_eq!(direct.num_rows(), 4);
        assert_eq!(direct.num_cols(), 4);
        assert_eq!(direct.row_stride(), 8);
        assert_eq!(direct.series_count(), 1);
        assert_eq!(direct.series_length(), 4);
        assert_eq!(direct.matrix_layout(), MatrixLayout::RowMajor(4));
        assert!(direct.is_row_major());

        assert_eq!(
            direct.as_ref(),
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

        let direct = lag_matrix_2d(&data, MatrixLayout::RowMajor(4), 3, lag, 5).unwrap();

        assert_eq!(direct.num_lags(), 4);
        assert_eq!(direct.num_rows(), 4);
        assert_eq!(direct.num_cols(), 8);
        assert_eq!(direct.row_stride(), 5);
        assert_eq!(direct.series_count(), 2);
        assert_eq!(direct.series_length(), 4);
        assert_eq!(direct.matrix_layout(), MatrixLayout::RowMajor(4));
        assert!(direct.is_row_major());

        assert_eq!(
            direct.as_ref(),
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

        let direct = lag_matrix_2d(&data, MatrixLayout::ColumnMajor(4), 3, lag, 9).unwrap();

        assert_eq!(direct.num_lags(), 4);
        assert_eq!(direct.num_rows(), 4);
        assert_eq!(direct.num_cols(), 8);
        assert_eq!(direct.row_stride(), 9);
        assert_eq!(direct.series_count(), 2);
        assert_eq!(direct.series_length(), 4);
        assert_eq!(direct.matrix_layout(), MatrixLayout::ColumnMajor(4));
        assert!(!direct.is_row_major());

        assert_eq!(
            direct.as_ref(),
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
