use crate::{lag_matrix, lag_matrix_2d, LagError, MatrixLayout};
use ndarray::prelude::*;
use ndarray::{Array1, OwnedRepr};

/// Provides the [`lag_matrix`](LagMatrixFromArray::lag_matrix) function for [`Array1`] and [`Array2`] types.
pub trait LagMatrixFromArray<A>
where
    A: Copy,
{
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
    fn lag_matrix(&self, lags: usize, fill: A, stride: usize) -> Result<Array2<A>, LagError>;
}

impl<A> LagMatrixFromArray<A> for Array1<A>
where
    A: Copy,
{
    fn lag_matrix(&self, lags: usize, fill: A, stride: usize) -> Result<Array2<A>, LagError> {
        if let Some(slice) = self.as_slice() {
            let lagged = lag_matrix(slice, lags, fill, stride)?;
            let series_len = slice.len();
            let actual_stride = lagged.len() / series_len;
            Ok(make_array(lags, lagged, series_len, actual_stride))
        } else {
            Err(LagError::InvalidMemoryLayout)
        }
    }
}

impl<A> LagMatrixFromArray<A> for Array2<A>
where
    A: Copy,
{
    fn lag_matrix(&self, lags: usize, fill: A, stride: usize) -> Result<Array2<A>, LagError> {
        if let Some(slice) = self.as_slice_memory_order() {
            if self.is_standard_layout() {
                let series_len = self.ncols();
                let num_series = self.nrows();
                let lagged = lag_matrix_2d(
                    slice,
                    MatrixLayout::RowMajor(series_len),
                    lags,
                    fill,
                    stride,
                )?;

                let actual_stride = stride.max(series_len);
                Ok(make_array_2d_row_major(
                    lags,
                    lagged,
                    series_len,
                    num_series,
                    actual_stride,
                ))
            } else {
                let series_len = self.nrows();
                let num_series = self.ncols();
                let lagged = lag_matrix_2d(
                    slice,
                    MatrixLayout::ColumnMajor(series_len),
                    lags,
                    fill,
                    stride,
                )?;

                let actual_stride = stride.max(series_len);
                Ok(make_array_2d_column_major(
                    lags,
                    lagged,
                    series_len,
                    num_series,
                    actual_stride,
                ))
            }
        } else {
            Err(LagError::InvalidMemoryLayout)
        }
    }
}

fn make_array<A>(
    lags: usize,
    lagged: Vec<A>,
    series_len: usize,
    actual_stride: usize,
) -> ArrayBase<OwnedRepr<A>, Ix2> {
    let array = if actual_stride == series_len {
        Array2::<A>::from_shape_vec((series_len, lags + 1), lagged).expect("the shape is valid")
    } else {
        Array2::<A>::from_shape_vec((series_len, lags + 1).strides((actual_stride, 1)), lagged)
            .expect("the shape is valid")
    };
    array
}

fn make_array_2d_row_major<A>(
    lags: usize,
    lagged: Vec<A>,
    series_len: usize,
    series_count: usize,
    actual_stride: usize,
) -> ArrayBase<OwnedRepr<A>, Ix2> {
    let array = if actual_stride == series_len {
        Array2::<A>::from_shape_vec((series_len, series_count * (lags + 1)), lagged)
            .expect("the shape is valid")
    } else {
        Array2::<A>::from_shape_vec(
            (series_count * (lags + 1), series_len).strides((actual_stride, 1)),
            lagged,
        )
        .expect("the shape is valid")
    };
    array
}

fn make_array_2d_column_major<A>(
    lags: usize,
    lagged: Vec<A>,
    series_len: usize,
    series_count: usize,
    actual_stride: usize,
) -> ArrayBase<OwnedRepr<A>, Ix2> {
    let array = if actual_stride == series_len {
        Array2::<A>::from_shape_vec((series_count * (lags + 1), series_len), lagged)
            .expect("the shape is valid")
            .reversed_axes()
    } else {
        Array2::<A>::from_shape_vec(
            (series_count * (lags + 1), series_len).strides((1, actual_stride)),
            lagged,
        )
        .expect("the shape is valid")
        .reversed_axes()
    };
    array
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[rustfmt::skip]
    fn test_lag() {
        let data = Array1::from_iter([42.0, 40.0, 38.0, 36.0]);
        let lag = f64::INFINITY;

        let array = data.lag_matrix(3, lag, 0).unwrap();

        assert_eq!(array.ncols(), 4);
        assert_eq!(array.nrows(), 4);
        assert_eq!(
            array.as_slice().unwrap(),
            &[
                42.0, 40.0, 38.0, 36.0, // original data
                 lag, 42.0, 40.0, 38.0, // first lag
                 lag,  lag, 42.0, 40.0, // second lag
                 lag,  lag,  lag, 42.0  // third lag
            ]
        );
    }

    #[test]
    #[rustfmt::skip]
    fn test_strided_lag_2() {
        let data = Array1::from_iter([42.0, 40.0, 38.0, 36.0]);
        let lag = f64::INFINITY;

        let array = data.lag_matrix(3, lag, 8).unwrap();

        assert_eq!(array.ncols(), 4);
        assert_eq!(array.nrows(), 4);
        assert_eq!(
            array.as_standard_layout().as_slice().unwrap(),
            &[
                42.0, 40.0, 38.0, 36.0, // original data
                 lag, 42.0, 40.0, 38.0, // first lag
                 lag,  lag, 42.0, 40.0, // second lag
                 lag,  lag,  lag, 42.0  // third lag
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

        let data = Array2::from_shape_vec((2, 4), data.into_iter().collect()).unwrap();

        // Using infinity for padding because NaN doesn't equal itself.
        let lag = f64::INFINITY;

        let array = data.lag_matrix(3, lag, 5).unwrap();

        assert_eq!(array.ncols(), 4);
        assert_eq!(array.nrows(), 8);
        assert_eq!(
            array.as_standard_layout().as_slice().unwrap(),
            &[
                 1.0,  2.0,  3.0,  4.0, // original data
                -1.0, -2.0, -3.0, -4.0,
                 lag,  1.0,  2.0,  3.0, // first lag
                 lag, -1.0, -2.0, -3.0,
                 lag,  lag,  1.0,  2.0, // second lag
                 lag,  lag, -1.0, -2.0,
                 lag,  lag,  lag,  1.0, // third lag
                 lag,  lag,  lag, -1.0,
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

        let data = Array2::from_shape_vec((2, 4), data.into_iter().collect())
            .unwrap()
            .reversed_axes();

        // Using infinity for padding because NaN doesn't equal itself.
        let lag = f64::INFINITY;

        let array = data.lag_matrix(3, lag, 9).unwrap();

        assert_eq!(array.ncols(), 8);
        assert_eq!(array.nrows(), 4);
        assert_eq!(
            array.as_standard_layout().as_slice().unwrap(),
            &[
                //   original
                //   |-----|    first lag
                //   |     |     |-----|    second lag
                //   |     |     |     |     |-----|    third lag
                //   |     |     |     |     |     |     |-----|
                //   ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓
                1.0, -1.0,  lag,  lag,  lag,  lag,  lag,  lag,
                2.0, -2.0,  1.0, -1.0,  lag,  lag,  lag,  lag,
                3.0, -3.0,  2.0, -2.0,  1.0, -1.0,  lag,  lag,
                4.0, -4.0,  3.0, -3.0,  2.0, -2.0,  1.0, -1.0
            ]
        );
    }
}
