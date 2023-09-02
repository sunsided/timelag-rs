# Changelog

All notable changes to this project will be documented in this file.
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- Changed functions operating on slices to return a `LagMatrix` struct
  providing meta information about the generated matrix.
- Lag indices can now be specified as any `IntoIterator<Item = usize>`, e.g. range, array or slice.

## [0.3.0] - 2023-09-02

### Added

- Added support for `ndarray::Array1` inputs.
- Added support for `ndarray::Array2` inputs.

### Changed

- Renamed `MatrixLayout::RowWise` and `MatrixLayout::ColumnWise` to
  `MatrixLayout::RowMajor` and `MatrixLayout::ColumnMajor`.

## [0.2.0] - 2023-09-02

### Added

- Added `lag_matrix_2d` with `MatrixLayout::RowWise` and `MatrixLayout::ColumnWise` support.

## [0.1.0] - 2023-09-01

## Added

- Added the `lag_matrix` function, as well as `CreateLagMatrix` trait for slice-able types of copyable data.

### Internal

- 🎉 Initial release.

[0.3.0]: https://github.com/sunsided/timelag-rs/releases/tag/0.3.0
[0.2.0]: https://github.com/sunsided/timelag-rs/releases/tag/0.2.0
[0.1.0]: https://github.com/sunsided/timelag-rs/releases/tag/0.1.0
