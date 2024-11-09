# Changelog

All notable changes to this project will be documented in this file.
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2024-11-09

[0.5.0]: https://github.com/sunsided/timelag-rs/releases/tag/v0.5.0

### Added

- Added `Into<Box[T]>` for `TimeLag`.
- The crate is now explicitly labeled `forbid(unsafe)` unless the `unsafe` feature is enabled.

### Changed

- Upgraded `ndarray` from 0.15 to 0.16.

### Internal

- Add criterion benchmarks to the project.
- Change `Into<_>` implementations to `From<_>` for broader support.

## [0.4.0] - 2023-09-02

[0.4.0]: https://github.com/sunsided/timelag-rs/releases/tag/0.4.0

### Changed

- [#2](https://github.com/sunsided/timelag-rs/pull/2):
  Changed functions operating on slices to return a `LagMatrix` struct
  providing meta information about the generated matrix.
- [#3](https://github.com/sunsided/timelag-rs/pull/3):
  Lag indices can now be specified as any `IntoIterator<Item = usize>`, e.g. range, array or slice.

## [0.3.0] - 2023-09-02

[0.3.0]: https://github.com/sunsided/timelag-rs/releases/tag/0.3.0

### Added

- [#1](https://github.com/sunsided/timelag-rs/pull/1):
  Added support for `ndarray::Array1` and `ndarray::Array2` inputs via the `ndarray` feature.

### Changed

- Renamed `MatrixLayout::RowWise` and `MatrixLayout::ColumnWise` to
  `MatrixLayout::RowMajor` and `MatrixLayout::ColumnMajor`.

## [0.2.0] - 2023-09-02

[0.2.0]: https://github.com/sunsided/timelag-rs/releases/tag/0.2.0

### Added

- Added `lag_matrix_2d` with `MatrixLayout::RowWise` and `MatrixLayout::ColumnWise` support.

## [0.1.0] - 2023-09-01

[0.1.0]: https://github.com/sunsided/timelag-rs/releases/tag/0.1.0

## Added

- Added the `lag_matrix` function, as well as `CreateLagMatrix` trait for slice-able types of copyable data.

### Internal

- 🎉 Initial release.
