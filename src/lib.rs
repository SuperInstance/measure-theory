#![deny(unsafe_code)]

//! # measure-theory
//!
//! Measure theory foundations: sigma-algebras, measures, Lebesgue integration,
//! convergence theorems, product measures, Radon-Nikodym theorem, and L^p spaces.
//!
//! This crate provides the mathematical building blocks for probability theory,
//! stochastic processes, and functional analysis.

pub mod sigma_algebra;
pub mod measure;
pub mod measurable_function;
pub mod lebesgue_integral;
pub mod convergence;
pub mod product_measure;
pub mod radon_nikodym;
pub mod lp_spaces;
pub mod probability_space;

pub use sigma_algebra::*;
pub use measure::*;
pub use measurable_function::*;
pub use lebesgue_integral::*;
pub use convergence::*;
pub use product_measure::*;
pub use radon_nikodym::*;
pub use lp_spaces::*;
pub use probability_space::*;
