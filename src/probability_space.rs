//! Probability spaces: measure-theoretic foundations with convenience methods.
//!
//! A probability space combines a measurable space with a probability measure
//! and provides convenience methods for expectation, variance, and divergence.

use crate::sigma_algebra::{SigmaAlgebra, MeasurableSet};
use crate::measure::Measure;
use crate::measurable_function::MeasurableFunction;
use crate::lebesgue_integral::{lebesgue_integral, expectation, variance};

/// A probability space (Ω, 𝒜, P) with an optional label.
#[derive(Debug, Clone)]
pub struct ProbabilitySpace {
    label: String,
    measure: Measure,
}

impl ProbabilitySpace {
    /// Create a new probability space.
    pub fn new(label: &str, measure: Measure) -> Self {
        assert!(measure.is_probability(), "Must be a probability measure");
        ProbabilitySpace {
            label: label.to_string(),
            measure,
        }
    }

    /// Get the label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Get the probability measure.
    pub fn measure(&self) -> &Measure {
        &self.measure
    }

    /// Get the sigma-algebra.
    pub fn algebra(&self) -> &SigmaAlgebra {
        self.measure.algebra()
    }

    /// Compute the probability of an event.
    pub fn probability(&self, event: &MeasurableSet) -> f64 {
        self.measure.measure(event)
    }

    /// Compute the expected value of a random variable.
    pub fn expect(&self, f: &MeasurableFunction) -> f64 {
        expectation(f, &self.measure)
    }

    /// Compute the variance of a random variable.
    pub fn var(&self, f: &MeasurableFunction) -> f64 {
        variance(f, &self.measure)
    }

    /// Compute the integral of a function.
    pub fn integrate(&self, f: &MeasurableFunction) -> f64 {
        lebesgue_integral(f, &self.measure)
    }
}

/// Compute the total variation distance between two probability spaces.
///
/// δ(P₁, P₂) = sup_A |P₁(A) - P₂(A)|
pub fn total_variation_distance(space1: &ProbabilitySpace, space2: &ProbabilitySpace) -> f64 {
    let mut max_diff: f64 = 0.0;
    for set in space1.algebra().sets() {
        let diff = (space1.measure.measure(set) - space2.measure.measure(set)).abs();
        max_diff = max_diff.max(diff);
    }
    max_diff
}

/// Compute the Kullback-Leibler divergence D_KL(P₁ || P₂).
///
/// For discrete spaces: D_KL = Σ P₁(x) · ln(P₁(x) / P₂(x))
pub fn kl_divergence(space1: &ProbabilitySpace, space2: &ProbabilitySpace) -> f64 {
    let mut kl = 0.0;
    for point in space1.algebra().space() {
        let singleton: MeasurableSet = std::iter::once(point.clone()).collect();
        let p1 = space1.measure.measure(&singleton);
        let p2 = space2.measure.measure(&singleton);
        if p1 > 1e-14 && p2 > 1e-14 {
            kl += p1 * (p1 / p2).ln();
        }
    }
    kl
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MeasurableSet, Point, SigmaAlgebra};

    fn make_space(n: usize) -> MeasurableSet {
        (0..n).map(|i| Point::Int(i as i64)).collect()
    }

    #[test]
    fn test_probability_space_creation() {
        let sa = SigmaAlgebra::power_set(make_space(3));
        let mu = Measure::uniform(sa);
        let space = ProbabilitySpace::new("space-1", mu);
        assert_eq!(space.label(), "space-1");
        assert!(space.probability(space.algebra().space()) - 1.0 < 1e-10);
    }

    #[test]
    fn test_probability_space_expectation() {
        let sa = SigmaAlgebra::power_set(make_space(2));
        let mu = Measure::uniform(sa);
        let space = ProbabilitySpace::new("space-2", mu);
        let f = MeasurableFunction::new(space.algebra().clone(), vec![
            (Point::Int(0), 0.0),
            (Point::Int(1), 1.0),
        ]);
        assert!((space.expect(&f) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_total_variation() {
        let sa = SigmaAlgebra::power_set(make_space(2));
        let mu1 = Measure::probability(sa.clone(), &[
            (Point::Int(0), 0.5),
            (Point::Int(1), 0.5),
        ]);
        let mu2 = Measure::probability(sa, &[
            (Point::Int(0), 0.9),
            (Point::Int(1), 0.1),
        ]);
        let s1 = ProbabilitySpace::new("a", mu1);
        let s2 = ProbabilitySpace::new("b", mu2);
        let tv = total_variation_distance(&s1, &s2);
        assert!((tv - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence() {
        let sa = SigmaAlgebra::power_set(make_space(2));
        let mu1 = Measure::probability(sa.clone(), &[
            (Point::Int(0), 0.5),
            (Point::Int(1), 0.5),
        ]);
        let mu2 = Measure::probability(sa, &[
            (Point::Int(0), 0.75),
            (Point::Int(1), 0.25),
        ]);
        let s1 = ProbabilitySpace::new("a", mu1);
        let s2 = ProbabilitySpace::new("b", mu2);
        let kl = kl_divergence(&s1, &s2);
        // KL = 0.5*ln(0.5/0.75) + 0.5*ln(0.5/0.25)
        assert!(kl > 0.0, "KL divergence should be positive");
    }
}
