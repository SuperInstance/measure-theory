//! Measurable functions: functions between measurable spaces.

use crate::sigma_algebra::{SigmaAlgebra, MeasurableSet, Point};
use crate::measure::Measure;
use serde::{Serialize, Deserialize};


/// A measurable function f: (Ω, 𝒜) → ℝ.
/// Represented as a mapping from points to real values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurableFunction {
    values: Vec<(Point, f64)>,
    domain_algebra: SigmaAlgebra,
}

impl MeasurableFunction {
    /// Create a measurable function from point-value pairs.
    pub fn new(domain: SigmaAlgebra, values: Vec<(Point, f64)>) -> Self {
        MeasurableFunction {
            values,
            domain_algebra: domain,
        }
    }

    /// Evaluate the function at a point.
    pub fn eval(&self, point: &Point) -> Option<f64> {
        self.values.iter().find(|(p, _)| p == point).map(|(_, v)| *v)
    }

    /// Get all point-value pairs.
    pub fn values(&self) -> &[(Point, f64)] {
        &self.values
    }

    /// Check measurability: preimage of any interval should be measurable.
    /// For finite spaces with power-set sigma-algebras, all functions are measurable.
    pub fn is_measurable(&self, codomain: &SigmaAlgebra) -> bool {
        // For each distinct value, the preimage must be in the sigma-algebra
        let mut distinct_values: Vec<f64> = self.values.iter().map(|(_, v)| *v).collect();
        distinct_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        distinct_values.dedup();

        for val in &distinct_values {
            let preimage: MeasurableSet = self.values.iter()
                .filter(|(_, v)| (*v - *val).abs() < 1e-10)
                .map(|(p, _)| p.clone())
                .collect();
            if !self.domain_algebra.contains(&preimage) {
                return false;
            }
        }
        // Also check that codomain contains the range
        for val in &distinct_values {
            // Each value maps to a point - check that point is in codomain
            let as_point = Point::Int(*val as i64);
            if !codomain.space().contains(&as_point) {
                // Not necessarily a failure for real-valued functions
            }
        }
        true
    }

    /// Preimage of a set: f⁻¹(B) = {ω : f(ω) ∈ B}.
    pub fn preimage(&self, set: &MeasurableSet) -> MeasurableSet {
        self.values.iter()
            .filter(|(_p, v)| {
                // Check if the value corresponds to a point in the set
                set.iter().any(|sp| match sp {
                    Point::Int(i) => (*i as f64 - *v).abs() < 1e-10,
                    _ => false,
                })
            })
            .map(|(p, _)| p.clone())
            .collect()
    }

    /// Preimage of an interval [a, b].
    pub fn preimage_interval(&self, a: f64, b: f64) -> MeasurableSet {
        self.values.iter()
            .filter(|(_, v)| *v >= a - 1e-10 && *v <= b + 1e-10)
            .map(|(p, _)| p.clone())
            .collect()
    }

    /// Compose with another measurable function (g ∘ f).
    pub fn compose(&self, other: &MeasurableFunction) -> Option<MeasurableFunction> {
        let mut result = Vec::new();
        for (point, val) in &self.values {
            let int_point = Point::Int(*val as i64);
            if let Some(new_val) = other.eval(&int_point) {
                result.push((point.clone(), new_val));
            }
        }
        if result.len() == self.values.len() {
            Some(MeasurableFunction::new(self.domain_algebra.clone(), result))
        } else {
            None
        }
    }

    /// Add two measurable functions pointwise.
    pub fn add(&self, other: &MeasurableFunction) -> MeasurableFunction {
        let mut result = Vec::new();
        for (point, val) in &self.values {
            if let Some(other_val) = other.eval(point) {
                result.push((point.clone(), val + other_val));
            }
        }
        MeasurableFunction::new(self.domain_algebra.clone(), result)
    }

    /// Multiply two measurable functions pointwise.
    pub fn multiply(&self, other: &MeasurableFunction) -> MeasurableFunction {
        let mut result = Vec::new();
        for (point, val) in &self.values {
            if let Some(other_val) = other.eval(point) {
                result.push((point.clone(), val * other_val));
            }
        }
        MeasurableFunction::new(self.domain_algebra.clone(), result)
    }

    /// Scale by a constant.
    pub fn scale(&self, c: f64) -> MeasurableFunction {
        let values = self.values.iter().map(|(p, v)| (p.clone(), v * c)).collect();
        MeasurableFunction::new(self.domain_algebra.clone(), values)
    }

    /// Maximum value.
    pub fn max(&self) -> f64 {
        self.values.iter().map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max)
    }

    /// Minimum value.
    pub fn min(&self) -> f64 {
        self.values.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min)
    }

    /// Supremum (same as max for finite spaces).
    pub fn sup(&self) -> f64 {
        self.max()
    }

    /// Essential supremum with respect to a measure.
    pub fn ess_sup(&self, measure: &Measure) -> f64 {
        // Smallest c such that μ({f > c}) = 0
        let mut candidates: Vec<f64> = self.values.iter().map(|(_, v)| *v).collect();
        candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        candidates.dedup();
        
        let mut result = f64::NEG_INFINITY;
        for val in &candidates {
            let level_set: MeasurableSet = self.values.iter()
                .filter(|(_, v)| *v > *val + 1e-10)
                .map(|(p, _)| p.clone())
                .collect();
            if measure.measure(&level_set) > 1e-10 {
                result = result.max(*val);
            }
        }
        if result == f64::NEG_INFINITY {
            self.max()
        } else {
            result
        }
    }

    /// Get the domain algebra.
    pub fn algebra(&self) -> &SigmaAlgebra {
        &self.domain_algebra
    }

    /// Create |f| (absolute value of the function).
    pub fn abs_function(&self) -> MeasurableFunction {
        let values = self.values.iter()
            .map(|(p, v)| (p.clone(), v.abs()))
            .collect();
        MeasurableFunction::new(self.domain_algebra.clone(), values)
    }

    /// Create f^c (pointwise power by a constant).
    pub fn pow_const(&self, c: f64) -> MeasurableFunction {
        let values = self.values.iter()
            .map(|(p, v)| (p.clone(), v.powf(c)))
            .collect();
        MeasurableFunction::new(self.domain_algebra.clone(), values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_space(n: usize) -> MeasurableSet {
        (0..n).map(|i| Point::Int(i as i64)).collect()
    }

    #[test]
    fn test_measurable_function_basics() {
        let space = make_space(3);
        let sa = SigmaAlgebra::power_set(space);
        let f = MeasurableFunction::new(sa, vec![
            (Point::Int(0), 1.0),
            (Point::Int(1), 2.0),
            (Point::Int(2), 3.0),
        ]);
        assert_eq!(f.eval(&Point::Int(1)), Some(2.0));
        assert_eq!(f.max(), 3.0);
        assert_eq!(f.min(), 1.0);
    }

    #[test]
    fn test_preimage() {
        let space = make_space(4);
        let sa = SigmaAlgebra::power_set(space);
        let f = MeasurableFunction::new(sa, vec![
            (Point::Int(0), 1.0),
            (Point::Int(1), 2.0),
            (Point::Int(2), 1.0),
            (Point::Int(3), 2.0),
        ]);
        let target: MeasurableSet = vec![Point::Int(1)].into_iter().collect();
        let pre = f.preimage(&target);
        let expected: MeasurableSet = vec![Point::Int(0), Point::Int(2)].into_iter().collect();
        assert_eq!(pre, expected);
    }

    #[test]
    fn test_preimage_interval() {
        let space = make_space(4);
        let sa = SigmaAlgebra::power_set(space);
        let f = MeasurableFunction::new(sa, vec![
            (Point::Int(0), 1.0),
            (Point::Int(1), 2.0),
            (Point::Int(2), 3.0),
            (Point::Int(3), 4.0),
        ]);
        let pre = f.preimage_interval(2.0, 3.0);
        assert!(pre.contains(&Point::Int(1)));
        assert!(pre.contains(&Point::Int(2)));
        assert!(!pre.contains(&Point::Int(0)));
    }

    #[test]
    fn test_add_functions() {
        let space = make_space(3);
        let sa = SigmaAlgebra::power_set(space);
        let f = MeasurableFunction::new(sa.clone(), vec![
            (Point::Int(0), 1.0),
            (Point::Int(1), 2.0),
            (Point::Int(2), 3.0),
        ]);
        let g = MeasurableFunction::new(sa, vec![
            (Point::Int(0), 10.0),
            (Point::Int(1), 20.0),
            (Point::Int(2), 30.0),
        ]);
        let h = f.add(&g);
        assert_eq!(h.eval(&Point::Int(1)), Some(22.0));
    }

    #[test]
    fn test_multiply_functions() {
        let space = make_space(2);
        let sa = SigmaAlgebra::power_set(space);
        let f = MeasurableFunction::new(sa.clone(), vec![
            (Point::Int(0), 2.0),
            (Point::Int(1), 3.0),
        ]);
        let g = MeasurableFunction::new(sa, vec![
            (Point::Int(0), 4.0),
            (Point::Int(1), 5.0),
        ]);
        let h = f.multiply(&g);
        assert_eq!(h.eval(&Point::Int(0)), Some(8.0));
        assert_eq!(h.eval(&Point::Int(1)), Some(15.0));
    }
}
