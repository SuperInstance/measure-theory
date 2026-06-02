//! Measures: positive measures, probability measures, counting measure.

use crate::sigma_algebra::{SigmaAlgebra, MeasurableSet, Point};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// A measure on a measurable space (Ω, 𝒜).
///
/// A function μ: 𝒜 → [0, ∞] satisfying:
/// 1. μ(∅) = 0
/// 2. Countable additivity: μ(⋃ Aₙ) = Σ μ(Aₙ) for disjoint Aₙ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measure {
    name: String,
    space: SigmaAlgebra,
    values: HashMap<String, f64>, // encoded set → measure value
}

/// Encode a measurable set as a stable string key.
fn encode_set(set: &MeasurableSet) -> String {
    let items: Vec<String> = set.iter().map(|p| format!("{:?}", p)).collect();
    items.join(",")
}

impl Measure {
    /// Create a new measure from explicit values on each measurable set.
    pub fn new(name: &str, space: SigmaAlgebra, values: &[(MeasurableSet, f64)]) -> Self {
        let mut map = HashMap::new();
        for (set, val) in values {
            assert!(space.contains(set), "Set must be in the sigma-algebra");
            assert!(*val >= 0.0, "Measure must be non-negative");
            map.insert(encode_set(set), *val);
        }
        // Ensure empty set has measure 0
        map.insert(encode_set(&BTreeSet::new()), 0.0);

        Measure {
            name: name.to_string(),
            space,
            values: map,
        }
    }

    /// Create a probability measure from point masses.
    pub fn probability(space: SigmaAlgebra, point_masses: &[(Point, f64)]) -> Self {
        let mut set_values: Vec<(MeasurableSet, f64)> = Vec::new();
        let _full_space = space.space().clone();
        
        // Build measure for each measurable set by summing point masses
        for set in space.sets() {
            let mut total = 0.0;
            for (point, mass) in point_masses {
                if set.contains(point) {
                    total += mass;
                }
            }
            set_values.push((set.clone(), total));
        }

        Measure::new("P", space, &set_values)
    }

    /// Create a uniform probability measure on a finite space.
    pub fn uniform(space: SigmaAlgebra) -> Self {
        let n = space.space().len() as f64;
        let point_masses: Vec<(Point, f64)> = space.space()
            .iter()
            .map(|p| (p.clone(), 1.0 / n))
            .collect();
        Self::probability(space, &point_masses)
    }

    /// Create a counting measure: μ(A) = |A|.
    pub fn counting(space: SigmaAlgebra) -> Self {
        let mut set_values: Vec<(MeasurableSet, f64)> = Vec::new();
        for set in space.sets() {
            set_values.push((set.clone(), set.len() as f64));
        }
        Measure::new("counting", space, &set_values)
    }

    /// Create a Dirac (point) measure concentrated at a single point.
    pub fn dirac(space: SigmaAlgebra, point: Point) -> Self {
        let point_masses = vec![(point, 1.0)];
        let mut m = Self::probability(space, &point_masses);
        m.name = "δ".to_string();
        m
    }

    /// Evaluate the measure of a measurable set.
    pub fn measure(&self, set: &MeasurableSet) -> f64 {
        self.values.get(&encode_set(set)).copied().unwrap_or(0.0)
    }

    /// Get the sigma-algebra.
    pub fn algebra(&self) -> &SigmaAlgebra {
        &self.space
    }

    /// Get the name of this measure.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Is this a probability measure? (μ(Ω) = 1)
    pub fn is_probability(&self) -> bool {
        (self.measure(self.space.space()) - 1.0).abs() < 1e-10
    }

    /// Is this a finite measure? (μ(Ω) < ∞)
    pub fn is_finite(&self) -> bool {
        self.measure(self.space.space()).is_finite()
    }

    /// Is this sigma-finite? (For finite spaces, same as finite.)
    pub fn is_sigma_finite(&self) -> bool {
        self.is_finite()
    }

    /// Verify measure axioms.
    pub fn verify_axioms(&self) -> bool {
        // 1. Empty set has measure 0
        if self.measure(&BTreeSet::new()).abs() > 1e-10 {
            return false;
        }
        // 2. All values are non-negative
        for val in self.values.values() {
            if *val < -1e-10 {
                return false;
            }
        }
        // 3. Additivity for disjoint sets
        let all_sets: Vec<_> = self.space.sets().iter().collect();
        for i in 0..all_sets.len() {
            for j in (i+1)..all_sets.len() {
                let inter: MeasurableSet = all_sets[i].intersection(all_sets[j]).cloned().collect();
                if inter.is_empty() {
                    // Disjoint: μ(A ∪ B) = μ(A) + μ(B)
                    let union: MeasurableSet = all_sets[i].union(all_sets[j]).cloned().collect();
                    let expected = self.measure(all_sets[i]) + self.measure(all_sets[j]);
                    if (self.measure(&union) - expected).abs() > 1e-10 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Monotonicity: if A ⊆ B then μ(A) ≤ μ(B).
    pub fn verify_monotonicity(&self) -> bool {
        let all_sets: Vec<_> = self.space.sets().iter().collect();
        for i in 0..all_sets.len() {
            for j in 0..all_sets.len() {
                if all_sets[i].is_subset(all_sets[j])
                    && self.measure(all_sets[i]) > self.measure(all_sets[j]) + 1e-10
                {
                    return false;
                }
            }
        }
        true
    }

    /// Subadditivity: μ(A ∪ B) ≤ μ(A) + μ(B).
    pub fn verify_subadditivity(&self) -> bool {
        let all_sets: Vec<_> = self.space.sets().iter().collect();
        for i in 0..all_sets.len() {
            for j in (i+1)..all_sets.len() {
                let union: MeasurableSet = all_sets[i].union(all_sets[j]).cloned().collect();
                let sum = self.measure(all_sets[i]) + self.measure(all_sets[j]);
                if self.measure(&union) > sum + 1e-10 {
                    return false;
                }
            }
        }
        true
    }
}

use std::collections::BTreeSet;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sigma_algebra::{SigmaAlgebra, Point};

    fn make_space(n: usize) -> MeasurableSet {
        (0..n).map(|i| Point::Int(i as i64)).collect()
    }

    #[test]
    fn test_uniform_probability() {
        let space = make_space(4);
        let sa = SigmaAlgebra::power_set(space);
        let mu = Measure::uniform(sa);
        assert!(mu.is_probability());
        assert!(mu.verify_axioms());
        assert!(mu.verify_monotonicity());
        
        let full = mu.algebra().space().clone();
        assert!((mu.measure(&full) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_counting_measure() {
        let space = make_space(3);
        let sa = SigmaAlgebra::power_set(space);
        let mu = Measure::counting(sa);
        assert!(mu.verify_axioms());
        
        let full = mu.algebra().space().clone();
        assert!((mu.measure(&full) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dirac_measure() {
        let space = make_space(3);
        let sa = SigmaAlgebra::power_set(space);
        let mu = Measure::dirac(sa, Point::Int(1));
        assert!(mu.is_probability());
        
        let singleton: MeasurableSet = vec![Point::Int(1)].into_iter().collect();
        assert!((mu.measure(&singleton) - 1.0).abs() < 1e-10);
        
        let other: MeasurableSet = vec![Point::Int(0)].into_iter().collect();
        assert!((mu.measure(&other)).abs() < 1e-10);
    }

    #[test]
    fn test_measure_axioms() {
        let space = make_space(4);
        let sa = SigmaAlgebra::power_set(space);
        let point_masses = vec![
            (Point::Int(0), 0.1),
            (Point::Int(1), 0.2),
            (Point::Int(2), 0.3),
            (Point::Int(3), 0.4),
        ];
        let mu = Measure::probability(sa, &point_masses);
        assert!(mu.verify_axioms());
        assert!(mu.is_probability());
        assert!(mu.is_finite());
        assert!(mu.is_sigma_finite());
    }

    #[test]
    fn test_subadditivity() {
        let space = make_space(4);
        let sa = SigmaAlgebra::power_set(space);
        let mu = Measure::uniform(sa);
        assert!(mu.verify_subadditivity());
    }

    #[test]
    fn test_monotonicity() {
        let space = make_space(5);
        let sa = SigmaAlgebra::power_set(space);
        let mu = Measure::uniform(sa);
        assert!(mu.verify_monotonicity());
    }
}
