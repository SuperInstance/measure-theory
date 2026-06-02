//! Lebesgue integral: from simple functions to general integrable functions.

use crate::sigma_algebra::{SigmaAlgebra, MeasurableSet, Point};
use crate::measure::Measure;
use crate::measurable_function::MeasurableFunction;


/// A simple function: finite linear combination of indicator functions.
/// φ = Σ aᵢ · 𝟙_{Aᵢ} where Aᵢ are disjoint measurable sets.
#[derive(Debug, Clone)]
pub struct SimpleFunction {
    /// (coefficient, measurable set) pairs
    terms: Vec<(f64, MeasurableSet)>,
    #[allow(dead_code)]
    domain_algebra: SigmaAlgebra,
}

impl SimpleFunction {
    /// Create a simple function from terms.
    pub fn new(domain: SigmaAlgebra, terms: Vec<(f64, MeasurableSet)>) -> Self {
        for (_, set) in &terms {
            assert!(domain.contains(set), "Set must be measurable");
        }
        SimpleFunction { terms, domain_algebra: domain }
    }

    /// Evaluate the simple function at a point.
    pub fn eval(&self, point: &Point) -> f64 {
        let mut total = 0.0;
        for (coeff, set) in &self.terms {
            if set.contains(point) {
                total += coeff;
            }
        }
        total
    }

    /// Compute the Lebesgue integral of this simple function.
    /// ∫ φ dμ = Σ aᵢ · μ(Aᵢ)
    pub fn integrate(&self, measure: &Measure) -> f64 {
        let mut total = 0.0;
        for (coeff, set) in &self.terms {
            total += coeff * measure.measure(set);
        }
        total
    }

    /// Get the terms.
    pub fn terms(&self) -> &[(f64, MeasurableSet)] {
        &self.terms
    }
}

/// Compute the Lebesgue integral of a measurable function with respect to a measure.
///
/// For finite spaces, this is simply: ∫ f dμ = Σ f(ωᵢ) · μ({ωᵢ})
pub fn lebesgue_integral(f: &MeasurableFunction, measure: &Measure) -> f64 {
    let mut total = 0.0;
    for (point, value) in f.values() {
        let singleton: MeasurableSet = std::iter::once(point.clone()).collect();
        let point_measure = measure.measure(&singleton);
        total += value * point_measure;
    }
    total
}

/// Compute the Lebesgue integral of a non-negative function.
/// (For finite spaces, same as general integral.)
pub fn lebesgue_integral_nonneg(f: &MeasurableFunction, measure: &Measure) -> f64 {
    let result = lebesgue_integral(f, measure);
    result.max(0.0)
}

/// Approximate a measurable function from below by simple functions.
/// Returns a sequence of simple functions φₙ ↑ f.
pub fn simple_approximation(f: &MeasurableFunction, algebra: &SigmaAlgebra, n_levels: usize) -> Vec<SimpleFunction> {
    let mut result = Vec::new();
    let f_min = f.min();
    let f_max = f.max();
    
    for n in 1..=n_levels {
        let step = (f_max - f_min) / (n as f64);
        let mut terms = Vec::new();
        
        for k in 0..n {
            let level = f_min + step * (k as f64);
            let next_level = f_min + step * ((k + 1) as f64);
            
            // Collect points where f(x) ∈ [level, next_level)
            let set: MeasurableSet = f.values().iter()
                .filter(|(_, v)| *v >= level - 1e-10 && *v < next_level + 1e-10)
                .map(|(p, _)| p.clone())
                .collect();
            
            if !set.is_empty() {
                terms.push((level, set));
            }
        }
        
        result.push(SimpleFunction::new(algebra.clone(), terms));
    }
    
    result
}

/// Compute expectation of a measurable function with respect to a probability measure.
pub fn expectation(f: &MeasurableFunction, probability: &Measure) -> f64 {
    assert!(probability.is_probability(), "Must be a probability measure");
    lebesgue_integral(f, probability)
}

/// Compute variance: Var[f] = E[f²] - (E[f])²
pub fn variance(f: &MeasurableFunction, probability: &Measure) -> f64 {
    let ef = expectation(f, probability);
    let f_sq = f.multiply(f);
    let ef2 = expectation(&f_sq, probability);
    ef2 - ef * ef
}

/// Compute covariance: Cov[f, g] = E[fg] - E[f]E[g]
pub fn covariance(f: &MeasurableFunction, g: &MeasurableFunction, probability: &Measure) -> f64 {
    let ef = expectation(f, probability);
    let eg = expectation(g, probability);
    let fg = f.multiply(g);
    let efg = expectation(&fg, probability);
    efg - ef * eg
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_space(n: usize) -> MeasurableSet {
        (0..n).map(|i| Point::Int(i as i64)).collect()
    }

    #[test]
    fn test_simple_function_integral() {
        let space = make_space(4);
        let sa = SigmaAlgebra::power_set(space);
        let mu = Measure::counting(sa.clone());
        
        let a: MeasurableSet = vec![Point::Int(0), Point::Int(1)].into_iter().collect();
        let b: MeasurableSet = vec![Point::Int(2), Point::Int(3)].into_iter().collect();
        
        // φ = 3·𝟙_{0,1} + 5·𝟙_{2,3}
        let sf = SimpleFunction::new(sa, vec![
            (3.0, a),
            (5.0, b),
        ]);
        assert!((sf.integrate(&mu) - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_lebesgue_integral_uniform() {
        let space = make_space(4);
        let sa = SigmaAlgebra::power_set(space);
        let mu = Measure::uniform(sa);
        
        let f = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 1.0),
            (Point::Int(1), 2.0),
            (Point::Int(2), 3.0),
            (Point::Int(3), 4.0),
        ]);
        
        // E[f] = (1+2+3+4)/4 = 2.5
        let integral = lebesgue_integral(&f, &mu);
        assert!((integral - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_expectation_and_variance() {
        let space = make_space(2);
        let sa = SigmaAlgebra::power_set(space);
        let mu = Measure::uniform(sa);
        
        let f = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 0.0),
            (Point::Int(1), 1.0),
        ]);
        
        assert!((expectation(&f, &mu) - 0.5).abs() < 1e-10);
        // Var[f] = E[f²] - (E[f])² = 0.5 - 0.25 = 0.25
        assert!((variance(&f, &mu) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_covariance() {
        let space = make_space(3);
        let sa = SigmaAlgebra::power_set(space);
        let mu = Measure::uniform(sa);
        
        let f = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 1.0),
            (Point::Int(1), 2.0),
            (Point::Int(2), 3.0),
        ]);
        let g = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 3.0),
            (Point::Int(1), 2.0),
            (Point::Int(2), 1.0),
        ]);
        
        let cov = covariance(&f, &g, &mu);
        // E[f] = 2, E[g] = 2, E[fg] = (3+4+3)/3 = 10/3
        // Cov = 10/3 - 4 = -2/3
        assert!((cov - (-2.0/3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_simple_approximation() {
        let space = make_space(4);
        let sa = SigmaAlgebra::power_set(space);
        let f = MeasurableFunction::new(sa.clone(), vec![
            (Point::Int(0), 1.0),
            (Point::Int(1), 2.0),
            (Point::Int(2), 3.0),
            (Point::Int(3), 4.0),
        ]);
        
        let approx = simple_approximation(&f, &sa, 3);
        assert_eq!(approx.len(), 3);
        
        // Each approximation should have at least one term
        for sf in &approx {
            assert!(!sf.terms().is_empty());
        }
    }

    #[test]
    fn test_integral_linearity() {
        let space = make_space(3);
        let sa = SigmaAlgebra::power_set(space);
        let mu = Measure::uniform(sa);
        
        let f = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 1.0),
            (Point::Int(1), 2.0),
            (Point::Int(2), 3.0),
        ]);
        let g = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 4.0),
            (Point::Int(1), 5.0),
            (Point::Int(2), 6.0),
        ]);
        
        let f_plus_g = f.add(&g);
        let int_f = lebesgue_integral(&f, &mu);
        let int_g = lebesgue_integral(&g, &mu);
        let int_fg = lebesgue_integral(&f_plus_g, &mu);
        
        assert!((int_fg - (int_f + int_g)).abs() < 1e-10);
    }
}
