//! Convergence theorems: Monotone Convergence, Dominated Convergence, Fatou's Lemma.

use crate::measure::Measure;
use crate::measurable_function::MeasurableFunction;
use crate::lebesgue_integral::lebesgue_integral;

/// Result of a convergence theorem check.
#[derive(Debug, Clone)]
pub struct ConvergenceResult {
    pub theorem: String,
    pub limit_integral: f64,
    pub integral_of_limit: f64,
    pub holds: bool,
    pub error: f64,
}

/// Monotone Convergence Theorem (MCT).
///
/// If fₙ ≥ 0 and fₙ ↑ f pointwise, then lim ∫ fₙ dμ = ∫ f dμ.
pub fn monotone_convergence(
    sequence: &[MeasurableFunction],
    limit: &MeasurableFunction,
    measure: &Measure,
) -> ConvergenceResult {
    assert!(!sequence.is_empty(), "Sequence must not be empty");
    
    // Check monotonicity: fₙ(x) ≤ fₙ₊₁(x) for all x
    for n in 0..sequence.len().saturating_sub(1) {
        for (point, val_n) in sequence[n].values() {
            if let Some(val_next) = sequence[n + 1].eval(point) {
                assert!(*val_n <= val_next + 1e-10, 
                    "Sequence must be monotone increasing");
            }
        }
    }

    let integrals: Vec<f64> = sequence.iter().map(|f| lebesgue_integral(f, measure)).collect();
    let limit_of_integrals = integrals.last().copied().unwrap_or(0.0);
    let integral_of_limit = lebesgue_integral(limit, measure);
    
    ConvergenceResult {
        theorem: "Monotone Convergence".to_string(),
        limit_integral: limit_of_integrals,
        integral_of_limit,
        holds: (limit_of_integrals - integral_of_limit).abs() < 1e-8,
        error: (limit_of_integrals - integral_of_limit).abs(),
    }
}

/// Verify MCT holds on a constructed monotone sequence.
pub fn verify_monotone_convergence(
    sequence: &[MeasurableFunction],
    limit: &MeasurableFunction,
    measure: &Measure,
) -> bool {
    monotone_convergence(sequence, limit, measure).holds
}

/// Fatou's Lemma.
///
/// For any sequence of non-negative functions fₙ:
/// ∫ liminf fₙ dμ ≤ liminf ∫ fₙ dμ
pub fn fatou_lemma(
    sequence: &[MeasurableFunction],
    liminf_function: &MeasurableFunction,
    measure: &Measure,
) -> ConvergenceResult {
    assert!(!sequence.is_empty());
    
    let integrals: Vec<f64> = sequence.iter().map(|f| lebesgue_integral(f, measure)).collect();
    
    // liminf of integrals
    let mut liminf_integrals = f64::INFINITY;
    for i in 0..integrals.len() {
        let inf_from_i = integrals[i..].iter().cloned().fold(f64::INFINITY, f64::min);
        liminf_integrals = liminf_integrals.min(inf_from_i);
    }
    
    let integral_of_liminf = lebesgue_integral(liminf_function, measure);
    
    ConvergenceResult {
        theorem: "Fatou's Lemma".to_string(),
        limit_integral: liminf_integrals,
        integral_of_limit: integral_of_liminf,
        holds: integral_of_liminf <= liminf_integrals + 1e-8,
        error: (liminf_integrals - integral_of_liminf).max(0.0),
    }
}

/// Verify Fatou's Lemma inequality holds.
pub fn verify_fatou(
    sequence: &[MeasurableFunction],
    liminf_function: &MeasurableFunction,
    measure: &Measure,
) -> bool {
    fatou_lemma(sequence, liminf_function, measure).holds
}

/// Dominated Convergence Theorem (DCT).
///
/// If fₙ → f pointwise and |fₙ| ≤ g for some integrable g, then
/// lim ∫ fₙ dμ = ∫ f dμ.
pub fn dominated_convergence(
    sequence: &[MeasurableFunction],
    limit: &MeasurableFunction,
    dominating: &MeasurableFunction,
    measure: &Measure,
) -> ConvergenceResult {
    assert!(!sequence.is_empty());
    
    // Check domination: |fₙ(x)| ≤ g(x) for all n and x
    for f_n in sequence {
        for (point, val) in f_n.values() {
            if let Some(g_val) = dominating.eval(point) {
                assert!(val.abs() <= g_val + 1e-10,
                    "Function must be dominated: |fₙ(x)| ≤ g(x)");
            }
        }
    }

    let integrals: Vec<f64> = sequence.iter().map(|f| lebesgue_integral(f, measure)).collect();
    let limit_of_integrals = integrals.last().copied().unwrap_or(0.0);
    let integral_of_limit = lebesgue_integral(limit, measure);
    
    // Also check that g is integrable
    let g_integral = lebesgue_integral(dominating, measure);
    assert!(g_integral.is_finite(), "Dominating function must be integrable");
    
    ConvergenceResult {
        theorem: "Dominated Convergence".to_string(),
        limit_integral: limit_of_integrals,
        integral_of_limit,
        holds: (limit_of_integrals - integral_of_limit).abs() < 1e-8,
        error: (limit_of_integrals - integral_of_limit).abs(),
    }
}

/// Verify DCT holds.
pub fn verify_dominated_convergence(
    sequence: &[MeasurableFunction],
    limit: &MeasurableFunction,
    dominating: &MeasurableFunction,
    measure: &Measure,
) -> bool {
    dominated_convergence(sequence, limit, dominating, measure).holds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MeasurableSet, Point, SigmaAlgebra};

    fn make_space(n: usize) -> MeasurableSet {
        (0..n).map(|i| Point::Int(i as i64)).collect()
    }

    #[test]
    fn test_monotone_convergence_increasing() {
        let space = make_space(3);
        let sa = SigmaAlgebra::power_set(space);
        let mu = Measure::uniform(sa);
        
        // f₁ ≤ f₂ ≤ f₃
        let f1 = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 1.0), (Point::Int(1), 2.0), (Point::Int(2), 3.0),
        ]);
        let f2 = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 2.0), (Point::Int(1), 3.0), (Point::Int(2), 4.0),
        ]);
        let f3 = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 3.0), (Point::Int(1), 4.0), (Point::Int(2), 5.0),
        ]);
        let limit = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 3.0), (Point::Int(1), 4.0), (Point::Int(2), 5.0),
        ]);
        
        let result = monotone_convergence(&[f1, f2, f3], &limit, &mu);
        assert!(result.holds, "MCT should hold for monotone increasing sequence");
    }

    #[test]
    fn test_mct_convergent_integrals() {
        let space = make_space(4);
        let sa = SigmaAlgebra::power_set(space);
        let mu = Measure::uniform(sa);
        
        // Build a sequence converging to f
        let f1 = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 0.25), (Point::Int(1), 0.5), (Point::Int(2), 0.75), (Point::Int(3), 1.0),
        ]);
        let f2 = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 0.5), (Point::Int(1), 0.75), (Point::Int(2), 1.0), (Point::Int(3), 1.25),
        ]);
        let f3 = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 1.0), (Point::Int(1), 1.0), (Point::Int(2), 1.0), (Point::Int(3), 1.5),
        ]);
        let limit = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 1.0), (Point::Int(1), 1.0), (Point::Int(2), 1.0), (Point::Int(3), 1.5),
        ]);
        
        let result = monotone_convergence(&[f1, f2, f3], &limit, &mu);
        assert!(result.holds);
    }

    #[test]
    fn test_fatou_lemma() {
        let space = make_space(2);
        let sa = SigmaAlgebra::power_set(space);
        let mu = Measure::uniform(sa);
        
        // Sequence where integrals go up and down
        let f1 = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 2.0), (Point::Int(1), 0.0),
        ]);
        let f2 = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 0.0), (Point::Int(1), 2.0),
        ]);
        // liminf fₙ = 0 everywhere
        let liminf = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 0.0), (Point::Int(1), 0.0),
        ]);
        
        let result = fatou_lemma(&[f1, f2], &liminf, &mu);
        assert!(result.holds, "Fatou's lemma should hold");
        // ∫ liminf f = 0 ≤ liminf ∫ fₙ = 1
    }

    #[test]
    fn test_dominated_convergence() {
        let space = make_space(3);
        let sa = SigmaAlgebra::power_set(space);
        let mu = Measure::uniform(sa);
        
        // fₙ → f, all dominated by g = 10
        let f1 = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 1.0), (Point::Int(1), 2.0), (Point::Int(2), 3.0),
        ]);
        let f2 = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 1.5), (Point::Int(1), 2.5), (Point::Int(2), 3.5),
        ]);
        let limit = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 2.0), (Point::Int(1), 3.0), (Point::Int(2), 4.0),
        ]);
        let g = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 10.0), (Point::Int(1), 10.0), (Point::Int(2), 10.0),
        ]);
        
        let result = dominated_convergence(&[f1, f2, limit.clone()], &limit, &g, &mu);
        assert!(result.holds, "DCT should hold");
    }

    #[test]
    fn test_dct_with_approximating_sequence() {
        let space = make_space(4);
        let sa = SigmaAlgebra::power_set(space);
        let mu = Measure::uniform(sa);
        
        // Sequence approximating f from below
        let f1 = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 0.5), (Point::Int(1), 1.0), (Point::Int(2), 1.5), (Point::Int(3), 2.0),
        ]);
        let f2 = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 0.8), (Point::Int(1), 1.3), (Point::Int(2), 1.8), (Point::Int(3), 2.3),
        ]);
        let limit = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 1.0), (Point::Int(1), 1.5), (Point::Int(2), 2.0), (Point::Int(3), 2.5),
        ]);
        let g = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 5.0), (Point::Int(1), 5.0), (Point::Int(2), 5.0), (Point::Int(3), 5.0),
        ]);
        
        let result = dominated_convergence(&[f1, f2, limit.clone()], &limit, &g, &mu);
        assert!(result.holds);
    }

    #[test]
    fn test_fatou_strict_inequality() {
        let space = make_space(2);
        let sa = SigmaAlgebra::power_set(space);
        let mu = Measure::uniform(sa);
        
        // "Typewriter" sequence: alternates between (2,0) and (0,2)
        let f1 = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 2.0), (Point::Int(1), 0.0),
        ]);
        let f2 = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 0.0), (Point::Int(1), 2.0),
        ]);
        // liminf fₙ = 0 pointwise
        let liminf = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 0.0), (Point::Int(1), 0.0),
        ]);
        
        let result = fatou_lemma(&[f1, f2], &liminf, &mu);
        assert!(result.holds);
        // ∫ liminf = 0 < liminf ∫ fₙ = 1 (strict inequality)
        assert!(result.integral_of_limit < result.limit_integral);
    }
}
