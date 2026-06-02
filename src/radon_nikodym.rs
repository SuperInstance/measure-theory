//! Radon-Nikodym theorem: densities and absolute continuity.

use crate::sigma_algebra::MeasurableSet;
use crate::measure::Measure;
use crate::measurable_function::MeasurableFunction;

/// Check if measure ν is absolutely continuous with respect to μ (ν ≪ μ).
///
/// ν ≪ μ iff for every A: μ(A) = 0 ⟹ ν(A) = 0.
pub fn is_absolutely_continuous(nu: &Measure, mu: &Measure) -> bool {
    for set in mu.algebra().sets() {
        if mu.measure(set).abs() < 1e-14
            && nu.measure(set).abs() > 1e-14
        {
            return false;
        }
    }
    true
}

/// Compute the Radon-Nikodym derivative dν/dμ.
///
/// For finite discrete spaces with point masses, the density at point x is:
///   f(x) = ν({x}) / μ({x})  (when μ({x}) > 0)
///
/// Returns the density as a measurable function.
pub fn radon_nikodym_derivative(
    nu: &Measure,
    mu: &Measure,
) -> Option<MeasurableFunction> {
    if !is_absolutely_continuous(nu, mu) {
        return None;
    }

    let space = mu.algebra().space();
    let mut values = Vec::new();
    for point in space.iter() {
        let singleton: MeasurableSet = std::iter::once(point.clone()).collect();
        let mu_val = mu.measure(&singleton);
        let nu_val = nu.measure(&singleton);

        if mu_val.abs() < 1e-14 {
            // μ({x}) = 0, so ν({x}) must also be 0 (abs. cont.)
            // density is indeterminate; set to 0
            values.push((point.clone(), 0.0));
        } else {
            values.push((point.clone(), nu_val / mu_val));
        }
    }

    Some(MeasurableFunction::new(mu.algebra().clone(), values))
}

/// Verify the Radon-Nikodym theorem: ∫_A (dν/dμ) dμ = ν(A) for all measurable A.
pub fn verify_radon_nikodym(
    nu: &Measure,
    mu: &Measure,
    density: &MeasurableFunction,
) -> bool {
    for set in mu.algebra().sets() {
        // Compute ∫_A f dμ = Σ f(x) · μ({x}) for x ∈ A
        let mut integral = 0.0;
        for point in set.iter() {
            let singleton: MeasurableSet = std::iter::once(point.clone()).collect();
            if let Some(f_val) = density.eval(point) {
                integral += f_val * mu.measure(&singleton);
            }
        }
        let nu_val = nu.measure(set);
        if (integral - nu_val).abs() > 1e-8 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MeasurableSet, Point, SigmaAlgebra};

    fn make_space(n: usize) -> MeasurableSet {
        (0..n).map(|i| Point::Int(i as i64)).collect()
    }

    #[test]
    fn test_absolute_continuity() {
        let sa = SigmaAlgebra::power_set(make_space(3));
        let mu = Measure::uniform(sa.clone());
        let nu = Measure::probability(sa, &[
            (Point::Int(0), 0.2),
            (Point::Int(1), 0.3),
            (Point::Int(2), 0.5),
        ]);
        assert!(is_absolutely_continuous(&nu, &mu));
    }

    #[test]
    fn test_radon_nikodym_derivative() {
        let sa = SigmaAlgebra::power_set(make_space(3));
        let mu = Measure::uniform(sa.clone());
        let nu = Measure::probability(sa, &[
            (Point::Int(0), 0.2),
            (Point::Int(1), 0.3),
            (Point::Int(2), 0.5),
        ]);

        let density = radon_nikodym_derivative(&nu, &mu).expect("should exist");
        // Uniform gives 1/3 per point, so density = point_mass / (1/3) = 3 * point_mass
        assert!((density.eval(&Point::Int(0)).unwrap() - 0.6).abs() < 1e-10);
        assert!((density.eval(&Point::Int(1)).unwrap() - 0.9).abs() < 1e-10);
        assert!((density.eval(&Point::Int(2)).unwrap() - 1.5).abs() < 1e-10);

        assert!(verify_radon_nikodym(&nu, &mu, &density));
    }

    #[test]
    fn test_density_integrates_to_one() {
        let sa = SigmaAlgebra::power_set(make_space(4));
        let mu = Measure::uniform(sa.clone());
        let nu = Measure::probability(sa, &[
            (Point::Int(0), 0.1),
            (Point::Int(1), 0.2),
            (Point::Int(2), 0.3),
            (Point::Int(3), 0.4),
        ]);

        let density = radon_nikodym_derivative(&nu, &mu).unwrap();
        let total: f64 = mu.algebra().space().iter()
            .map(|p| {
                let singleton: MeasurableSet = std::iter::once(p.clone()).collect();
                density.eval(p).unwrap() * mu.measure(&singleton)
            })
            .sum();
        assert!((total - 1.0).abs() < 1e-10);
    }
}
