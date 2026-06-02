//! L^p spaces: norms, completeness, and Holder/Minkowski inequalities.

use crate::measure::Measure;
use crate::measurable_function::MeasurableFunction;
use crate::lebesgue_integral::lebesgue_integral;

/// Compute the L^p norm of a measurable function.
///
/// ‖f‖_p = (∫ |f|^p dμ)^{1/p}
pub fn lp_norm(f: &MeasurableFunction, measure: &Measure, p: f64) -> f64 {
    assert!(p >= 1.0, "p must be >= 1 for L^p norm");
    let f_abs = f.abs_function();
    let f_p = f_abs.pow_const(p);
    let integral = lebesgue_integral(&f_p, measure);
    integral.powf(1.0 / p)
}

/// Compute the L^∞ (essential supremum) norm.
pub fn linf_norm(f: &MeasurableFunction, measure: &Measure) -> f64 {
    f.ess_sup(measure)
}

/// Holder's inequality: ‖fg‖_1 ≤ ‖f‖_p · ‖g‖_q where 1/p + 1/q = 1.
pub fn holders_inequality(
    f: &MeasurableFunction,
    g: &MeasurableFunction,
    measure: &Measure,
    p: f64,
) -> (f64, bool) {
    assert!(p > 1.0, "p must be > 1 for Holder conjugate");
    let q = p / (p - 1.0);

    let fg = f.multiply(g);
    let lhs = lebesgue_integral(&fg, measure).abs();
    let rhs = lp_norm(f, measure, p) * lp_norm(g, measure, q);

    (lhs, lhs <= rhs + 1e-10)
}

/// Minkowski's inequality: ‖f + g‖_p ≤ ‖f‖_p + ‖g‖_p.
pub fn minkowski_inequality(
    f: &MeasurableFunction,
    g: &MeasurableFunction,
    measure: &Measure,
    p: f64,
) -> (f64, bool) {
    let f_plus_g = f.add(g);
    let lhs = lp_norm(&f_plus_g, measure, p);
    let rhs = lp_norm(f, measure, p) + lp_norm(g, measure, p);

    (lhs, lhs <= rhs + 1e-10)
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MeasurableSet, Point, SigmaAlgebra};

    fn make_space(n: usize) -> MeasurableSet {
        (0..n).map(|i| Point::Int(i as i64)).collect()
    }

    #[test]
    fn test_lp_norm_l1() {
        let sa = SigmaAlgebra::power_set(make_space(4));
        let mu = Measure::uniform(sa);
        let f = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 1.0),
            (Point::Int(1), -2.0),
            (Point::Int(2), 3.0),
            (Point::Int(3), -4.0),
        ]);
        let norm = lp_norm(&f, &mu, 1.0);
        // L1 = (1+2+3+4)/4 = 2.5
        assert!((norm - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_lp_norm_l2() {
        let sa = SigmaAlgebra::power_set(make_space(2));
        let mu = Measure::uniform(sa);
        let f = MeasurableFunction::new(mu.algebra().clone(), vec![
            (Point::Int(0), 3.0),
            (Point::Int(1), 4.0),
        ]);
        let norm = lp_norm(&f, &mu, 2.0);
        // (9/2 + 16/2)^(1/2) = (12.5)^(0.5) ≈ 3.535
        assert!((norm - (12.5_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_holders_inequality() {
        let sa = SigmaAlgebra::power_set(make_space(3));
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
        let (lhs, holds) = holders_inequality(&f, &g, &mu, 2.0);
        assert!(holds, "Holder's inequality should hold: {} <= bound", lhs);
    }

    #[test]
    fn test_minkowski_inequality() {
        let sa = SigmaAlgebra::power_set(make_space(3));
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
        let (lhs, holds) = minkowski_inequality(&f, &g, &mu, 2.0);
        assert!(holds, "Minkowski inequality should hold: {} <= bound", lhs);
    }
}
