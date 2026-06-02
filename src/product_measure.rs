//! Product measures: constructing measures on product spaces.

use crate::sigma_algebra::{SigmaAlgebra, MeasurableSet, Point};
use crate::measure::Measure;

/// A product measurable space (Ω₁ × Ω₂, 𝒜₁ ⊗ 𝒜₂).
#[derive(Debug, Clone)]
pub struct ProductSpace {
    algebra1: SigmaAlgebra,
    algebra2: SigmaAlgebra,
}

impl ProductSpace {
    /// Create a product measurable space from two sigma-algebras.
    pub fn new(algebra1: SigmaAlgebra, algebra2: SigmaAlgebra) -> Self {
        ProductSpace { algebra1, algebra2 }
    }

    /// Get the first component sigma-algebra.
    pub fn first_algebra(&self) -> &SigmaAlgebra {
        &self.algebra1
    }

    /// Get the second component sigma-algebra.
    pub fn second_algebra(&self) -> &SigmaAlgebra {
        &self.algebra2
    }

    /// Generate the product sigma-algebra (for finite spaces, the power set of the product).
    pub fn product_algebra(&self) -> SigmaAlgebra {
        let space1 = self.algebra1.space();
        let space2 = self.algebra2.space();
        let product_space: MeasurableSet = space1.iter()
            .flat_map(|p1| space2.iter().map(move |p2| product_point(p1, p2)))
            .collect();
        SigmaAlgebra::power_set(product_space)
    }
}

/// Encode a pair of points as a single point for the product space.
fn product_point(p1: &Point, p2: &Point) -> Point {
    Point::Str(format!("({:?},{:?})", p1, p2))
}

/// Construct the product measure μ₁ ⊗ μ₂.
///
/// For measurable rectangles A × B: (μ₁ ⊗ μ₂)(A × B) = μ₁(A) · μ₂(B).
pub fn product_measure(mu1: &Measure, mu2: &Measure) -> Measure {
    let space1 = mu1.algebra().space();
    let space2 = mu2.algebra().space();

    // Build the product space
    let product_points: Vec<(Point, Point)> = space1.iter()
        .flat_map(|p1| space2.iter().map(move |p2| (p1.clone(), p2.clone())))
        .collect();

    let product_space: MeasurableSet = product_points.iter()
        .map(|(p1, p2)| product_point(p1, p2))
        .collect();

    let sa = SigmaAlgebra::power_set(product_space);

    // Compute point masses: μ₁({x}) · μ₂({y}) for each (x, y)
    let mut set_values: Vec<(MeasurableSet, f64)> = Vec::new();
    for set in sa.sets() {
        let mut total = 0.0;
        for (p1, p2) in &product_points {
            let pp = product_point(p1, p2);
            if set.contains(&pp) {
                let s1: MeasurableSet = std::iter::once(p1.clone()).collect();
                let s2: MeasurableSet = std::iter::once(p2.clone()).collect();
                total += mu1.measure(&s1) * mu2.measure(&s2);
            }
        }
        set_values.push((set.clone(), total));
    }

    Measure::new("product", sa, &set_values)
}

/// Fubini's theorem: iterated integration.
///
/// ∫_{Ω₁×Ω₂} f d(μ₁⊗μ₂) = ∫_{Ω₁} [∫_{Ω₂} f(x,y) dμ₂(y)] dμ₁(x)
pub fn fubini_integral(
    f_values: &[(Point, Point, f64)],
    mu1: &Measure,
    mu2: &Measure,
) -> f64 {
    // For finite spaces: sum over all (x,y) pairs
    let mut total = 0.0;
    for (p1, p2, val) in f_values {
        let s1: MeasurableSet = std::iter::once((*p1).clone()).collect();
        let s2: MeasurableSet = std::iter::once((*p2).clone()).collect();
        total += val * mu1.measure(&s1) * mu2.measure(&s2);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MeasurableSet, Point, SigmaAlgebra};

    fn make_space(n: usize) -> MeasurableSet {
        (0..n).map(|i| Point::Int(i as i64)).collect()
    }

    #[test]
    fn test_product_space() {
        let sa1 = SigmaAlgebra::power_set(make_space(2));
        let sa2 = SigmaAlgebra::power_set(make_space(3));
        let ps = ProductSpace::new(sa1, sa2);
        let product_sa = ps.product_algebra();
        // 2 × 3 = 6 points, power set size = 2^6 = 64
        assert_eq!(product_sa.len(), 64);
    }

    #[test]
    fn test_product_measure_uniform() {
        let sa1 = SigmaAlgebra::power_set(make_space(2));
        let sa2 = SigmaAlgebra::power_set(make_space(3));
        let mu1 = Measure::uniform(sa1);
        let mu2 = Measure::uniform(sa2);
        let mu_prod = product_measure(&mu1, &mu2);
        assert!(mu_prod.is_probability());
    }

    #[test]
    fn test_fubini() {
        let sa1 = SigmaAlgebra::power_set(make_space(2));
        let sa2 = SigmaAlgebra::power_set(make_space(2));
        let mu1 = Measure::uniform(sa1);
        let mu2 = Measure::uniform(sa2);

        // f(x,y) = 1 everywhere → integral should be 1
        let f_values: Vec<(Point, Point, f64)> = vec![
            (Point::Int(0), Point::Int(0), 1.0),
            (Point::Int(0), Point::Int(1), 1.0),
            (Point::Int(1), Point::Int(0), 1.0),
            (Point::Int(1), Point::Int(1), 1.0),
        ];
        let result = fubini_integral(&f_values, &mu1, &mu2);
        assert!((result - 1.0).abs() < 1e-10);
    }
}
