//! Sigma-algebras and measurable spaces.

use serde::{Serialize, Deserialize};
use std::collections::{BTreeSet, HashSet};
use std::hash::Hash;

/// A point in a measurable space. We support both integer and string atoms.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Point {
    Int(i64),
    Str(String),
    Real(u64), // encoded as bits for ordering
}

impl From<i64> for Point {
    fn from(v: i64) -> Self { Point::Int(v) }
}

impl From<String> for Point {
    fn from(v: String) -> Self { Point::Str(v) }
}

/// A measurable set (subset of the sample space).
pub type MeasurableSet = BTreeSet<Point>;

/// A sigma-algebra on a sample space.
///
/// A collection of subsets that:
/// 1. Contains the empty set and the full space
/// 2. Closed under complementation
/// 3. Closed under countable unions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SigmaAlgebra {
    /// The full sample space (universe).
    space: MeasurableSet,
    /// The collection of measurable sets.
    sets: HashSet<MeasurableSet>,
}

impl SigmaAlgebra {
    /// Create the trivial sigma-algebra {∅, Ω}.
    pub fn trivial(space: MeasurableSet) -> Self {
        let mut sets = HashSet::new();
        sets.insert(BTreeSet::new());
        sets.insert(space.clone());
        SigmaAlgebra { space, sets }
    }

    /// Create the power set sigma-algebra (all subsets).
    pub fn power_set(space: MeasurableSet) -> Self {
        let mut sets = HashSet::new();
        let elements: Vec<Point> = space.iter().cloned().collect();
        let n = elements.len();
        // Generate all subsets for finite spaces
        for mask in 0u64..(1u64 << n) {
            let mut subset = BTreeSet::new();
            for (i, elem) in elements.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    subset.insert(elem.clone());
                }
            }
            sets.insert(subset);
        }
        SigmaAlgebra { space, sets }
    }

    /// Generate the sigma-algebra from a collection of generating sets.
    /// For finite spaces, we compute the closure directly.
    pub fn generate(space: MeasurableSet, generators: Vec<MeasurableSet>) -> Self {
        let mut sets: HashSet<MeasurableSet> = HashSet::new();
        sets.insert(BTreeSet::new());
        sets.insert(space.clone());

        for g in generators {
            sets.insert(g);
        }

        // Iteratively close under complement and finite union
        let mut changed = true;
        while changed {
            changed = false;
            let current: Vec<MeasurableSet> = sets.iter().cloned().collect();
            for s in &current {
                // Complement
                let comp: MeasurableSet = space.difference(s).cloned().collect();
                if !sets.contains(&comp) {
                    sets.insert(comp);
                    changed = true;
                }
            }
            let current: Vec<MeasurableSet> = sets.iter().cloned().collect();
            for i in 0..current.len() {
                for j in i..current.len() {
                    let union: MeasurableSet = current[i].union(&current[j]).cloned().collect();
                    if !sets.contains(&union) {
                        sets.insert(union);
                        changed = true;
                    }
                }
            }
        }

        SigmaAlgebra { space, sets }
    }

    /// Create the Borel sigma-algebra on {0, 1, ..., n-1}.
    /// For finite discrete spaces, this is the power set.
    pub fn borel_finite(n: usize) -> Self {
        let mut space = BTreeSet::new();
        for i in 0..n {
            space.insert(Point::Int(i as i64));
        }
        Self::power_set(space)
    }

    /// Get the sample space.
    pub fn space(&self) -> &MeasurableSet {
        &self.space
    }

    /// Check if a set is measurable (in the sigma-algebra).
    pub fn contains(&self, set: &MeasurableSet) -> bool {
        self.sets.contains(set)
    }

    /// Get all measurable sets.
    pub fn sets(&self) -> &HashSet<MeasurableSet> {
        &self.sets
    }

    /// Number of measurable sets.
    pub fn len(&self) -> usize {
        self.sets.len()
    }

    /// Is the sigma-algebra empty? (Should always be false for valid algebras.)
    pub fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }

    /// Complement of a measurable set.
    pub fn complement(&self, set: &MeasurableSet) -> MeasurableSet {
        self.space.difference(set).cloned().collect()
    }

    /// Union of two measurable sets.
    pub fn union(&self, a: &MeasurableSet, b: &MeasurableSet) -> MeasurableSet {
        a.union(b).cloned().collect()
    }

    /// Intersection of two measurable sets.
    pub fn intersection(&self, a: &MeasurableSet, b: &MeasurableSet) -> MeasurableSet {
        a.intersection(b).cloned().collect()
    }

    /// Verify sigma-algebra axioms hold.
    pub fn verify_axioms(&self) -> bool {
        // 1. Empty set is in the algebra
        if !self.sets.contains(&BTreeSet::new()) {
            return false;
        }
        // 2. Full space is in the algebra
        if !self.sets.contains(&self.space) {
            return false;
        }
        // 3. Closed under complement
        for s in &self.sets {
            let comp = self.complement(s);
            if !self.sets.contains(&comp) {
                return false;
            }
        }
        // 4. Closed under finite unions (sufficient for finite algebras)
        let all_sets: Vec<_> = self.sets.iter().collect();
        for i in 0..all_sets.len() {
            for j in i..all_sets.len() {
                let u = self.union(all_sets[i], all_sets[j]);
                if !self.sets.contains(&u) {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_space(n: usize) -> MeasurableSet {
        (0..n).map(|i| Point::Int(i as i64)).collect()
    }

    #[test]
    fn test_trivial_sigma_algebra() {
        let space = make_space(3);
        let sa = SigmaAlgebra::trivial(space.clone());
        assert!(sa.verify_axioms());
        assert_eq!(sa.len(), 2);
        assert!(sa.contains(&BTreeSet::new()));
        assert!(sa.contains(&space));
    }

    #[test]
    fn test_power_set() {
        let space = make_space(3);
        let sa = SigmaAlgebra::power_set(space);
        assert!(sa.verify_axioms());
        assert_eq!(sa.len(), 8); // 2^3
    }

    #[test]
    fn test_generated_sigma_algebra() {
        let space = make_space(4);
        let gen: MeasurableSet = vec![Point::Int(0), Point::Int(1)].into_iter().collect();
        let sa = SigmaAlgebra::generate(space.clone(), vec![gen]);
        assert!(sa.verify_axioms());
        assert_eq!(sa.len(), 4); // {∅, {0,1}, {2,3}, {0,1,2,3}}
    }

    #[test]
    fn test_borel_finite() {
        let sa = SigmaAlgebra::borel_finite(3);
        assert!(sa.verify_axioms());
        assert_eq!(sa.len(), 8);
    }

    #[test]
    fn test_complement() {
        let space = make_space(3);
        let sa = SigmaAlgebra::power_set(space);
        let a: MeasurableSet = vec![Point::Int(0), Point::Int(1)].into_iter().collect();
        let comp = sa.complement(&a);
        let expected: MeasurableSet = vec![Point::Int(2)].into_iter().collect();
        assert_eq!(comp, expected);
    }

    #[test]
    fn test_de_morgan_laws() {
        let space = make_space(4);
        let sa = SigmaAlgebra::power_set(space);
        let a: MeasurableSet = vec![Point::Int(0), Point::Int(1)].into_iter().collect();
        let b: MeasurableSet = vec![Point::Int(1), Point::Int(2)].into_iter().collect();
        
        // (A ∪ B)^c = A^c ∩ B^c
        let union_comp = sa.complement(&sa.union(&a, &b));
        let inter_comp = sa.intersection(&sa.complement(&a), &sa.complement(&b));
        assert_eq!(union_comp, inter_comp);
    }
}
