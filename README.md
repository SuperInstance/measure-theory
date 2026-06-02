# measure-theory

Measure theory in Rust. Lebesgue integration and L^p spaces.

A Rust library implementing the core constructions of measure theory and probability theory as explicit, computable data structures on finite discrete sample spaces.

## Features

- **Sigma-algebras** — trivial, power set, generated, Borel (finite)
- **Measures** — probability, uniform, counting, Dirac
- **Measurable functions** — pointwise operations, preimages, essential supremum
- **Lebesgue integral** — simple and general integration, expectation, variance, covariance
- **Convergence theorems** — Monotone Convergence (MCT), Fatou's Lemma, Dominated Convergence (DCT)
- **Product measures** — product sigma-algebras, Fubini's theorem
- **Radon-Nikodym** — absolute continuity, density computation, verification
- **L^p spaces** — norms, Hölder's inequality, Minkowski's inequality
- **Probability spaces** — labeled probability spaces, total variation distance, KL divergence

## Quick Start

```toml
[dependencies]
measure-theory = "0.1"
```

```rust
use measure_theory::*;

let space: MeasurableSet = (0..4).map(|i| Point::Int(i)).collect();
let sa = SigmaAlgebra::power_set(space);
let mu = Measure::uniform(sa);
assert!(mu.is_probability());
```

## License

MIT OR Apache-2.0
