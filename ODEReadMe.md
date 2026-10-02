# ODE / Lyapunov module — API reference

Covers: `TimeSequence.h`, `ODE.h`, `ODE.cpp`, `Lorenz.cuh`, `PhasePortrait3d.h`, `ODETest.cu`.

A small class hierarchy for stepping GPU-resident ODE state forward in time
and estimating the largest Lyapunov exponent of the resulting trajectory,
with an optional hook for systems under a constraint (e.g. an incompressible
velocity field). `TimeSequence` is the abstract root; `ODE` supplies a
classical RK4 stepper; `Lorenz` is a concrete worked example.

---

## `TimeSequence<Real>` — `TimeSequence.h`

Abstract base for anything that can produce a sequence of states one step
at a time, plus the general-purpose Lyapunov-exponent machinery built on
top of that single `nextPoint()` primitive.

### Handle ownership

```cpp
protected:
    virtual Handle& handle() const = 0;
```

Pure virtual, no storage in this class. Every point in a sequence depends
on the one before it, so there is never a reason to use a different
`Handle`/stream across calls on the same object — a derived class stores
exactly one `Handle` (by reference) and returns it here. Every method below
calls `handle()` internally; none of them take a `Handle` parameter.

### Stepping

```cpp
virtual void nextPoint(const Vec<Real>& currentPoint, Vec<Real> nextPoint) const = 0;
void trajectory(Mat<Real>& points) const;
void pointAt(size_t t, const Vec<Real>& initialPoint, Vec<Real> result) const;
```

- `nextPoint` — pure virtual. Computes one successor state. Precondition:
  input and destination have equal length; aliasing (`nextPoint(x, x)`)
  safety is **not** guaranteed by this base class — it depends entirely on
  the derived implementation (see `ODE::rungeKutta4` below, which *is* safe
  for aliasing regardless of `dxdt`'s complexity).
- `trajectory` — fills `points` column by column, column 0 preserved, each
  later column the successor of the previous. `void`: mutates `points` in
  place rather than returning a copy.
- `pointAt` — advances `t` steps from `initialPoint` into `result` without
  retaining intermediate states. Also `void`, same reasoning.

### Constrained-system hooks

```cpp
protected:
    virtual void norm(Vec<Real>& v, Singleton<Real>& normResult) const;   // default: plain Euclidean norm
    virtual void projection(Vec<Real>& v) const;                          // default: no-op
```

Both `protected virtual`, both with safe unconstrained defaults. A derived
class representing a constrained system (more coordinates than true degrees
of freedom — e.g. a divergence-free velocity field) overrides `projection`
to remove whatever component of `v` would violate the constraint, and
optionally overrides `norm` with a restricted norm if the representation
carries an explicitly redundant coordinate that a plain Euclidean norm
would wrongly include. Both are called internally by `lyapunovInterval`
and `largestLyapunovExponent` below; an unconstrained subclass (e.g.
`Lorenz`) never touches either.

### Lyapunov exponent estimation

```cpp
Real lyapunovInterval(Vec<Real>& x, Vec<Real>& y, Vec<Real>& v,
                       Vec<Real>& singletons2, size_t stepsPerInterval, Real epsilon) const;

Real largestLyapunovExponent(const Vec<Real>& initialPoint, Mat<Real>& buffer,
                              Vec<Real>& singletons, size_t numIntervals,
                              size_t stepsPerInterval, Real epsilon, Real dt = 1) const;
```

Implements the rescaled deviation-vector method (Hartl, *Lyapunov exponents
in constrained and unconstrained ordinary differential equations*, 2003,
Sec. II A): evolve two nearby trajectories `x`/`y`, periodically measure
their separation, rescale back to size `epsilon`, and accumulate
`log(separation / epsilon)` across `numIntervals` renormalization intervals
of `stepsPerInterval` steps each.

- `lyapunovInterval` — one renormalization interval. `v` is scratch for the
  deviation vector; `x`/`y` are advanced in place via `nextPoint`.
  `singletons2` needs 2 slots (norm result, rescale factor). Returns
  `log(d/epsilon)` for this interval — one term of Eq. (2.8) in the paper.
- `largestLyapunovExponent` — the full estimate. `buffer` needs ≥3 columns
  (x, y, deviation-vector scratch); `singletons` needs ≥2 slots. `dt` is the
  physical step size, used only to convert total RK4 steps into physical
  time for the final division; defaults to 1 for discrete maps.

---

## `ODE<Real>` — `ODE.h` / `ODE.cpp`

Extends `TimeSequence<Real>` with a classical 4-stage Runge-Kutta (RK4)
stepper for systems of the form dx/dt = f(t, x), plus the `Handle` storage
every subclass needs.

```cpp
ODE(size_t numDims, Real stepSizeScalar, Handle& hand);

virtual void dxdt(Real t, const Vec<Real>& x, Vec<Real> dst,
                   Vec<Real> addToX, const Singleton<Real> scalarForAddToX) const = 0;

void rungeKutta4(Real t, const Vec<Real>& x, Vec<Real> dst) const;
void nextPoint(const Vec<Real>& currentPoint, Vec<Real> nextPoint) const override;  // calls rungeKutta4
```

- **Constructor** — `hand` is stored by reference for the object's whole
  lifetime (see `handle()` below); caller keeps it alive at least that long.
  Allocates its own RK4 scratch (`buffers`, four step-size scalars) once,
  up front.
- **`dxdt`** — pure virtual, the only thing a concrete subclass must
  supply. Computes `dst = f(x + scalarForAddToX * addToX)`. No `Handle`
  parameter — implementations call the inherited `handle()` for their own
  kernel launches (note: as a member inherited from a template base, this
  needs `this->handle()` inside a derived class's own method body, not a
  bare `handle()`, per C++ two-phase lookup).
- **`rungeKutta4`** — the integrator itself; four `dxdt` evaluations
  combined with weights 1/6, 1/3, 1/3, 1/6 (standard RK4, not SSPRK3
  despite older comments in this codebase once saying otherwise). **Safe
  for `x` and `dst` aliasing regardless of how `dxdt` is implemented** —
  every `dxdt` call writes into private internal scratch, never into `x`
  or `dst` directly; the only place `x` and `dst` meet is the final
  elementwise combine, which is safe even in place. This holds even for a
  `dxdt` doing a full spatial stencil read across neighboring grid points.
- **`handle()`** — `protected override`, returns the stored `hand`
  reference. Shared by every `ODE` subclass; `Lorenz` does not provide its
  own.
- **Special members** — `ODE()` default construction and both
  copy/move-assignment are unavailable (not defaulted): `hand` is a
  reference, which cannot be left unbound or rebound. Copy and move
  *construction* remain available (a reference just gets rebound to the
  same target).

---

## `Lorenz<Real>` — `Lorenz.cuh`

Concrete `ODE<Real>` subclass: the classic chaotic Lorenz system, standard
parameters σ=10, ρ=28, β=8/3 (the set with a well-known largest Lyapunov
exponent, λ₁ ≈ 0.905).

```cpp
explicit Lorenz(Real stepSize, Handle& handle);   // forwards to ODE<Real>(3, stepSize, handle)

void dxdt(Real t, const Vec<Real>& x, Vec<Real> dst,
          Vec<Real> addToX, const Singleton<Real> scalarForAddToX) const override;
```

State is always length 3 (x, y, z). `dxdt` launches a single-thread kernel
computing:

```
dx/dt = 10·(y − x)
dy/dt = x·(28 − z) − y
dz/dt = x·y − (8/3)·z
```

No explicit time dependence (`t` is unused). Does not override `norm` or
`projection` — fully unconstrained.

---

## `PhasePortrait<Real>` — `PhasePortrait3d.h`

Owns a VTK window for visualizing one or more GPU-computed trajectories.
Unrelated to the `TimeSequence` hierarchy except that one `draw()` overload
accepts a `TimeSequence<Real>*` to generate a trajectory on demand.

```cpp
explicit PhasePortrait(const char* title = "Phase portrait");

void draw(const Mat<Real>& points, Handle& handle, XYZ<double> RGB = {0.15, 0.8, 1});

void draw(Mat<Real> buffer, TimeSequence<Real>* ode, XYZ<Real> start,
          Handle& handle, XYZ<double> RGB);

void show();   // blocking -- starts the VTK interaction loop until the window is closed
void clear();
```

- First `draw` overload — plots an already-computed trajectory matrix
  (3 rows, one point per column).
- Second overload — generates the trajectory itself: seeds `buffer`'s
  first column with `start`, calls `ode->trajectory(buffer)` (note: no
  `handle` argument to `trajectory` — `ode` uses whichever `Handle` it was
  constructed with), then draws the result. `handle` here is only used for
  the GPU→CPU transfer in `draw`, not for stepping `ode`.
- `show()` blocks the calling thread until the window is closed by a
  human. Not safe to call unconditionally inside an automated test suite
  — see `ODETrajectory.CreatesLorenzCurve` below for the pattern this
  codebase uses to avoid hanging CI.
- `clear()` removes all drawn curves without closing the window.

---

## Usage examples — `ODETest.cu`

### `ODETrajectory.CreatesLorenzCurve`

Smoke test: builds two Lorenz trajectories from nearby initial conditions
and draws them. `portrait.show()` is gated behind an environment variable:

```cpp
const char* showPortrait = std::getenv("SHOW_LORENZ_PORTRAIT");
if (showPortrait && std::strcmp(showPortrait, "1") == 0) {
    portrait.show();
}
```

`SHOW_LORENZ_PORTRAIT` is set to `"1"` or `"0"` by CMake's
`SHOW_LORENZ_PORTRAIT` option, but only when the test is run via `ctest` —
a raw binary run or most IDE "Run" configurations won't have it set at
all, in which case the window is skipped by default (deliberately: showing
by default would hang every later test behind a human closing the window).

### `ODELyapunov.ComputesLorenzLargestExponent`

Validates `largestLyapunovExponent` against the Lorenz system's known
λ₁ ≈ 0.9056, using `numIntervals=2500`, `stepsPerInterval=20`, `h=0.005`,
tolerance `0.08`:

```cpp
double estimatedLambda = equation.largestLyapunovExponent(
    initialPoint, buffer, singletons, numIntervals, stepsPerInterval, epsilon, h
);
handle.synch();
EXPECT_NEAR(estimatedLambda, expectedLambda, tolerance);
```

### `ODELyapunov.ConvergenceStudy`

Not part of the routine fast suite — run with
`--gtest_filter=ODELyapunov.ConvergenceStudy`. Produces three CSV tables
(written to the working directory, also printed to the test log):

| File | Varies | Holds fixed | Shows |
|---|---|---|---|
| `resolution_convergence.csv` | RK4 timestep `h` | interval duration τ, `numIntervals` | integration-accuracy convergence |
| `sample_count_convergence.csv` | `numIntervals` | `h`, `stepsPerInterval` | statistical convergence (Hartl's σ/√N) |
| `starting_point_convergence.csv` | initial point | everything else | independence from starting point |

