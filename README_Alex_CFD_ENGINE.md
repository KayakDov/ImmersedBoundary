# cfdSolver

GPU-accelerated 3D convection-diffusion CFD solver. C++/CUDA port of the
original Fortran solver (`ConvMain_3D_Q2D.f90` and its supporting files),
driven by a single JSON configuration file instead of `Conv.dat` and a set
of hardcoded Fortran constants.

## Building

The build is a single CMake project (root `CMakeLists.txt`) targeting Linux
and Windows. It auto-detects its CUDA toolkit, GPU architecture, and (on
Linux) compilers rather than hardcoding any of them, so the same commands
work unmodified on different machines. Requires CMake ≥ 3.24 and working C,
C++, CUDA, and Fortran compilers.

### Quick start

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Run from the repository root; `build/` is an out-of-source build directory
(any name works). This builds all three targets described under **Build
products**, below.

### Prerequisites

- **CUDA toolkit.** Auto-detected: configuring asks `nvidia-smi` for the
  installed driver's max supported CUDA version, searches standard install
  locations (NVIDIA HPC SDK, `/usr/local/cuda-*` on Linux, `C:/Program
  Files/NVIDIA GPU Computing Toolkit/CUDA/v*` on Windows, `$CUDA_PATH`,
  `~/cuda-*`), and picks the newest toolkit the driver supports. To pick a
  specific version instead, pass `-DCUDA_TOOLKIT_VERSION=12.9`, or point
  directly at an install with `-DCUDAToolkit_ROOT=/path/to/toolkit`.
- **GPU architecture.** Defaults to `-DCMAKE_CUDA_ARCHITECTURES=native`,
  which targets whatever GPU is physically present on the build machine (no
  hardcoded architecture to go stale on different hardware). Override with
  an explicit architecture (e.g. `-DCMAKE_CUDA_ARCHITECTURES=89`) if
  configuring on a machine with no GPU attached, such as a cluster
  login/build node.
- **C/C++/Fortran compilers.** On Linux, Intel's oneAPI compilers
  (`icx`/`icpx`/`ifx`) are auto-detected and preferred if installed under
  `/opt/intel/oneapi/compiler/` (or `$HOME/intel/oneapi/...` /
  `$ONEAPI_ROOT`); otherwise CMake's normal default compilers are used. Set
  `CMAKE_C_COMPILER` / `CMAKE_CXX_COMPILER` / `CMAKE_Fortran_COMPILER`
  explicitly to override. Windows has no such auto-detection — it uses your
  normal configured toolchain (e.g. MSVC plus an installed Fortran
  compiler). *(Linux + Intel compiler only: if configuring prints
  `libintlc.so not found`, `cfdSolver`/`run_unit_tests` may fail at runtime
  with `libimf.so: cannot open shared object file` — install the matching
  Intel oneAPI runtime, or make sure its `lib/` directory is discoverable.)*
- **GoogleTest.** Fetched automatically at configure time via
  `FetchContent` — needs network access the first time you configure.
- **VTK** *(required by default, but optional)*. A "Lorenz phase-portrait
  viewer" feature unrelated to the CFD solver itself is built into the
  library by default and requires VTK (`CommonCore`, `CommonDataModel`,
  `InteractionStyle`, `RenderingCore`, `RenderingOpenGL2`). If VTK isn't
  installed, configure with `-DBUILD_LORENZ_VIEWER=OFF` to skip it:

  ```sh
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_LORENZ_VIEWER=OFF
  cmake --build build --config Release
  ```

### Build products

| Target | What it is |
|---|---|
| `cfdSolver` | The solver executable described under **Running**, below. |
| `run_unit_tests` | GoogleTest binary: unit tests plus the `cfdSolver`-specific tests (including the golden-value regression test — see the caveat below). |
| `run_benchmarks` | Benchmark binary. Deliberately not run by `ctest` (see **Running tests**, below). |
| `CudaBandedLib` | The static library the three executables above link against. Not something you run directly. |

Build a single target with `cmake --build build --target <name>`, or
everything (the default) with no `--target`.

### Running tests

```sh
cd build
ctest --output-on-failure
```

(equivalently, `cmake --build build --target run_all_tests`). Either runs
every registered test. `run_benchmarks` is never registered with `ctest`,
so benchmark runs never show up here — run them explicitly instead:

```sh
cmake --build build --target benchmark
```

or run `build/run_benchmarks` directly (it supports normal GoogleTest flags
like `--gtest_filter=...`).

### Platform-independence caveat

Everything above is portable between Linux and Windows. One specific test
is not: the golden-value regression test
(`Tests/cfdSolver/test_golden_value.cpp`, part of `run_unit_tests`) checks
`cfdSolver`'s output against a separate Fortran reference program, and it
currently points at that program via **hardcoded absolute paths** baked
into `CMakeLists.txt`:

```cmake
CFD_FORTRAN_EXE="/home/usr/Documents/TelAvivU/Alex Code/CFDSolverModified/cmake-build-debug/CFDSolver"
CFD_FORTRAN_BASE_CONV="/home/usr/Documents/TelAvivU/Alex Code/CFDSolverModified/Conv.dat"
```

(`cfdSolver`'s own path, by contrast, is resolved automatically through
CMake and never goes stale.) On any machine other than the original
development machine, either edit those two paths in `CMakeLists.txt` to
point at your own build of the Fortran reference project, or exclude that
one test:

```sh
ctest --output-on-failure -E GoldenValueTest
```

### Installing (optional)

`cmake --install build` installs `CudaBandedLib`, its Fortran `.mod`
files, and a CMake package config, so another CMake project can consume it
with `find_package(CudaBandedLib REQUIRED)` /
`target_link_libraries(... CudaBandedLib::CudaBandedLib)`. Not needed just
to build and run `cfdSolver` yourself.

## Running

The built executable takes exactly one argument: the path to a JSON config
file.

```sh
./cfdSolver path/to/config.json
```

**Exit codes:**

| Code | Meaning |
|------|---------|
| `0` | Ran to completion successfully. |
| `1` | Usage error (wrong number of arguments) or a config error — a malformed/missing/invalid field in the JSON. The specific problem is printed to stderr as `Config error: ...`. |
| `2` | A runtime error during the simulation itself (device or solve failure). Printed to stderr as `Simulation error: ...`. |

If you run it with no arguments (or more than one), it prints a one-line
usage message to stderr and exits `1`.

Config errors are deliberately specific — a missing or malformed field
names itself (e.g. `physics.grashof is required but missing from the
config file`) rather than failing generically, so a bad config file is
easy to fix without reading the source.

## Configuration file reference

The config file is a single JSON object with the sections below. Fields
marked **required** have no default and the program will refuse to start
without them; everything else falls back to the default shown if omitted.
A full, valid example is in [`example_config.json`](./example_config.json)
and reproduced at the bottom of this section.

### `grid` — required

| Field | Type | Meaning |
|---|---|---|
| `Nx`, `Ny`, `Nz` | int, **required** | Grid resolution along x/y/z. Each must be at least 2. |

### `physics` — required

Every field here is **required** — none has a physically sensible default.

| Field | Type | Meaning |
|---|---|---|
| `aspect_ratio` | double, required | Cavity's X-extent relative to its Z (height) extent. |
| `width_ratio` | double, required | Cavity's Y-extent relative to its Z (height) extent. |
| `grashof` | double, required | Grashof number (buoyancy vs. viscous forces). |
| `biot` | double, required | Biot number (surface vs. internal thermal resistance). |
| `prandtl` | double, required | Prandtl number (momentum vs. thermal diffusivity). |
| `hartmann` | double, required | Hartmann number (magnetic vs. viscous forces). |

### `numerics`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `eps` | double | `1e-7` | Carried over from the original Fortran config for parity; not read by the current solve path. Safe to omit. |
| `eps_convergence` | double | `1e-6` | Convergence threshold — the run stops early once the max residual (velocity + temperature) drops below this. |
| `max_iterations` | int | `9000` | Hard cap on solver iterations. |
| `time_step` | double | `1e-4` | Simulation time step (Htime). |
| `run_duration` | double, **required** | — | How much *additional* simulation time this run should advance by. Not a printing interval — `print_every`/`write_checkpoint_every` control that independently. |
| `print_every` | int | `1000` | How often (in iterations) to print progress. Must be at least 1. |

### `diagnostics` — fully optional

| Field | Type | Default | Meaning |
|---|---|---|---|
| `run_conservation_checks` | bool | `false` | Re-runs the finite-difference operators an extra time per `print_every` step purely to print conservation diagnostics. Expensive — off by default. |
| `mode` | string | `"full"` | One of `"full"`, `"temperature_only"`, `"flow_only"` — which physics to advance. |

### `temperature` — required section

| Field | Type | Default | Meaning |
|---|---|---|---|
| `stationary` | bool | `false` | If `true`, drops the temperature time-derivative term (steady-state temperature). |
| `boundary_x`, `boundary_y`, `boundary_z` | string, **required** | — | Boundary condition per axis: `"dirichlet"` or `"neumann"` (lowercase, exact match — anything else is a config error). |

### `potential` — required section

| Field | Type | Default | Meaning |
|---|---|---|---|
| `boundary_x`, `boundary_y`, `boundary_z` | string, **required** | — | Same values as above (`"dirichlet"`/`"neumann"`), for the electric potential field. |

### `time_averaging` — fully optional

| Field | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | bool | `false` | Turns on Fourier time-averaging accumulation. |
| `fourier_frequencies` | array of double | `[]` (empty) | Frequencies to decompose the time-averaged signal into. Empty is valid — plain time-averaging with no Fourier decomposition. Any number of values (the original Fortran format was fixed at exactly 2). |

### `gpu` — fully optional

Six independent sub-objects, one per field: `temperature`, `vx`, `vy`,
`vz`, `pressure`, `potential`. Each is optional, and each of its own two
fields is independently optional:

| Field | Type | Meaning |
|---|---|---|
| `device` | int | Which physical GPU to use for this field (0-based — matches `nvidia-smi -L` / `CUDA_VISIBLE_DEVICES` numbering). |
| `use_thomas_algorithm` | bool | `true`: Thomas-algorithm-optimized 3D solve (faster, slightly less accurate). `false`: full 3D eigendecomposition. |

Defaults if the sub-object or field is omitted:

| Sub-object | `device` default | `use_thomas_algorithm` default |
|---|---|---|
| `temperature` | `0` | `true` |
| `vx` | `0` | `true` |
| `vy` | `1` | `true` |
| `vz` | `1` | `true` |
| `pressure` | `0` | `false` (pure Poisson solve — Thomas doesn't apply) |
| `potential` | `1` | `false` (pure Poisson solve — Thomas doesn't apply) |

### `mesh` — fully optional

| Field | Type | Default | Meaning |
|---|---|---|---|
| `stretching_factor` | double | `10.0` | Tanh mesh-stretching factor (denser cells near the walls). Must be positive. |

### `restart` — fully optional

| Field | Type | Default | Meaning |
|---|---|---|---|
| `read_from_file` | string or `null` | `null` | `null`/omitted: fresh start. A path: resume from that checkpoint file. |
| `write_checkpoint_every` | int | `1000` | How often (in iterations) to write a resumable checkpoint. Must be at least 1. |

### `output` — fully optional

| Field | Type | Default | Meaning |
|---|---|---|---|
| `directory` | string | `"."` | Where output files are written. |
| `write_serie_and_history` | bool | `true` | Writes `Serie.dat`/`History.dat` (kinetic energy / Nusselt number per iteration). Matches the original Fortran, which wrote these unconditionally — set `false` to skip the extra per-iteration reduction cost if you don't need them. |
| `write_field_snapshots` | bool | `false` | Writes full 3D field snapshots (`Tmpr.dat`, `Vx.dat`, `Vy.dat`, `Vz.dat`, `Pres.dat`, `Potential.dat`) once, at the end of the run. |

### Full example

```json
{
  "grid": { "Nx": 100, "Ny": 100, "Nz": 100 },
  "physics": {
    "aspect_ratio": 1.0, "width_ratio": 1.0, "grashof": 1.85185185e7,
    "biot": 0.0, "prandtl": 0.054, "hartmann": 1000.0
  },
  "numerics": {
    "eps_convergence": 0.000001, "max_iterations": 9000,
    "time_step": 0.0001, "run_duration": 0.1, "print_every": 1000
  },
  "diagnostics": { "run_conservation_checks": false, "mode": "full" },
  "temperature": {
    "stationary": false,
    "boundary_x": "dirichlet", "boundary_y": "neumann", "boundary_z": "neumann"
  },
  "potential": {
    "boundary_x": "dirichlet", "boundary_y": "neumann", "boundary_z": "neumann"
  },
  "time_averaging": { "enabled": false, "fourier_frequencies": [1.17114, 3.32] },
  "gpu": {
    "temperature": { "device": 0, "use_thomas_algorithm": true },
    "vx":          { "device": 0, "use_thomas_algorithm": true },
    "vy":          { "device": 0, "use_thomas_algorithm": true },
    "vz":          { "device": 0, "use_thomas_algorithm": true },
    "pressure":    { "device": 0, "use_thomas_algorithm": false },
    "potential":   { "device": 0, "use_thomas_algorithm": false }
  },
  "mesh": { "stretching_factor": 10.0 },
  "restart": { "read_from_file": null, "write_checkpoint_every": 1000 },
  "output": { "directory": "./output", "write_serie_and_history": true, "write_field_snapshots": false }
}
```

Note that this example pins every `gpu` sub-object's `device` to `0`
explicitly — that's a choice made in the example file, not the built-in
default (see the defaults table above, where `vy`/`vz`/`potential` default
to device `1`).
