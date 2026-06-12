#!/bin/bash
# build.sh — compile the Gelfgat TPT benchmark
# Requires: gfortran (or nvfortran), LAPACK, BLAS
# On Ubuntu: sudo apt install gfortran liblapack-dev libblas-dev

set -e

if command -v nvfortran &> /dev/null; then
    FC=nvfortran
    OMPFLAG="-mp"
    EXTRA_FLAGS=""
    echo "Using nvfortran"
else
    FC=gfortran
    OMPFLAG="-fopenmp"
    EXTRA_FLAGS="-ffree-line-length-none"
    echo "Using gfortran"
fi

OPTFLAGS="-O3 -march=native"
FFLAGS="${OPTFLAGS} ${OMPFLAG} ${EXTRA_FLAGS}"

SRCS=(
    modfv_3D.f90
    EVD_eigenvector_ag1.f90
    EVD_laptmpr_3D.f90
    MeshStretch.f90
    EVD_solver_DGEMM_3D_v2.f90
    EVD_solver_DGEMM_3D_v3.f90
    EVD_Thomas3D_dgemm_OMP_v2_time.f90
    EVD_Thomas3D_dgemm_OMP_v3_time.f90
    apply_lap_residual.f90
    TestMain_3D.f90
)

echo "Compiling..."
${FC} ${FFLAGS} "${SRCS[@]}" -llapack -lblas -o Test_TPF.run

echo ""
echo "Build succeeded: ./Test_TPF.run"
echo "Run with:  echo 4 | ./Test_TPF.run"
echo "Watch:     tail -f benchmark_results.txt"
