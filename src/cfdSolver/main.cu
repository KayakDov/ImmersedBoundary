/**
 * @file main.cu
 * @ingroup cfd_solver
 * @brief Program entry point. Replaces every file-open/parameter-read
 * statement at the top of ConvMain_3D_Q2D.f90 with a single master
 * config file, named on the command line.
 *
 * @warning Not compile-verified: see CfdSimulation.cu's file-level
 * warning.
 */

#include <cstdio>
#include <exception>
#include "CfdSimulation.h"

/**
 * @brief Entry point.
 * @param argc Argument count.
 * @param argv argv[1] must be the path to a master JSON config file (see
 * SimulationConfig.h and example_config.json).
 * @return 0 on success; 1 on a usage or configuration error; 2 on a
 * runtime (device/solve) error.
 */
int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <config.json>\n", argc > 0 ? argv[0] : "cfdSolver");
        return 1;
    }

    SimulationConfig cfg;
    try {
        cfg = SimulationConfig::loadFromFile(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Config error: %s\n", e.what());
        return 1;
    }

    try {
        CfdSimulation<double> simulation(cfg);
        simulation.initialize();
        simulation.run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Simulation error: %s\n", e.what());
        return 2;
    }

    return 0;
}
