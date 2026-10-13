/**
 * @file test_config.cpp
 * @brief GoogleTest suite for SimulationConfig: round-trips a config
 * matching the real Conv.dat settings this project has used throughout
 * (100^3 grid, Time=0.1, the 0,1,1,1,0,0 GPU split, EVD_BCx=0/EVD_BCy=1/
 * EVD_BCz=1), and exercises the required-field and bad-enum-value error
 * paths.
 */

#include <gtest/gtest.h>
#include "SimulationConfig.h"
#include <cstdio>
#include <fstream>
#include <string>

namespace {

/// Writes `contents` to a uniquely-named temp file for the duration of
/// one test, removing it on destruction.
class TempJsonFile {
public:
    TempJsonFile(std::string name, const std::string& contents) : path_(std::move(name)) {
        std::ofstream f(path_);
        f << contents;
    }
    ~TempJsonFile() { std::remove(path_.c_str()); }
    const std::string& path() const { return path_; }

private:
    std::string path_;
};

constexpr const char* kRealSettingsJson = R"({
  "grid": { "Nx": 100, "Ny": 100, "Nz": 100 },
  "physics": {
    "aspect_ratio": 1.0, "width_ratio": 1.0, "grashof": 1.85185185e7,
    "biot": 0.0, "prandtl": 0.054, "hartmann": 1000.0
  },
  "numerics": {
    "eps": 0.0000001, "eps_convergence": 0.000001, "max_iterations": 9000,
    "time_step": 0.0001, "time_interval": 0.1, "print_every": 1000
  },
  "diagnostics": { "run_conservation_checks": false, "mode": "full" },
  "temperature": {
    "stationary": false,
    "boundary_x": "dirichlet", "boundary_y": "neumann", "boundary_z": "neumann"
  },
  "potential": {
    "boundary_x": "neumann", "boundary_y": "neumann", "boundary_z": "neumann"
  },
  "time_averaging": { "enabled": false, "fourier_frequencies": [1.17114, 3.32] },
  "gpu": {
    "temperature": { "device": 0, "use_thomas_algorithm": true },
    "vx":          { "device": 0, "use_thomas_algorithm": true },
    "vy":          { "device": 1, "use_thomas_algorithm": true },
    "vz":          { "device": 1, "use_thomas_algorithm": true },
    "pressure":    { "device": 0, "use_thomas_algorithm": false },
    "potential":   { "device": 1, "use_thomas_algorithm": false }
  },
  "mesh": { "stretching_factor": 10.0 },
  "restart": { "read_from_file": null, "write_checkpoint_every": 1000 },
  "output": { "directory": "." }
})";

} // namespace

TEST(SimulationConfigTest, RoundTripsRealSettings) {
    TempJsonFile file("test_real_settings.json", kRealSettingsJson);
    SimulationConfig cfg = SimulationConfig::loadFromFile(file.path());

    EXPECT_EQ(cfg.Nx, 100);
    EXPECT_EQ(cfg.Ny, 100);
    EXPECT_EQ(cfg.Nz, 100);
    EXPECT_EQ(cfg.aspectRatio, 1.0);
    EXPECT_EQ(cfg.widthRatio, 1.0);
    EXPECT_EQ(cfg.grashof, 1.85185185e7);
    EXPECT_EQ(cfg.biot, 0.0);
    EXPECT_EQ(cfg.prandtl, 0.054);
    EXPECT_EQ(cfg.hartmann, 1000.0);
    EXPECT_EQ(cfg.epsConvergence, 0.000001);
    EXPECT_EQ(cfg.maxIterations, 9000);
    EXPECT_EQ(cfg.timeStep, 0.0001);
    EXPECT_EQ(cfg.timeInterval, 0.1);
    EXPECT_EQ(cfg.printEvery, 1000);
    EXPECT_FALSE(cfg.runConservationChecks);
    EXPECT_EQ(cfg.solveMode, SolveMode::Full);
    EXPECT_FALSE(cfg.temperatureStationary);
    EXPECT_EQ(cfg.temperatureBoundaryX, BoundaryKind::Dirichlet);
    EXPECT_EQ(cfg.temperatureBoundaryY, BoundaryKind::Neumann);
    EXPECT_EQ(cfg.temperatureBoundaryZ, BoundaryKind::Neumann);
    EXPECT_EQ(cfg.potentialBoundaryX, BoundaryKind::Neumann);
    EXPECT_EQ(cfg.potentialBoundaryY, BoundaryKind::Neumann);
    EXPECT_EQ(cfg.potentialBoundaryZ, BoundaryKind::Neumann);
    EXPECT_FALSE(cfg.timeAveragingEnabled);
    ASSERT_EQ(cfg.fourierFrequencies.size(), 2u);
    EXPECT_EQ(cfg.fourierFrequencies[0], 1.17114);
    EXPECT_EQ(cfg.fourierFrequencies[1], 3.32);
    EXPECT_EQ(cfg.gpuTemperature.device, 0);
    EXPECT_TRUE(cfg.gpuTemperature.useThomasAlgorithm);
    EXPECT_EQ(cfg.gpuVx.device, 0);
    EXPECT_TRUE(cfg.gpuVx.useThomasAlgorithm);
    EXPECT_EQ(cfg.gpuVy.device, 1);
    EXPECT_TRUE(cfg.gpuVy.useThomasAlgorithm);
    EXPECT_EQ(cfg.gpuVz.device, 1);
    EXPECT_TRUE(cfg.gpuVz.useThomasAlgorithm);
    EXPECT_EQ(cfg.gpuPressure.device, 0);
    EXPECT_FALSE(cfg.gpuPressure.useThomasAlgorithm);
    EXPECT_EQ(cfg.gpuPotential.device, 1);
    EXPECT_FALSE(cfg.gpuPotential.useThomasAlgorithm);
    EXPECT_EQ(cfg.meshStretchingFactor, 10.0);
    EXPECT_FALSE(cfg.restartFile.has_value());
    EXPECT_EQ(cfg.checkpointEvery, 1000);
    EXPECT_EQ(cfg.outputDirectory, ".");
}

TEST(SimulationConfigTest, MissingRequiredSectionThrowsClearError) {
    TempJsonFile file("test_missing_grid.json",
        R"({"physics":{"aspect_ratio":1,"width_ratio":1,"grashof":1,"biot":0,"prandtl":0.05,"hartmann":1000},)"
        R"("temperature":{"boundary_x":"dirichlet","boundary_y":"dirichlet","boundary_z":"dirichlet"},)"
        R"("potential":{"boundary_x":"neumann","boundary_y":"neumann","boundary_z":"neumann"},)"
        R"("numerics":{"time_interval":0.1}})");

    try {
        SimulationConfig::loadFromFile(file.path());
        FAIL() << "expected loadFromFile to throw for a config missing the grid section";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("grid"), std::string::npos) << "error message was: " << e.what();
    }
}

TEST(SimulationConfigTest, BadBoundaryKindThrowsClearError) {
    TempJsonFile file("test_bad_bc.json",
        R"({"grid":{"Nx":10,"Ny":10,"Nz":10},)"
        R"("physics":{"aspect_ratio":1,"width_ratio":1,"grashof":1,"biot":0,"prandtl":0.05,"hartmann":1000},)"
        R"("numerics":{"time_interval":0.1},)"
        R"("temperature":{"boundary_x":"neuman","boundary_y":"dirichlet","boundary_z":"dirichlet"},)"
        R"("potential":{"boundary_x":"neumann","boundary_y":"neumann","boundary_z":"neumann"}})");

    try {
        SimulationConfig::loadFromFile(file.path());
        FAIL() << "expected loadFromFile to throw for an invalid boundary-kind string";
    } catch (const std::exception& e) {
        // The actual typo should be quoted back in the error message.
        EXPECT_NE(std::string(e.what()).find("neuman"), std::string::npos) << "error message was: " << e.what();
    }
}

TEST(SimulationConfigTest, DefaultsApplyForMinimalConfig) {
    TempJsonFile file("test_minimal.json",
        R"({"grid":{"Nx":20,"Ny":20,"Nz":20},)"
        R"("physics":{"aspect_ratio":1,"width_ratio":1,"grashof":1e7,"biot":0,"prandtl":0.054,"hartmann":1000},)"
        R"("numerics":{"time_interval":0.05},)"
        R"("temperature":{"boundary_x":"dirichlet","boundary_y":"dirichlet","boundary_z":"dirichlet"},)"
        R"("potential":{"boundary_x":"neumann","boundary_y":"neumann","boundary_z":"neumann"}})");

    SimulationConfig minimal = SimulationConfig::loadFromFile(file.path());

    EXPECT_EQ(minimal.printEvery, 1000);
    EXPECT_FALSE(minimal.runConservationChecks);
    EXPECT_EQ(minimal.gpuVy.device, 1);
    EXPECT_EQ(minimal.meshStretchingFactor, 10.0);
    EXPECT_FALSE(minimal.restartFile.has_value());
}
