#include "SimulationConfig.h"
#include "third_party/json.hpp"
#include <fstream>
#include <stdexcept>
#include <sstream>

using json = nlohmann::json;

namespace {

/// @brief Parses "dirichlet"/"neumann" into a BoundaryKind, or throws.
BoundaryKind parseBoundaryKind(const json& j, const std::string& fieldPath) {
    if (!j.is_string())
        throw std::runtime_error(fieldPath + " must be a string: \"dirichlet\" or \"neumann\"");
    std::string s = j.get<std::string>();
    if (s == "dirichlet") return BoundaryKind::Dirichlet;
    if (s == "neumann")   return BoundaryKind::Neumann;
    throw std::runtime_error(fieldPath + " = \"" + s + "\" is not valid -- must be \"dirichlet\" or \"neumann\"");
}

/// @brief Parses "full"/"temperature_only"/"flow_only" into a SolveMode, or throws.
SolveMode parseSolveMode(const json& j, const std::string& fieldPath) {
    if (!j.is_string())
        throw std::runtime_error(fieldPath + " must be a string: \"full\", \"temperature_only\", or \"flow_only\"");
    std::string s = j.get<std::string>();
    if (s == "full")              return SolveMode::Full;
    if (s == "temperature_only")  return SolveMode::TemperatureOnly;
    if (s == "flow_only")         return SolveMode::FlowOnly;
    throw std::runtime_error(fieldPath + " = \"" + s + "\" is not valid -- must be \"full\", \"temperature_only\", or \"flow_only\"");
}

/// @brief Reads a required field, throwing a clear, path-qualified error if
/// it's absent -- deliberately never defaults a physical parameter silently.
template<typename T>
T requireField(const json& obj, const std::string& key, const std::string& sectionPath) {
    if (!obj.contains(key))
        throw std::runtime_error(sectionPath + "." + key + " is required but missing from the config file");
    try {
        return obj.at(key).get<T>();
    } catch (const json::exception& e) {
        throw std::runtime_error(sectionPath + "." + key + " has the wrong type: " + e.what());
    }
}

/// @brief Returns obj[key] if present, else defaultValue.
template<typename T>
T optionalField(const json& obj, const std::string& key, T defaultValue) {
    if (!obj.contains(key)) return defaultValue;
    return obj.at(key).get<T>();
}

/// @brief Loads the required grid section (Nx, Ny, Nz), validating each is at least 2.
void loadGrid(const json& root, SimulationConfig& cfg) {
    if (!root.contains("grid"))
        throw std::runtime_error("config is missing the required \"grid\" section (Nx, Ny, Nz)");
    const auto& g = root.at("grid");
    cfg.Nx = requireField<int>(g, "Nx", "grid");
    cfg.Ny = requireField<int>(g, "Ny", "grid");
    cfg.Nz = requireField<int>(g, "Nz", "grid");
    if (cfg.Nx < 2 || cfg.Ny < 2 || cfg.Nz < 2)
        throw std::runtime_error("grid.Nx/Ny/Nz must each be at least 2 (got "
            + std::to_string(cfg.Nx) + "," + std::to_string(cfg.Ny) + "," + std::to_string(cfg.Nz) + ")");
}

/// @brief Loads the required physics section: no field has a sensible default.
void loadPhysics(const json& root, SimulationConfig& cfg) {
    if (!root.contains("physics"))
        throw std::runtime_error("config is missing the required \"physics\" section");
    const auto& p = root.at("physics");
    cfg.aspectRatio = requireField<double>(p, "aspect_ratio", "physics");
    cfg.widthRatio  = requireField<double>(p, "width_ratio", "physics");
    cfg.grashof     = requireField<double>(p, "grashof", "physics");
    cfg.biot        = requireField<double>(p, "biot", "physics");
    cfg.prandtl     = requireField<double>(p, "prandtl", "physics");
    cfg.hartmann    = requireField<double>(p, "hartmann", "physics");
}

/// @brief Loads the numerics section: defaults for everything except run_duration.
void loadNumerics(const json& root, SimulationConfig& cfg) {
    const auto& n = root.contains("numerics") ? root.at("numerics") : json::object();
    cfg.eps             = optionalField(n, "eps", cfg.eps);
    cfg.epsConvergence  = optionalField(n, "eps_convergence", cfg.epsConvergence);
    cfg.maxIterations   = optionalField(n, "max_iterations", cfg.maxIterations);
    cfg.timeStep        = optionalField(n, "time_step", cfg.timeStep);
    if (!n.contains("run_duration"))
        throw std::runtime_error("numerics.run_duration is required (how much simulation time this run should advance by -- unrelated to print_every/checkpoint_every, which only control how often progress is reported)");
    cfg.runDuration     = n.at("run_duration").get<double>();
    cfg.printEvery      = optionalField(n, "print_every", cfg.printEvery);
    if (cfg.printEvery < 1)
        throw std::runtime_error("numerics.print_every must be at least 1");
}

/// @brief Loads the optional diagnostics section (conservation checks, solve mode).
void loadDiagnostics(const json& root, SimulationConfig& cfg) {
    const auto& d = root.contains("diagnostics") ? root.at("diagnostics") : json::object();
    cfg.runConservationChecks = optionalField(d, "run_conservation_checks", cfg.runConservationChecks);
    if (d.contains("mode"))
        cfg.solveMode = parseSolveMode(d.at("mode"), "diagnostics.mode");
}

/// @brief Loads the required temperature section (stationary flag, boundary_x/y/z).
void loadTemperature(const json& root, SimulationConfig& cfg) {
    if (!root.contains("temperature"))
        throw std::runtime_error("config is missing the required \"temperature\" section (boundary_x/y/z)");
    const auto& t = root.at("temperature");
    cfg.temperatureStationary = optionalField(t, "stationary", cfg.temperatureStationary);
    cfg.temperatureBoundaryX = parseBoundaryKind(requireField<json>(t, "boundary_x", "temperature"), "temperature.boundary_x");
    cfg.temperatureBoundaryY = parseBoundaryKind(requireField<json>(t, "boundary_y", "temperature"), "temperature.boundary_y");
    cfg.temperatureBoundaryZ = parseBoundaryKind(requireField<json>(t, "boundary_z", "temperature"), "temperature.boundary_z");
}

/// @brief Loads the required potential section (boundary_x/y/z).
void loadPotential(const json& root, SimulationConfig& cfg) {
    if (!root.contains("potential"))
        throw std::runtime_error("config is missing the required \"potential\" section (boundary_x/y/z)");
    const auto& p = root.at("potential");
    cfg.potentialBoundaryX = parseBoundaryKind(requireField<json>(p, "boundary_x", "potential"), "potential.boundary_x");
    cfg.potentialBoundaryY = parseBoundaryKind(requireField<json>(p, "boundary_y", "potential"), "potential.boundary_y");
    cfg.potentialBoundaryZ = parseBoundaryKind(requireField<json>(p, "boundary_z", "potential"), "potential.boundary_z");
}

/// @brief Loads the fully-optional time_averaging section (off by default).
void loadTimeAveraging(const json& root, SimulationConfig& cfg) {
    const auto& ta = root.contains("time_averaging") ? root.at("time_averaging") : json::object();
    cfg.timeAveragingEnabled = optionalField(ta, "enabled", cfg.timeAveragingEnabled);
    if (ta.contains("fourier_frequencies")) {
        if (!ta.at("fourier_frequencies").is_array())
            throw std::runtime_error("time_averaging.fourier_frequencies must be an array of numbers");
        cfg.fourierFrequencies = ta.at("fourier_frequencies").get<std::vector<double>>();
    }
}

/**
 * @brief Loads the gpu section: each field falls back to its own default
 * (see SimulationConfig.h) when the config file doesn't override it.
 */
void loadGpu(const json& root, SimulationConfig& cfg) {
    auto parseGpu = [&](const char* key, FieldGpuConfig defaults) -> FieldGpuConfig {
        if (!root.contains("gpu") || !root.at("gpu").contains(key)) return defaults;
        const auto& g = root.at("gpu").at(key);
        FieldGpuConfig out = defaults;
        out.device = optionalField(g, "device", defaults.device);
        out.useThomasAlgorithm = optionalField(g, "use_thomas_algorithm", defaults.useThomasAlgorithm);
        return out;
    };
    cfg.gpuTemperature = parseGpu("temperature", cfg.gpuTemperature);
    cfg.gpuVx          = parseGpu("vx", cfg.gpuVx);
    cfg.gpuVy          = parseGpu("vy", cfg.gpuVy);
    cfg.gpuVz          = parseGpu("vz", cfg.gpuVz);
    cfg.gpuPressure    = parseGpu("pressure", cfg.gpuPressure);
    cfg.gpuPotential   = parseGpu("potential", cfg.gpuPotential);
}

/// @brief Loads the optional mesh section (stretching_factor), validating it's positive.
void loadMesh(const json& root, SimulationConfig& cfg) {
    const auto& m = root.contains("mesh") ? root.at("mesh") : json::object();
    cfg.meshStretchingFactor = optionalField(m, "stretching_factor", cfg.meshStretchingFactor);
    if (cfg.meshStretchingFactor <= 0.0)
        throw std::runtime_error("mesh.stretching_factor must be positive");
}

/// @brief Loads the optional restart section (checkpoint read/write settings).
void loadRestart(const json& root, SimulationConfig& cfg) {
    const auto& r = root.contains("restart") ? root.at("restart") : json::object();
    if (r.contains("read_from_file") && !r.at("read_from_file").is_null())
        cfg.restartFile = r.at("read_from_file").get<std::string>();
    cfg.checkpointEvery = optionalField(r, "write_checkpoint_every", cfg.checkpointEvery);
    if (cfg.checkpointEvery < 1)
        throw std::runtime_error("restart.write_checkpoint_every must be at least 1");
}

/// @brief Loads the optional output section (directory, which files to write).
void loadOutput(const json& root, SimulationConfig& cfg) {
    const auto& o = root.contains("output") ? root.at("output") : json::object();
    cfg.outputDirectory = optionalField(o, "directory", cfg.outputDirectory);
    cfg.writeSerieAndHistory = optionalField(o, "write_serie_and_history", cfg.writeSerieAndHistory);
    cfg.writeFieldSnapshots = optionalField(o, "write_field_snapshots", cfg.writeFieldSnapshots);
}

} // namespace

SimulationConfig SimulationConfig::loadFromFile(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Could not open config file: " + path);

    json root;
    try {
        in >> root;
    } catch (const json::parse_error& e) {
        throw std::runtime_error("Malformed JSON in " + path + ": " + e.what());
    }

    SimulationConfig cfg;
    loadGrid(root, cfg);
    loadPhysics(root, cfg);
    loadNumerics(root, cfg);
    loadDiagnostics(root, cfg);
    loadTemperature(root, cfg);
    loadPotential(root, cfg);
    loadTimeAveraging(root, cfg);
    loadGpu(root, cfg);
    loadMesh(root, cfg);
    loadRestart(root, cfg);
    loadOutput(root, cfg);
    return cfg;
}
