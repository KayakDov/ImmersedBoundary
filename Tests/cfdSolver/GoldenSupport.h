#pragma once

/**
 * @file GoldenSupport.h
 * @brief Host-only helpers for the golden-value test: renders Conv.dat
 * input, and reads/compares every Fortran-vs-C++ output file format
 * (plain 3-column tables, 3D Tecplot blocks, and the Conv.out/History.dat
 * iteration report) within numeric tolerance.
 */
#include "third_party/json.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cfd_golden {
namespace fs = std::filesystem;
using Json = nlohmann::json;

/// @brief Strips leading/trailing whitespace.
inline std::string trim(const std::string& s) {
    const auto a = s.find_first_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}

/// @brief Trims and collapses internal whitespace runs to a single space.
inline std::string compact(const std::string& s) {
    return std::regex_replace(trim(s), std::regex("\\s+"), " ");
}

/// @brief Reads and parses a JSON file, throwing if it cannot be opened.
inline Json readJson(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open " + path.string());
    Json j;
    in >> j;
    return j;
}

/// @brief Writes text to a file, throwing if the write fails.
inline void writeText(const fs::path& path, const std::string& text) {
    std::ofstream out(path);
    out << text;
    out.close();
    if (!out) throw std::runtime_error("Cannot write " + path.string());
}

/**
 * @brief Renders a double into Conv.dat's fixed-format REAL input field.
 *
 * ConvMain reads REALs with G15.8, NOT list-directed input. An integer
 * spelling such as "1000" is therefore read as 0.00001000. Always emit a
 * decimal point, fit the entire field into 15 characters, and require an
 * exact double round trip. Reject an unrepresentable input instead of
 * silently changing the physics on one side of the comparison.
 */
inline std::string fortranReal(double x) {
    if (!std::isfinite(x)) throw std::runtime_error("Non-finite Conv.dat input");
    std::ostringstream fixed;
    fixed.imbue(std::locale::classic());
    fixed << std::fixed << std::setprecision(17) << x;
    std::string s = fixed.str();
    while (s.back() == '0') s.pop_back(); /* Retains the decimal point. */
    if (s.size() <= 15 && std::stod(s) == x) return s;
    for (int p = 16; p >= 1; --p) {
        std::ostringstream sci;
        sci.imbue(std::locale::classic());
        sci << std::scientific << std::setprecision(p) << x;
        s = sci.str();
        if (s.size() <= 15 && std::stod(s) == x) return s;
    }
    throw std::runtime_error("Value cannot round-trip in Fortran's G15.8 input field: " + fixed.str());
}

/// @brief Renders an int into Conv.dat's I5 input field, throwing if it overflows.
inline std::string fortranInt(int x) {
    const std::string s = std::to_string(x);
    if (s.size() > 5) throw std::runtime_error("Integer exceeds Fortran's I5 input field: " + s);
    return s;
}

/// @brief Maps a JSON boundary-kind string to Conv.dat's 0/1 (dirichlet/neumann) code.
inline int boundary(const Json& j, const char* key) {
    const auto s = j.at(key).get<std::string>();
    if (s == "dirichlet") return 0;
    if (s == "neumann") return 1;
    throw std::runtime_error(std::string("Invalid boundary: ") + key + "=" + s);
}

/**
 * @brief Renders a scenario's JSON config as a full Conv.dat input file.
 *
 * Covers every Conv.dat physics/numerics/BC/averaging input. Only Ncpus
 * comes from the baseline; the supplied Fortran overrides it anyway.
 */
inline std::string renderConvDat(const Json& j, const fs::path& baseline) {
    std::ifstream in(baseline);
    if (!in) throw std::runtime_error("Cannot open baseline " + baseline.string());
    std::vector<std::string> base;
    for (std::string s; std::getline(in, s);) base.push_back(s);
    if (base.size() < 27) throw std::runtime_error("Baseline Conv.dat needs at least 27 lines");
    const auto n = j.value("numerics", Json::object());
    const auto d = j.value("diagnostics", Json::object());
    const auto ta = j.value("time_averaging", Json::object());
    const auto restart = j.value("restart", Json::object());
    const auto mesh = j.value("mesh", Json::object());
    const auto output = j.value("output", Json::object());
    if (restart.contains("read_from_file") && !restart.at("read_from_file").is_null())
        throw std::runtime_error("Golden comparison requires fresh starts on both engines");
    if (mesh.value("stretching_factor", 10.0) != 10.0)
        throw std::runtime_error("Reference Fortran hardcodes mesh stretching_factor=10");
    if (!output.value("write_serie_and_history", true))
        throw std::runtime_error("Golden comparison requires Serie.dat and History.dat");
    if (n.value("print_every", 1000) < 1)
        throw std::runtime_error("print_every must be positive (negative means restart in Fortran)");

    std::vector<std::string> lines;
    const auto real = [&](double x) { lines.push_back(fortranReal(x)); };
    const auto integer = [&](int x) { lines.push_back(fortranInt(x)); };
    for (const char* k : {"aspect_ratio", "width_ratio", "grashof", "biot", "prandtl", "hartmann"})
        real(j.at("physics").at(k).get<double>());
    for (const char* k : {"Nx", "Ny", "Nz"}) integer(j.at("grid").at(k).get<int>());
    real(n.value("eps", 1e-7));
    real(n.value("eps_convergence", 1e-6));
    integer(n.value("max_iterations", 9000));
    real(n.value("time_step", 1e-4));
    real(n.at("run_duration").get<double>());
    integer(n.value("print_every", 1000));
    integer(d.value("run_conservation_checks", false) ? 0 : 1);
    const auto mode = d.value("mode", std::string("full"));
    if (mode != "full" && mode != "temperature_only" && mode != "flow_only")
        throw std::runtime_error("Unsupported diagnostics.mode: " + mode);
    integer(mode == "full" ? 0 : mode == "temperature_only" ? -1 : 1);
    integer(j.at("temperature").value("stationary", false) ? 0 : 1);
    for (const char* section : {"temperature", "potential"})
        for (const char* k : {"boundary_x", "boundary_y", "boundary_z"})
            integer(boundary(j.at(section), k));
    lines.push_back(base.at(24)); /* Ncpus, not a physical parameter. */
    integer(ta.value("enabled", false) ? 0 : 1);
    const auto frequencies = ta.value("fourier_frequencies", std::vector<double>{});
    integer(static_cast<int>(frequencies.size()));
    for (double f : frequencies) real(f);
    std::ostringstream out;
    for (const auto& line : lines) out << line << '\n';
    return out.str();
}

/// @brief Creates and returns a fresh, uniquely-named scratch directory for one run.
inline fs::path uniqueRunDirectory(const std::string& scenario) {
    static std::atomic<unsigned long long> serial{0};
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    for (unsigned i = 0; i < 1000; ++i) {
        auto p = fs::temp_directory_path() /
            ("cfd_golden_" + scenario + "_" + std::to_string(stamp) + "_" + std::to_string(serial++));
        /* Atomic directory creation also protects against separate processes
           choosing the same timestamp. Never delete another run's directory. */
        if (fs::create_directory(p)) return p;
    }
    throw std::runtime_error("Cannot allocate a unique golden-test directory");
}

/// @brief Single-quotes a path for safe use in a shell command line.
inline std::string shellQuote(const fs::path& path) {
    std::string out = "'";
    for (char c : path.string()) out += c == '\'' ? "'\\''" : std::string(1, c);
    return out + "'";
}

/// @brief Accumulates mismatch counts and up to 20 example messages for one comparison.
struct Comparison {
    std::size_t mismatches = 0;
    std::vector<std::string> examples;
    void fail(const std::string& message) {
        ++mismatches;
        if (examples.size() < 20) examples.push_back(message);
    }
    std::string summary() const {
        std::ostringstream s;
        s << mismatches << " mismatch(es)";
        for (const auto& e : examples) s << '\n' << e;
        if (mismatches > examples.size()) s << "\nFurther differences omitted; inspect retained output files.";
        return s.str();
    }
};

/// @brief Parses a Fortran-style real (D/d exponent marker accepted) into a finite double.
inline bool number(std::string s, double& x) {
    std::replace(s.begin(), s.end(), 'D', 'E');
    std::replace(s.begin(), s.end(), 'd', 'e');
    char* end = nullptr;
    x = std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0' && std::isfinite(x);
}

/**
 * @brief Half a unit in the last PRINTED decimal position of token @p s.
 *
 * Used only for the low-precision Tecplot and Conv.out reports, never for
 * History/Serie.
 */
inline double roundingRadius(std::string s, double x) {
    if (x == 0) return 0; /* Scientific-format zero does not imply +/- 5e-6. */
    const auto dot = s.find('.');
    if (dot == std::string::npos) return 0;
    const auto e = s.find_first_of("eEdD");
    const auto end = e == std::string::npos ? s.size() : e;
    const int exponent = e == std::string::npos ? 0 : std::stoi(s.substr(e + 1));
    return 0.5 * std::pow(10.0, exponent - static_cast<int>(end - dot - 1));
}

/// @brief Compares two numeric tokens within a relative/absolute/rounding tolerance, recording any mismatch.
inline void compareNumber(Comparison& result, const std::string& where,
                          const std::string& a, const std::string& b, bool printed,
                          double rtol = 1e-6, double atol = 1e-12, int significantDigits = 0) {
    double x = 0, y = 0;
    if (!number(a, x) || !number(b, y)) {
        result.fail(where + ": invalid/non-finite numeric token; Fortran=" + a + " C++=" + b);
        return;
    }
    double tolerance = atol + rtol * std::max(std::abs(x), std::abs(y));
    if (printed) {
        /* printf %g strips trailing zeros: "0.1" in a %.8g report still
           has eight significant digits, not a +/- 0.05 uncertainty. */
        const auto radius = [&](const std::string& token, double value) {
            if (significantDigits == 0) return roundingRadius(token, value);
            return value == 0 ? 0.0 : 0.5 * std::pow(10.0,
                std::floor(std::log10(std::abs(value))) - significantDigits + 1);
        };
        tolerance += radius(a, x) + radius(b, y);
    }
    if (std::abs(x - y) > tolerance) {
        std::ostringstream s;
        s << where << ": Fortran=" << a << " C++=" << b
          << " absDiff=" << std::abs(x - y) << " allowed=" << tolerance;
        result.fail(s.str());
    }
}

/// @brief Splits a line on whitespace.
inline std::vector<std::string> tokens(const std::string& line) {
    std::istringstream in(line);
    std::vector<std::string> v;
    for (std::string t; in >> t;) v.push_back(t);
    return v;
}

/// @brief A parsed 2D Tecplot table (or plain 3-column data): variable labels, I/J extents, and rows.
struct Table {
    std::vector<std::string> labels;
    std::size_t ni = 0, nj = 0;
    std::vector<std::vector<std::string>> rows;
};

/// @brief Reads a 3-column data file, optionally parsing a 2D Tecplot VARIABLES/ZONE header first.
inline Table readTable(const fs::path& path, bool tecplot) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open " + path.string());
    Table t;
    std::string line;
    if (tecplot) {
        const std::regex variables(R"re(^\s*VARIABLES\s*=\s*"([^"]*)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*$)re");
        const std::regex zone(R"(^\s*ZONE\s+F\s*=\s*POINT\s*,\s*I\s*=\s*(\d+)\s*,\s*J\s*=\s*(\d+)\s*$)");
        std::smatch m;
        if (!std::getline(in, line) || !std::regex_match(line, m, variables))
            throw std::runtime_error("Invalid Tecplot VARIABLES header: " + path.string());
        for (int i = 1; i <= 3; ++i) t.labels.push_back(trim(m[i].str()));
        if (!std::getline(in, line) || !std::regex_match(line, m, zone))
            throw std::runtime_error("Invalid Tecplot ZONE header: " + path.string());
        t.ni = std::stoull(m[1].str());
        t.nj = std::stoull(m[2].str());
    }
    while (std::getline(in, line)) {
        auto row = tokens(line);
        if (row.empty()) continue;
        if (row.size() != 3) throw std::runtime_error("Expected 3 columns in " + path.string() + " at data row " + std::to_string(t.rows.size() + 1));
        for (const auto& s : row) {
            double x;
            if (!number(s, x)) throw std::runtime_error("Invalid/non-finite number in " + path.string() + ": " + s);
        }
        t.rows.push_back(std::move(row));
    }
    if (in.bad()) throw std::runtime_error("Read failure: " + path.string());
    if (t.rows.empty()) throw std::runtime_error("No data in " + path.string());
    if (tecplot && (t.ni == 0 || t.nj == 0 || t.rows.size() / t.ni != t.nj || t.rows.size() % t.ni != 0))
        throw std::runtime_error("Tecplot row count disagrees with I*J: " + path.string());
    return t;
}

/// @brief Compares two 3-column (optionally 2D Tecplot) data files cell-by-cell.
inline Comparison compareTables(const fs::path& a, const fs::path& b, bool tecplot) {
    Comparison result;
    const auto x = readTable(a, tecplot), y = readTable(b, tecplot);
    if (x.labels != y.labels) result.fail("Tecplot variable names differ");
    if (x.ni != y.ni || x.nj != y.nj) result.fail("Tecplot grid dimensions differ");
    if (x.rows.size() != y.rows.size()) result.fail("Data row counts differ: Fortran=" + std::to_string(x.rows.size()) + " C++=" + std::to_string(y.rows.size()));
    for (std::size_t i = 0; i < std::min(x.rows.size(), y.rows.size()); ++i)
        for (std::size_t c = 0; c < 3; ++c)
            compareNumber(result, "data row " + std::to_string(i + 1) + " column " + std::to_string(c + 1),
                          x.rows[i][c], y.rows[i][c], tecplot);
    return result;
}

/**
 * @brief A parsed 3D Tecplot table: variable labels, I/J/K extents, and rows.
 *
 * Matches writeTecplot3D()'s own format (Tmpr_av.dat/Vx_av.dat/Vy_av.dat/
 * Vz_av.dat): VARIABLES="X","Y","Z","<name>" over ZONE F=POINT,I=..,J=..,
 * K=.. -- four columns and a K dimension, unlike the 2D Y-averaged
 * diagnostics' three columns and I/J above. Kept as a separate
 * Table3D/readTable3D/compareTables3D trio rather than overloading
 * Table/readTable/compareTables with a variable column count, so the
 * existing, already-verified 2D path is untouched.
 */
struct Table3D {
    std::vector<std::string> labels;
    std::size_t ni = 0, nj = 0, nk = 0;
    std::vector<std::vector<std::string>> rows;
};

/**
 * @brief Reads a 3D Tecplot VARIABLES/ZONE block as a flat token stream.
 *
 * pointw.f90's own data format is `3(E12.5,1x)` -- three slots -- but
 * Point_Write writes four values per point (X,Y,Z,FFF). Fortran's
 * format-reversion rule means it wraps: X,Y,Z land on one physical line,
 * FFF alone starts the next. (Point_Write_2D doesn't have this problem --
 * it writes exactly 3 values into the same 3-slot format, an exact fit,
 * which is why readTable above can assume one line is one row.) Reading
 * the data section as one flat token stream, not line-by-line, makes that
 * wrap invisible here: four consecutive tokens is one point regardless of
 * which physical lines they fell across, on either side of the
 * comparison.
 */
inline Table3D readTable3D(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open " + path.string());
    Table3D t;
    std::string line;
    const std::regex variables(R"re(^\s*VARIABLES\s*=\s*"([^"]*)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*$)re");
    const std::regex zone(R"(^\s*ZONE\s+F\s*=\s*POINT\s*,\s*I\s*=\s*(\d+)\s*,\s*J\s*=\s*(\d+)\s*,\s*K\s*=\s*(\d+)\s*$)");
    std::smatch m;
    if (!std::getline(in, line) || !std::regex_match(line, m, variables))
        throw std::runtime_error("Invalid Tecplot VARIABLES header: " + path.string());
    for (int i = 1; i <= 4; ++i) t.labels.push_back(trim(m[i].str()));
    if (!std::getline(in, line) || !std::regex_match(line, m, zone))
        throw std::runtime_error("Invalid Tecplot ZONE header: " + path.string());
    t.ni = std::stoull(m[1].str());
    t.nj = std::stoull(m[2].str());
    t.nk = std::stoull(m[3].str());
    std::vector<std::string> flat;
    for (std::string tok; in >> tok;) flat.push_back(tok);
    if (in.bad()) throw std::runtime_error("Read failure: " + path.string());
    if (flat.size() % 4 != 0)
        throw std::runtime_error("Token count " + std::to_string(flat.size()) + " not a multiple of 4 in " + path.string());
    for (std::size_t i = 0; i < flat.size(); i += 4) {
        std::vector<std::string> row(flat.begin() + (long)i, flat.begin() + (long)i + 4);
        for (const auto& s : row) {
            double x;
            if (!number(s, x)) throw std::runtime_error("Invalid/non-finite number in " + path.string() + ": " + s);
        }
        t.rows.push_back(std::move(row));
    }
    if (t.rows.empty()) throw std::runtime_error("No data in " + path.string());
    if (t.ni == 0 || t.nj == 0 || t.nk == 0 || t.rows.size() != t.ni * t.nj * t.nk)
        throw std::runtime_error("Tecplot row count disagrees with I*J*K: " + path.string());
    return t;
}

/// @brief Compares two 3D Tecplot data files cell-by-cell.
inline Comparison compareTables3D(const fs::path& a, const fs::path& b) {
    Comparison result;
    const auto x = readTable3D(a), y = readTable3D(b);
    if (x.labels != y.labels) result.fail("Tecplot variable names differ");
    if (x.ni != y.ni || x.nj != y.nj || x.nk != y.nk) result.fail("Tecplot grid dimensions differ");
    if (x.rows.size() != y.rows.size()) result.fail("Data row counts differ: Fortran=" + std::to_string(x.rows.size()) + " C++=" + std::to_string(y.rows.size()));
    for (std::size_t i = 0; i < std::min(x.rows.size(), y.rows.size()); ++i)
        for (std::size_t c = 0; c < 4; ++c)
            compareNumber(result, "data row " + std::to_string(i + 1) + " column " + std::to_string(c + 1),
                          x.rows[i][c], y.rows[i][c], /*printed=*/true);
    return result;
}

/// @brief The final-report scalar labels every golden scenario must produce.
inline const std::vector<std::string>& finalLabels() {
    static const std::vector<std::string> labels = {"Nusselt number", "Midplane Nusselt number",
        "kinetic energy", "2D Nusselt number", "2D kinetic energy"};
    return labels;
}

/// @brief A parsed Conv.out/History.dat-style report: scalar fields plus one row per iteration.
struct Report {
    std::map<std::string, std::string> scalars;
    std::vector<std::vector<std::string>> iterations;
};

/// @brief Parses a Conv.out/History.dat-style report into its scalar fields and per-iteration rows.
inline Report readReport(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open " + path.string());
    Report r;
    const std::regex step(R"(^Istp=\s*(\S+)\s+Time Step=\s*(\S+)\s+Time=\s*(\S+)\s+End Time=\s*(\S+)\s*$)");
    const std::regex residual(R"(^RNSx=\s*(\S+)\s+RNSy=\s*(\S+)\s+RNSz=\s*(\S+)\s+RTmpr=\s*(\S+)\s+RDP=\s*(\S+)\s*$)");
    bool pending = false;
    for (std::string line; std::getline(in, line);) {
        line = compact(line);
        if (line.empty()) continue;
        std::smatch m;
        if (line.rfind("Istp=", 0) == 0) {
            if (pending || !std::regex_match(line, m, step)) throw std::runtime_error("Malformed iteration report in " + path.string());
            std::vector<std::string> row;
            for (int i = 1; i <= 4; ++i) row.push_back(m[i].str());
            r.iterations.push_back(row);
            pending = true;
        } else if (line.rfind("RNSx=", 0) == 0) {
            if (!pending || !std::regex_match(line, m, residual)) throw std::runtime_error("Malformed residual report in " + path.string());
            for (int i = 1; i <= 5; ++i) r.iterations.back().push_back(m[i].str());
            pending = false;
        } else {
            const auto eq = line.find('=');
            if (eq != std::string::npos && line.find('=', eq + 1) == std::string::npos) {
                const auto key = trim(line.substr(0, eq));
                if (!r.scalars.emplace(key, trim(line.substr(eq + 1))).second)
                    throw std::runtime_error("Duplicate report field: " + key + " in " + path.string());
            }
        }
    }
    if (in.bad() || pending) throw std::runtime_error("Incomplete report in " + path.string());
    for (const auto& key : finalLabels())
        if (!r.scalars.count(key)) throw std::runtime_error("Missing report field '" + key + "' in " + path.string());
    return r;
}

/**
 * @brief Compares two reports' scalar fields and per-iteration rows within tolerance.
 * @param skipLabels finalLabels() entries to leave uncompared, for a scenario
 * where Fortran's own number is known not to be the one worth matching --
 * currently just "Nusselt number"/"Midplane Nusselt number" when
 * time-averaging is on: Solution_time_Q2D.f90's substitution
 * (Tmpr=Tmpr+Teta) runs before ekinem_3D.f90's Nusselt()/Nusselt_middle()
 * read Tmpr, and those already have their own explicit `1/AspRa + Snu`
 * term, so Fortran ends up counting Teta's contribution twice for exactly
 * these two numbers. cfdSolver's printFinalReport() deliberately keeps
 * temperature_ Teta-free to avoid that same double-count (see
 * CfdSimulation.cu's AverageFlow::run() call site), so its number is the
 * internally-consistent one here, not Fortran's. Empty means nothing is
 * skipped, the behavior every scenario before this one still gets.
 */
inline Comparison compareReports(const fs::path& a, const fs::path& b, const std::vector<std::string>& skipLabels) {
    Comparison result;
    const auto x = readReport(a), y = readReport(b);
    for (const auto& key : finalLabels()) {
        if (std::find(skipLabels.begin(), skipLabels.end(), key) != skipLabels.end()) continue;
        compareNumber(result, key, x.scalars.at(key), y.scalars.at(key), false);
    }
    if (x.iterations.size() != y.iterations.size()) result.fail("Iteration report counts differ");
    for (std::size_t i = 0; i < std::min(x.iterations.size(), y.iterations.size()); ++i)
        for (std::size_t c = 0; c < 9; ++c)
            compareNumber(result, "iteration report " + std::to_string(i + 1) + " field " + std::to_string(c + 1),
                          x.iterations[i][c], y.iterations[i][c], c != 0, c == 0 ? 0 : 1e-6, c == 0 ? 0 : 1e-12,
                          c == 2 || c == 3 ? 8 : 4);
    return result;
}

/**
 * @brief Checks what Fortran actually READ back from Conv.dat, not just what was written.
 *
 * Catches e.g. G15.8/I5 truncation or using a different executable.
 */
inline Comparison verifyFortranEcho(const fs::path& report, const Json& j) {
    Comparison result;
    const auto r = readReport(report);
    const auto check = [&](const char* label, double expected) {
        const auto it = r.scalars.find(label);
        if (it == r.scalars.end()) result.fail(std::string("Missing Fortran input echo: ") + label);
        else {
            std::ostringstream exact;
            exact.imbue(std::locale::classic());
            exact << std::scientific << std::setprecision(17) << expected;
            /* Fortran echoes these with G15.8 (eight significant digits). */
            compareNumber(result, std::string("Fortran input echo: ") + label, it->second, exact.str(), false, 1e-7);
        }
    };
    const std::vector<std::pair<const char*, const char*>> physics = {
        {"Aspect Ratio", "aspect_ratio"}, {"Width Ratio", "width_ratio"}, {"Grashof number", "grashof"},
        {"Biot number", "biot"}, {"Prandtl number", "prandtl"}, {"Hartmann number", "hartmann"}};
    for (const auto& p : physics) check(p.first, j.at("physics").at(p.second).get<double>());
    const auto n = j.at("numerics");
    check("Initial time step", n.value("time_step", 1e-4));
    check("Time interval", n.at("run_duration").get<double>());
    check("Print step", n.value("print_every", 1000));
    return result;
}
} // namespace cfd_golden
