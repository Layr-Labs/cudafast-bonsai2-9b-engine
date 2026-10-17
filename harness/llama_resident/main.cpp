#include "engine_fake.hpp"
#include "engine_llama.hpp"
#include "session.hpp"
#include "wire.hpp"

#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

static void usage(const char *prog) {
    std::cerr << "usage: " << prog
              << " [--fake] --model <gguf> --socket <path> --ctx <n> --spec serial|mtp|dflash --draft-len <d>\n"
              << "       [--draft-model <gguf>] [--n-batch <n>] [--n-ubatch <n>] [--ident <string>]\n";
}

static std::string getenv_str(const char *name) {
    const char *v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

// Keep the resident on the fastest cores of a heterogeneous CPU (GB10: ten
// Cortex-X925 next to ten Cortex-A725). The host side of every decode round
// (sampling, the draft walk, graph launches) is serial, and threads created
// later (CUDA, ggml) inherit the mask. Uniform or unreadable CPUs: no change.
static void pin_to_fastest_cores() {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return;
    }
    long best = 0;
    long slowest = 0;
    long freq[CPU_SETSIZE] = {};
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) {
            continue;
        }
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/cpuinfo_max_freq");
        if (!(f >> freq[cpu]) || freq[cpu] <= 0) {
            return;
        }
        best = std::max(best, freq[cpu]);
        slowest = slowest == 0 ? freq[cpu] : std::min(slowest, freq[cpu]);
    }
    if (best == 0 || slowest == best) {
        return;
    }
    cpu_set_t fast;
    CPU_ZERO(&fast);
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed) && freq[cpu] == best) {
            CPU_SET(cpu, &fast);
        }
    }
    if (sched_setaffinity(0, sizeof(fast), &fast) == 0) {
        std::cerr << "resident: pinned to the " << CPU_COUNT(&fast) << " fastest of " << CPU_COUNT(&allowed) << " cores\n";
    }
}

int main(int argc, char **argv) {
    pin_to_fastest_cores();
    bool fake = false;
    std::string model;
    std::string draft_model;
    std::string socket_path;
    int32_t ctx_size = 8192;
    std::string spec_mode = "serial";
    int draft_len = 0;
    int32_t n_batch = 2048;
    int32_t n_ubatch = 512;
    std::string ident = "llama-resident";

    if (getenv_str("RESIDENT_BACKEND") == "fake") {
        fake = true;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--fake") {
            fake = true;
        } else if (arg == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (arg == "--draft-model" && i + 1 < argc) {
            draft_model = argv[++i];
        } else if (arg == "--socket" && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (arg == "--ctx" && i + 1 < argc) {
            ctx_size = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--spec" && i + 1 < argc) {
            spec_mode = argv[++i];
        } else if (arg == "--draft-len" && i + 1 < argc) {
            draft_len = std::stoi(argv[++i]);
        } else if (arg == "--n-batch" && i + 1 < argc) {
            n_batch = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--n-ubatch" && i + 1 < argc) {
            n_ubatch = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--ident" && i + 1 < argc) {
            ident = argv[++i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (socket_path.empty()) {
        socket_path = getenv_str("RESIDENT_SOCKET");
    }
    if (socket_path.empty()) {
        std::cerr << "resident: --socket or RESIDENT_SOCKET is required\n";
        return 1;
    }

    const char *depth_env = std::getenv("RESIDENT_SPEC_DEPTH");
    if (depth_env && draft_len == 0 && spec_mode == "serial") {
        draft_len = std::stoi(depth_env);
        if (draft_len > 0) {
            spec_mode = "mtp";
        }
    }

    if (spec_mode != "serial" && spec_mode != "mtp" && spec_mode != "dflash") {
        std::cerr << "resident: --spec must be serial, mtp or dflash\n";
        return 1;
    }
    if (spec_mode != "serial" && draft_len <= 0) {
        std::cerr << "resident: " << spec_mode << " spec needs --draft-len > 0\n";
        return 1;
    }
    if (spec_mode == "serial") {
        draft_len = 0;
    }
    if (spec_mode == "mtp" && draft_len > kImplementedDepth) {
        std::cerr << "resident: draft-len exceeds implemented depth " << kImplementedDepth << "\n";
        return 1;
    }
    // The drafter is a separate file only for dflash; anything else would load a model nothing reads.
    if ((spec_mode == "dflash") != !draft_model.empty()) {
        std::cerr << "resident: --draft-model is required with --spec dflash and refused otherwise\n";
        return 1;
    }

    const std::string ready_file = getenv_str("RESIDENT_READY_FILE");
    const double timeout_s = [] {
        const char *v = std::getenv("RESIDENT_PHASE_TIMEOUT_S");
        return v ? std::stod(v) : 1800.0;
    }();

    const auto load_epoch = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count();

    std::unique_ptr<IEngine> engine;
    if (fake) {
        engine = std::make_unique<FakeEngine>(draft_len, 131072, 8192, spec_mode, draft_model);
    } else {
        if (model.empty()) {
            std::cerr << "resident: --model is required\n";
            return 1;
        }
        std::cerr << "resident: loading the model\n";
        engine = std::make_unique<LlamaEngine>(model, spec_mode, draft_len, draft_model, ctx_size, n_batch, n_ubatch,
                                               ident);
        std::cerr << "resident: model loaded once for this window (" << ident << ", " << spec_mode
                  << ", depth=" << draft_len << ")\n";
    }

    Session session(std::move(engine), true, load_epoch);
    return run_resident_server(session, socket_path, ready_file, timeout_s);
}
