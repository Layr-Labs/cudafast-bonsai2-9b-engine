#include "engine_fake.hpp"
#include "engine_llama.hpp"
#include "session.hpp"
#include "wire.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
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

int main(int argc, char **argv) {
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
