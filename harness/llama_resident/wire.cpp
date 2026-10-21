#include "wire.hpp"

#include <nlohmann/json.hpp>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

using json = nlohmann::json;

// Enough for an 8192-token sync line plus JSON overhead.
constexpr size_t kMaxRequestLine = 4 * 1024 * 1024;

static void log_msg(const std::string &msg) {
    std::cerr << "resident: " << msg << std::endl;
}

static bool set_timeout(int fd, double timeout_s) {
    struct timeval tv {};
    tv.tv_sec = static_cast<time_t>(timeout_s);
    tv.tv_usec = static_cast<suseconds_t>((timeout_s - tv.tv_sec) * 1e6);
    return setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == 0 &&
           setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) == 0;
}

static void reject_oversized_line(int fd, std::string &unread) {
    unread.clear();
    shutdown(fd, SHUT_RDWR);
}

static std::string read_line(int fd, std::string &unread, bool &oversized) {
    std::string line;
    char chunk[8192];
    while (true) {
        if (!unread.empty()) {
            const size_t nl = unread.find('\n');
            if (nl != std::string::npos) {
                if (line.size() + nl > kMaxRequestLine) {
                    oversized = true;
                    reject_oversized_line(fd, unread);
                    return {};
                }
                line.append(unread.data(), nl);
                unread.erase(0, nl + 1);
                return line;
            }
            if (line.size() + unread.size() > kMaxRequestLine) {
                oversized = true;
                reject_oversized_line(fd, unread);
                return {};
            }
            line.append(unread);
            unread.clear();
            continue;
        }
        const ssize_t n = read(fd, chunk, sizeof(chunk));
        if (n == 0) {
            return {};
        }
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return {};
        }
        for (ssize_t i = 0; i < n; ++i) {
            const char ch = chunk[i];
            if (ch == '\n') {
                if (line.size() + static_cast<size_t>(i) > kMaxRequestLine) {
                    oversized = true;
                    reject_oversized_line(fd, unread);
                    return {};
                }
                line.append(chunk, static_cast<size_t>(i));
                unread.assign(chunk + i + 1, static_cast<size_t>(n - i - 1));
                return line;
            }
        }
        if (line.size() + static_cast<size_t>(n) > kMaxRequestLine) {
            oversized = true;
            reject_oversized_line(fd, unread);
            return {};
        }
        line.append(chunk, static_cast<size_t>(n));
    }
}

static bool write_all(int fd, const void *data, size_t len) {
    const char *buf = static_cast<const char *>(data);
    size_t off = 0;
    while (off < len) {
        const ssize_t n = send(fd, buf + off, len - off, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        off += static_cast<size_t>(n);
    }
    return true;
}

void serve_connection(Session &session, int client_fd, double timeout_s) {
    set_timeout(client_fd, timeout_s);
    std::string unread;
    while (true) {
        bool oversized = false;
        const std::string line = read_line(client_fd, unread, oversized);
        if (oversized || line.empty()) {
            break;
        }
        json reply;
        bool keep = true;
        try {
            json req = json::parse(line);
            if (!req.is_object()) {
                throw RefusedError("a request must be a JSON object");
            }
            if (!req.contains("op") || !req["op"].is_string()) {
                throw RefusedError("request carries no \"op\" string");
            }
            const std::string op = req["op"].get<std::string>();
            std::string body;
            keep = session.handle_op(op, &req, body);
            reply = json::parse(body);
        } catch (const RefusedError &e) {
            reply = {{"ok", false}, {"error", e.what()}};
        } catch (const std::exception &e) {
            reply = {{"ok", false}, {"error", e.what()}};
        }
        const std::string out = reply.dump() + "\n";
        if (!write_all(client_fd, out.data(), out.size())) {
            break;
        }
        if (!keep) {
            break;
        }
    }
}

int run_resident_server(Session &session, const std::string &socket_path, const std::string &ready_file,
                        double timeout_s) {
    if (!socket_path.empty() && ::unlink(socket_path.c_str()) != 0 && errno != ENOENT) {
        log_msg(std::string("cannot unlink socket: ") + std::strerror(errno));
        return 1;
    }

    const int listener = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listener < 0) {
        log_msg("socket() failed");
        return 1;
    }

    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    if (socket_path.size() >= sizeof(addr.sun_path)) {
        log_msg("socket path too long");
        close(listener);
        return 1;
    }
    std::memcpy(addr.sun_path, socket_path.c_str(), socket_path.size() + 1);
    if (bind(listener, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        log_msg(std::string("bind failed: ") + std::strerror(errno));
        close(listener);
        return 1;
    }
    chmod(socket_path.c_str(), 0600);
    if (listen(listener, 8) != 0) {
        log_msg("listen failed");
        close(listener);
        return 1;
    }

    if (!ready_file.empty()) {
        FILE *fp = fopen(ready_file.c_str(), "w");
        if (fp) {
            fprintf(fp, "%s\n", socket_path.c_str());
            fclose(fp);
        }
    }

    log_msg("listening on " + socket_path);
    int phase = 0;
    while (true) {
        const int client = accept(listener, nullptr, nullptr);
        if (client < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        phase++;
        session.invalidate();
        log_msg("phase " + std::to_string(phase) + " connected");
        serve_connection(session, client, timeout_s);
        close(client);
        log_msg("phase " + std::to_string(phase) + " closed");
    }

    close(listener);
    if (!socket_path.empty()) {
        ::unlink(socket_path.c_str());
    }
    if (!ready_file.empty()) {
        ::unlink(ready_file.c_str());
    }
    return 0;
}
