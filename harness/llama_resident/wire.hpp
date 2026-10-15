#pragma once

#include "session.hpp"

#include <memory>
#include <string>

void serve_connection(Session &session, int client_fd, double timeout_s);
int run_resident_server(Session &session, const std::string &socket_path, const std::string &ready_file,
                        double timeout_s);
