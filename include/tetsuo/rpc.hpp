// SPDX-License-Identifier: MIT
// TETSUO GPU Miner - JSON-RPC Client
// Copyright (c) 2024-2026 TETSUO Contributors

#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace tetsuo
{

/// JSON-RPC client for Bitcoin-compatible nodes
class RpcClient
{
public:
  /// Construct RPC client
  /// @param url RPC endpoint URL (e.g., "http://127.0.0.1:8337")
  /// @param user Optional RPC username
  /// @param pass Optional RPC password
  RpcClient (const std::string &url, const std::string &user = "",
             const std::string &pass = "");

  ~RpcClient ();

  /// Make JSON-RPC call
  /// @param method RPC method name
  /// @param params Optional parameters (defaults to empty array)
  /// @return JSON response object
  /// @throws std::runtime_error on network or parse error
  nlohmann::json call (const std::string &method, const nlohmann::json &params
                                                  = nlohmann::json::array ());

  /// Check if connected to node
  /// @return true if last call succeeded
  bool
  is_connected () const
  {
    return connected_;
  }

private:
  std::string url_;
  std::string auth_;
  bool connected_ = false;
};

} // namespace tetsuo
