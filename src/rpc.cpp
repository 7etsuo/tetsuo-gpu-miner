// SPDX-License-Identifier: MIT
// TETSUO GPU Miner - JSON-RPC Client Implementation
// Copyright (c) 2024-2026 TETSUO Contributors

#include "tetsuo/config.hpp"
#include "tetsuo/rpc.hpp"
#include <curl/curl.h>
#include <stdexcept>

namespace tetsuo
{

namespace
{
// CURL write callback
size_t
write_callback (void *contents, size_t size, size_t nmemb, std::string *userp)
{
  userp->append (static_cast<char *> (contents), size * nmemb);
  return size * nmemb;
}
}

RpcClient::RpcClient (const std::string &url, const std::string &user,
                      const std::string &pass)
    : url_ (url)
{
  if (!user.empty ())
    {
      auth_ = user + ":" + pass;
    }
}

RpcClient::~RpcClient () = default;

nlohmann::json
RpcClient::call (const std::string &method, const nlohmann::json &params)
{
  CURL *curl = curl_easy_init ();
  if (!curl)
    {
      connected_ = false;
      throw std::runtime_error ("Failed to initialize CURL");
    }

  // Build JSON-RPC request
  nlohmann::json request = { { "jsonrpc", "1.0" },
                             { "id", "miner" },
                             { "method", method },
                             { "params", params } };
  std::string post_data = request.dump ();

  std::string response;
  struct curl_slist *headers = nullptr;
  headers = curl_slist_append (headers, "Content-Type: application/json");

  curl_easy_setopt (curl, CURLOPT_URL, url_.c_str ());
  curl_easy_setopt (curl, CURLOPT_POSTFIELDS, post_data.c_str ());
  curl_easy_setopt (curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt (curl, CURLOPT_WRITEFUNCTION, write_callback);
  curl_easy_setopt (curl, CURLOPT_WRITEDATA, &response);
  curl_easy_setopt (curl, CURLOPT_TIMEOUT,
                    static_cast<long> (constants::RPC_TIMEOUT_SECONDS));

  if (!auth_.empty ())
    {
      curl_easy_setopt (curl, CURLOPT_USERPWD, auth_.c_str ());
    }

  CURLcode res = curl_easy_perform (curl);
  curl_slist_free_all (headers);
  curl_easy_cleanup (curl);

  if (res != CURLE_OK)
    {
      connected_ = false;
      throw std::runtime_error (std::string ("CURL error: ")
                                + curl_easy_strerror (res));
    }

  connected_ = true;

  try
    {
      return nlohmann::json::parse (response);
    }
  catch (const nlohmann::json::parse_error &e)
    {
      throw std::runtime_error (std::string ("JSON parse error: ")
                                + e.what ());
    }
}

} // namespace tetsuo
