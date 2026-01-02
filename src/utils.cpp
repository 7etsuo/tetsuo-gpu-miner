// SPDX-License-Identifier: MIT
// TETSUO GPU Miner - Utility Functions Implementation
// Copyright (c) 2024-2026 TETSUO Contributors

#include "tetsuo/utils.hpp"
#include <cstring>
#include <iomanip>
#include <openssl/sha.h>
#include <sstream>

namespace tetsuo
{

std::vector<uint8_t>
hex_to_bytes (std::string_view hex)
{
  std::vector<uint8_t> bytes;
  bytes.reserve (hex.length () / 2);

  for (size_t i = 0; i + 1 < hex.length (); i += 2)
    {
      unsigned int byte;
      auto result = std::sscanf (hex.data () + i, "%02x", &byte);
      if (result != 1)
        break;
      bytes.push_back (static_cast<uint8_t> (byte));
    }
  return bytes;
}

std::string
bytes_to_hex (const uint8_t *data, size_t len)
{
  std::ostringstream ss;
  ss << std::hex << std::setfill ('0');
  for (size_t i = 0; i < len; ++i)
    {
      ss << std::setw (2) << static_cast<int> (data[i]);
    }
  return ss.str ();
}

std::string
reverse_hex (std::string_view hex)
{
  std::string result;
  result.reserve (hex.length ());
  for (size_t i = hex.length (); i >= 2; i -= 2)
    {
      result.append (hex.substr (i - 2, 2));
    }
  return result;
}

void
sha256d (const uint8_t *data, size_t len, uint8_t *hash)
{
  uint8_t tmp[32];
  SHA256 (data, len, tmp);
  SHA256 (tmp, 32, hash);
}

std::vector<uint8_t>
base58_decode (std::string_view address)
{
  static const char *alphabet
      = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

  std::vector<uint8_t> decoded (25, 0);

  for (char c : address)
    {
      const char *p = std::strchr (alphabet, c);
      if (!p)
        continue;

      int carry = static_cast<int> (p - alphabet);
      for (int i = 24; i >= 0; --i)
        {
          carry += 58 * decoded[i];
          decoded[i] = static_cast<uint8_t> (carry % 256);
          carry /= 256;
        }
    }

  return decoded;
}

std::vector<uint8_t>
address_to_script (const std::string &address)
{
  auto decoded = base58_decode (address);

  // Build P2PKH script: OP_DUP OP_HASH160 <20 bytes> OP_EQUALVERIFY
  // OP_CHECKSIG
  std::vector<uint8_t> script;
  script.reserve (25);
  script.push_back (0x76); // OP_DUP
  script.push_back (0xa9); // OP_HASH160
  script.push_back (0x14); // Push 20 bytes

  // Copy pubkey hash (bytes 1-20 of decoded address)
  for (int i = 1; i <= 20; ++i)
    {
      script.push_back (decoded[i]);
    }

  script.push_back (0x88); // OP_EQUALVERIFY
  script.push_back (0xac); // OP_CHECKSIG

  return script;
}

} // namespace tetsuo
