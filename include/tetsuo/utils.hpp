// SPDX-License-Identifier: MIT
// TETSUO GPU Miner - Utility Functions
// Copyright (c) 2024-2026 TETSUO Contributors

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace tetsuo
{

/// Convert hexadecimal string to byte vector
/// @param hex Hexadecimal string (e.g., "deadbeef")
/// @return Vector of bytes
std::vector<uint8_t> hex_to_bytes (std::string_view hex);

/// Convert byte array to hexadecimal string
/// @param data Pointer to byte data
/// @param len Number of bytes
/// @return Lowercase hexadecimal string
std::string bytes_to_hex (const uint8_t *data, size_t len);

/// Reverse byte order in hexadecimal string (swap endianness)
/// @param hex Hexadecimal string
/// @return Reversed hex string (e.g., "aabbccdd" -> "ddccbbaa")
std::string reverse_hex (std::string_view hex);

/// Double SHA256 hash (SHA256d)
/// @param data Input data
/// @param len Input length in bytes
/// @param hash Output 32-byte hash
void sha256d (const uint8_t *data, size_t len, uint8_t *hash);

/// Decode Base58Check address to raw bytes
/// @param address Base58Check encoded address
/// @return 25-byte decoded address (version + 20-byte hash + 4-byte checksum)
std::vector<uint8_t> base58_decode (std::string_view address);

/// Convert P2PKH address to output script
/// @param address Base58Check P2PKH address (e.g., "T...")
/// @return P2PKH scriptPubKey (25 bytes: OP_DUP OP_HASH160 <20> OP_EQUALVERIFY
/// OP_CHECKSIG)
std::vector<uint8_t> address_to_script (const std::string &address);

} // namespace tetsuo
