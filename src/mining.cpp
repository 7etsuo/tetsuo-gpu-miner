// SPDX-License-Identifier: MIT
// TETSUO GPU Miner - Mining Functions Implementation
// Copyright (c) 2024-2026 TETSUO Contributors

#include "tetsuo/mining.hpp"
#include "tetsuo/utils.hpp"
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace tetsuo
{

void
bits_to_target (uint32_t bits, uint32_t *target)
{
  std::memset (target, 0, 32);

  uint32_t exp = (bits >> 24) & 0xFF;
  uint32_t mant = bits & 0x007FFFFF;

  // Handle sign bit in compact format
  if (bits & 0x00800000)
    {
      mant |= 0xFF800000;
    }

  // Calculate bit position: target = mantissa * 256^(exponent-3)
  int shift = 8 * (static_cast<int> (exp) - 3);
  if (shift >= 0 && shift < 256)
    {
      int word = shift / 32;       // Which 32-bit word
      int bit_offset = shift % 32; // Bit offset within word

      if (word < 8)
        {
          // Store mantissa in correct word (little-endian order)
          target[word] = mant << bit_offset;
          // Handle overflow to next word
          if (bit_offset > 0 && word < 7)
            {
              target[word + 1] |= mant >> (32 - bit_offset);
            }
        }
    }
}

std::string
build_coinbase (int height, int64_t value, const std::string &address)
{
  std::vector<uint8_t> tx;
  tx.reserve (150);

  // Version (4 bytes, little-endian)
  tx.push_back (0x01);
  tx.push_back (0x00);
  tx.push_back (0x00);
  tx.push_back (0x00);

  // Input count
  tx.push_back (0x01);

  // Previous output (null for coinbase)
  for (int i = 0; i < 32; ++i)
    tx.push_back (0x00);
  tx.push_back (0xff);
  tx.push_back (0xff);
  tx.push_back (0xff);
  tx.push_back (0xff);

  // Coinbase script with BIP34 height encoding
  std::vector<uint8_t> cb_script;
  if (height < 17)
    {
      cb_script.push_back (static_cast<uint8_t> (0x50 + height));
    }
  else if (height < 128)
    {
      cb_script.push_back (0x01);
      cb_script.push_back (static_cast<uint8_t> (height));
    }
  else if (height < 32768)
    {
      cb_script.push_back (0x02);
      cb_script.push_back (static_cast<uint8_t> (height & 0xff));
      cb_script.push_back (static_cast<uint8_t> ((height >> 8) & 0xff));
    }
  else
    {
      cb_script.push_back (0x03);
      cb_script.push_back (static_cast<uint8_t> (height & 0xff));
      cb_script.push_back (static_cast<uint8_t> ((height >> 8) & 0xff));
      cb_script.push_back (static_cast<uint8_t> ((height >> 16) & 0xff));
    }

  // Extra nonce (8 random bytes for uniqueness)
  cb_script.push_back (0x08);
  for (int i = 0; i < 8; ++i)
    {
      cb_script.push_back (static_cast<uint8_t> (std::rand () & 0xff));
    }

  tx.push_back (static_cast<uint8_t> (cb_script.size ()));
  tx.insert (tx.end (), cb_script.begin (), cb_script.end ());

  // Sequence
  tx.push_back (0xff);
  tx.push_back (0xff);
  tx.push_back (0xff);
  tx.push_back (0xff);

  // Output count
  tx.push_back (0x01);

  // Value (8 bytes, little-endian)
  for (int i = 0; i < 8; ++i)
    {
      tx.push_back (static_cast<uint8_t> ((value >> (i * 8)) & 0xff));
    }

  // Output script (P2PKH)
  auto script = address_to_script (address);
  tx.push_back (static_cast<uint8_t> (script.size ()));
  tx.insert (tx.end (), script.begin (), script.end ());

  // Locktime
  tx.push_back (0x00);
  tx.push_back (0x00);
  tx.push_back (0x00);
  tx.push_back (0x00);

  return bytes_to_hex (tx.data (), tx.size ());
}

std::string
compute_merkle_root (const std::string &coinbase_hex,
                     const std::vector<std::string> &tx_hashes)
{
  // Hash coinbase transaction
  auto coinbase = hex_to_bytes (coinbase_hex);
  uint8_t cb_hash[32];
  sha256d (coinbase.data (), coinbase.size (), cb_hash);

  // If no other transactions, coinbase hash is the merkle root
  if (tx_hashes.empty ())
    {
      return bytes_to_hex (cb_hash, 32);
    }

  // Build merkle tree
  std::vector<std::vector<uint8_t> > hashes;
  hashes.reserve (tx_hashes.size () + 1);

  // Add coinbase hash
  hashes.emplace_back (cb_hash, cb_hash + 32);

  // Add transaction hashes (reverse from display order to internal order)
  for (const auto &txid : tx_hashes)
    {
      auto h = hex_to_bytes (reverse_hex (txid));
      hashes.push_back (std::move (h));
    }

  // Build tree by pairwise hashing
  while (hashes.size () > 1)
    {
      // Duplicate last element if odd count
      if (hashes.size () % 2 != 0)
        {
          hashes.push_back (hashes.back ());
        }

      std::vector<std::vector<uint8_t> > next_level;
      next_level.reserve (hashes.size () / 2);

      for (size_t i = 0; i < hashes.size (); i += 2)
        {
          // Concatenate pair
          std::vector<uint8_t> combined;
          combined.reserve (64);
          combined.insert (combined.end (), hashes[i].begin (),
                           hashes[i].end ());
          combined.insert (combined.end (), hashes[i + 1].begin (),
                           hashes[i + 1].end ());

          // Hash pair
          uint8_t hash[32];
          sha256d (combined.data (), combined.size (), hash);
          next_level.emplace_back (hash, hash + 32);
        }

      hashes = std::move (next_level);
    }

  return bytes_to_hex (hashes[0].data (), 32);
}

} // namespace tetsuo
