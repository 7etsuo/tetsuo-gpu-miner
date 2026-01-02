// SPDX-License-Identifier: MIT
// TETSUO GPU Miner - Multi-GPU SHA256d Mining
// Copyright (c) 2024-2026 TETSUO Contributors
//
// High-performance CUDA miner for TETSUO blockchain
// Supports multiple GPUs with automatic work distribution

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <iomanip>

#include <curl/curl.h>

#include "tetsuo/config.hpp"
#include "tetsuo/utils.hpp"
#include "tetsuo/rpc.hpp"
#include "tetsuo/mining.hpp"
#include "tetsuo/cuda_utils.cuh"
#include "sha256.cuh"

using namespace tetsuo;

//==============================================================================
// Global State
//==============================================================================

static Config g_config;
static std::atomic<bool> g_running(true);
static std::atomic<uint64_t> g_total_hashes(0);
static std::atomic<uint64_t> g_blocks_found(0);
static std::atomic<uint64_t> g_gpu_hashes[constants::MAX_GPUS];
static int g_num_gpus = 0;
static std::mutex g_work_mutex;
static std::mutex g_submit_mutex;

//==============================================================================
// Signal Handler
//==============================================================================

void signal_handler(int) {
    std::cout << "\nShutting down..." << std::endl;
    g_running = false;
}

//==============================================================================
// RPC Interface
//==============================================================================

static std::unique_ptr<RpcClient> g_rpc;

MiningWork get_work() {
    MiningWork work;

    std::lock_guard<std::mutex> lock(g_work_mutex);

    try {
        nlohmann::json params = nlohmann::json::array({
            {{"rules", {"segwit"}}}
        });
        auto response = g_rpc->call("getblocktemplate", params);

        if (response.contains("error") && !response["error"].is_null()) {
            return work;
        }

        auto result = response["result"];

        work.height = result["height"].get<int>();
        std::string prev_hash = result["previousblockhash"].get<std::string>();
        std::string bits_hex = result["bits"].get<std::string>();
        int64_t curtime = result["curtime"].get<int64_t>();
        int version = result["version"].get<int>();
        int64_t coinbase_value = result["coinbasevalue"].get<int64_t>();

        // Get coinbase transaction from node
        std::string coinbasetxn;
        if (result.contains("coinbasetxn") && result["coinbasetxn"].contains("data")) {
            coinbasetxn = result["coinbasetxn"]["data"].get<std::string>();
        }

        // Get transactions
        std::vector<std::string> tx_data;
        std::vector<std::string> tx_hashes;
        if (result.contains("transactions")) {
            for (const auto& tx : result["transactions"]) {
                if (tx.contains("data")) {
                    tx_data.push_back(tx["data"].get<std::string>());
                }
                if (tx.contains("txid")) {
                    tx_hashes.push_back(tx["txid"].get<std::string>());
                }
            }
        }

        if (prev_hash.empty() || bits_hex.empty()) {
            return work;
        }

        uint32_t bits = std::stoul(bits_hex, nullptr, 16);
        bits_to_target(bits, work.target);

        // Build coinbase and merkle root
        std::string merkle_root;
        bool custom_coinbase = false;

        if (!g_config.mining_address.empty()) {
            // Custom coinbase with our address
            coinbasetxn = build_coinbase(work.height, coinbase_value, g_config.mining_address);
            merkle_root = compute_merkle_root(coinbasetxn, tx_hashes);
            custom_coinbase = true;
        } else {
            // Use node's merkle root
            merkle_root = result.value("merkleroot", std::string(64, '0'));
        }

        // Build 80-byte block header
        work.header.resize(constants::BLOCK_HEADER_SIZE);

        // Version (4 bytes, little-endian)
        work.header[0] = version & 0xFF;
        work.header[1] = (version >> 8) & 0xFF;
        work.header[2] = (version >> 16) & 0xFF;
        work.header[3] = (version >> 24) & 0xFF;

        // Previous block hash (32 bytes, internal order)
        auto prev_bytes = hex_to_bytes(reverse_hex(prev_hash));
        if (prev_bytes.size() == 32) {
            std::copy(prev_bytes.begin(), prev_bytes.end(), work.header.begin() + 4);
        }

        // Merkle root (32 bytes)
        std::vector<uint8_t> merkle_bytes;
        if (custom_coinbase) {
            // Already in internal order from compute_merkle_root
            merkle_bytes = hex_to_bytes(merkle_root);
        } else {
            // RPC returns display order, need to reverse
            merkle_bytes = hex_to_bytes(reverse_hex(merkle_root));
        }
        if (merkle_bytes.size() == 32) {
            std::copy(merkle_bytes.begin(), merkle_bytes.end(), work.header.begin() + 36);
        }

        // Timestamp (4 bytes, little-endian)
        work.header[68] = curtime & 0xFF;
        work.header[69] = (curtime >> 8) & 0xFF;
        work.header[70] = (curtime >> 16) & 0xFF;
        work.header[71] = (curtime >> 24) & 0xFF;

        // Difficulty bits (4 bytes, little-endian)
        work.header[72] = bits & 0xFF;
        work.header[73] = (bits >> 8) & 0xFF;
        work.header[74] = (bits >> 16) & 0xFF;
        work.header[75] = (bits >> 24) & 0xFF;

        // Nonce placeholder (4 bytes)
        work.header[76] = 0;
        work.header[77] = 0;
        work.header[78] = 0;
        work.header[79] = 0;

        // Build full block hex for submission
        std::ostringstream block_ss;
        block_ss << bytes_to_hex(work.header.data(), 80);

        // Transaction count (varint)
        size_t tx_count = 1 + tx_data.size();
        if (tx_count < 253) {
            block_ss << std::hex << std::setfill('0') << std::setw(2) << tx_count;
        } else if (tx_count <= 0xFFFF) {
            block_ss << "fd" << std::hex << std::setfill('0')
                     << std::setw(2) << (tx_count & 0xFF)
                     << std::setw(2) << ((tx_count >> 8) & 0xFF);
        }

        // Transactions
        block_ss << coinbasetxn;
        for (const auto& tx : tx_data) {
            block_ss << tx;
        }

        work.block_hex = block_ss.str();
        work.valid = true;

    } catch (const std::exception& e) {
        if (g_config.verbose) {
            std::cerr << "get_work error: " << e.what() << std::endl;
        }
    }

    return work;
}

bool submit_block(const std::string& block_hex) {
    std::lock_guard<std::mutex> lock(g_submit_mutex);

    try {
        auto response = g_rpc->call("submitblock", nlohmann::json::array({block_hex}));

        if (response.contains("result") && response["result"].is_null()) {
            return true;
        }

        std::cerr << "Block rejected: " << response.dump() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Submit error: " << e.what() << std::endl;
    }

    return false;
}

//==============================================================================
// Statistics Display
//==============================================================================

void print_stats(std::chrono::steady_clock::time_point start_time) {
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time).count();

    std::cout << "\r";

    double total_rate = 0;
    for (int i = 0; i < g_num_gpus; i++) {
        double rate = g_gpu_hashes[i].load() / elapsed;
        total_rate += rate;

        std::string unit = "H/s";
        if (rate > 1e9) { rate /= 1e9; unit = "GH/s"; }
        else if (rate > 1e6) { rate /= 1e6; unit = "MH/s"; }
        else if (rate > 1e3) { rate /= 1e3; unit = "kH/s"; }

        std::cout << "GPU" << i << ": " << std::fixed << std::setprecision(1)
                  << rate << " " << unit << " | ";
    }

    std::string unit = "H/s";
    if (total_rate > 1e9) { total_rate /= 1e9; unit = "GH/s"; }
    else if (total_rate > 1e6) { total_rate /= 1e6; unit = "MH/s"; }
    else if (total_rate > 1e3) { total_rate /= 1e3; unit = "kH/s"; }

    std::cout << "Total: " << std::fixed << std::setprecision(2)
              << total_rate << " " << unit
              << " | Blocks: " << g_blocks_found.load() << "      " << std::flush;
}

//==============================================================================
// GPU Mining Thread
//==============================================================================

void mine_on_gpu(int gpu_index, int device_id, std::chrono::steady_clock::time_point start_time) {
    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    std::cout << "GPU " << gpu_index << " (device " << device_id << "): " << prop.name
              << " (" << (prop.totalGlobalMem / 1024 / 1024 / 1024) << " GB, "
              << prop.multiProcessorCount << " SMs)" << std::endl;

    int grid_size = get_optimal_grid_size(device_id, g_config.block_size);
    int block_size = g_config.block_size;

    // Allocate device memory
    uint32_t* d_header;
    uint32_t* d_target;
    uint32_t* d_result_nonce;
    uint32_t* d_result_found;

    CUDA_CHECK(cudaMalloc(&d_header, constants::BLOCK_HEADER_SIZE));
    CUDA_CHECK(cudaMalloc(&d_target, 32));
    CUDA_CHECK(cudaMalloc(&d_result_nonce, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_result_found, sizeof(uint32_t)));

    uint32_t header[20];
    uint32_t result_nonce = 0;
    uint32_t result_found = 0;

    auto last_stats = start_time;
    int current_height = 0;

    // Each GPU starts at different nonce range
    uint32_t nonce_offset = gpu_index * constants::NONCE_SPACE_PER_GPU;

    while (g_running) {
        MiningWork work = get_work();
        if (!work.valid) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }

        if (work.height != current_height) {
            current_height = work.height;
            if (gpu_index == 0) {
                std::cout << "\nMining block " << work.height << std::endl;
            }
        }

        std::memcpy(header, work.header.data(), constants::BLOCK_HEADER_SIZE);
        CUDA_CHECK(cudaMemcpy(d_header, header, constants::BLOCK_HEADER_SIZE, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_target, work.target, 32, cudaMemcpyHostToDevice));

        uint32_t nonce = nonce_offset;
        bool block_found = false;

        while (g_running && !block_found &&
               nonce < nonce_offset + constants::NONCE_SPACE_PER_GPU - grid_size * block_size) {

            result_found = 0;
            CUDA_CHECK(cudaMemcpy(d_result_found, &result_found, sizeof(uint32_t), cudaMemcpyHostToDevice));

            mine_kernel<<<grid_size, block_size>>>(
                d_header, d_target, nonce, d_result_nonce, d_result_found
            );

            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&result_found, d_result_found, sizeof(uint32_t), cudaMemcpyDeviceToHost));

            uint32_t hashes_done = grid_size * block_size;
            g_gpu_hashes[gpu_index] += hashes_done;
            g_total_hashes += hashes_done;
            nonce += hashes_done;

            if (result_found) {
                CUDA_CHECK(cudaMemcpy(&result_nonce, d_result_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost));

                std::cout << "\n*** GPU " << gpu_index << " FOUND BLOCK! Nonce: "
                          << result_nonce << " ***" << std::endl;

                // Update nonce in header
                work.header[76] = result_nonce & 0xFF;
                work.header[77] = (result_nonce >> 8) & 0xFF;
                work.header[78] = (result_nonce >> 16) & 0xFF;
                work.header[79] = (result_nonce >> 24) & 0xFF;

                // Build submission
                std::string new_header_hex = bytes_to_hex(work.header.data(), constants::BLOCK_HEADER_SIZE);
                std::string block_hex = new_header_hex + work.block_hex.substr(160);

                if (submit_block(block_hex)) {
                    g_blocks_found++;
                    if (!g_config.mining_address.empty()) {
                        std::cout << "Block accepted! Reward -> " << g_config.mining_address << std::endl;
                    } else {
                        std::cout << "Block accepted!" << std::endl;
                    }
                }

                block_found = true;
            }

            // GPU 0 prints stats
            if (gpu_index == 0) {
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration<double>(now - last_stats).count() >= 1.0) {
                    print_stats(start_time);
                    last_stats = now;
                }
            }
        }
    }

    // Cleanup
    cudaFree(d_header);
    cudaFree(d_target);
    cudaFree(d_result_nonce);
    cudaFree(d_result_found);
}

//==============================================================================
// CLI Interface
//==============================================================================

void print_usage(const char* prog) {
    std::cout << "TETSUO GPU Miner v" << constants::VERSION << " - SHA256d CUDA Mining\n\n";
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -a, --address ADDR   Mining address (rewards sent here)\n";
    std::cout << "  -o, --url URL        RPC URL (default: http://127.0.0.1:8337)\n";
    std::cout << "  -u, --user USER      RPC username\n";
    std::cout << "  -p, --pass PASS      RPC password\n";
    std::cout << "  -d, --device IDs     CUDA device IDs, comma-separated (default: all)\n";
    std::cout << "  -b, --block-size N   CUDA block size (default: 256)\n";
    std::cout << "  -v, --verbose        Verbose output\n";
    std::cout << "  -h, --help           Show this help\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog << " -a TRJvAZFx... -o http://localhost:8337\n";
    std::cout << "  " << prog << " -d 0,1 -u user -p pass\n";
}

void parse_args(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-a" || arg == "--address") {
            if (++i < argc) g_config.mining_address = argv[i];
        } else if (arg == "-o" || arg == "--url") {
            if (++i < argc) g_config.rpc_url = argv[i];
        } else if (arg == "-u" || arg == "--user") {
            if (++i < argc) g_config.rpc_user = argv[i];
        } else if (arg == "-p" || arg == "--pass") {
            if (++i < argc) g_config.rpc_pass = argv[i];
        } else if (arg == "-d" || arg == "--device") {
            if (++i < argc) {
                std::string devs = argv[i];
                std::stringstream ss(devs);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    g_config.devices.push_back(std::stoi(item));
                }
            }
        } else if (arg == "-b" || arg == "--block-size") {
            if (++i < argc) g_config.block_size = std::stoi(argv[i]);
        } else if (arg == "-v" || arg == "--verbose") {
            g_config.verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
    }
}

//==============================================================================
// Main
//==============================================================================

int main(int argc, char** argv) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    parse_args(argc, argv);

    // Setup signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Check for CUDA devices
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    // Print banner
    std::cout << "╔════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     TETSUO GPU Miner v" << constants::VERSION << "              ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════╝" << std::endl;

    if (!g_config.mining_address.empty()) {
        std::cout << "Mining to: " << g_config.mining_address << std::endl;
    }
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;

    // Use all devices if none specified
    if (g_config.devices.empty()) {
        for (int i = 0; i < device_count; i++) {
            g_config.devices.push_back(i);
        }
    }

    // Initialize GPU hash counters
    g_num_gpus = static_cast<int>(g_config.devices.size());
    for (int i = 0; i < g_num_gpus; i++) {
        g_gpu_hashes[i] = 0;
    }

    // Initialize CURL
    curl_global_init(CURL_GLOBAL_ALL);

    // Create RPC client
    g_rpc = std::make_unique<RpcClient>(
        g_config.rpc_url, g_config.rpc_user, g_config.rpc_pass);

    // Test connection
    try {
        auto response = g_rpc->call("getblockcount");
        int64_t height = response["result"].get<int64_t>();
        std::cout << "Connected to node, current height: " << height << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to connect to node at " << g_config.rpc_url << std::endl;
        std::cerr << "Error: " << e.what() << std::endl;
        curl_global_cleanup();
        return 1;
    }

    std::cout << std::endl;

    auto start_time = std::chrono::steady_clock::now();

    // Launch mining threads
    std::vector<std::thread> threads;
    for (size_t i = 0; i < g_config.devices.size(); i++) {
        threads.emplace_back(mine_on_gpu, static_cast<int>(i), g_config.devices[i], start_time);
    }

    // Wait for threads
    for (auto& t : threads) {
        t.join();
    }

    std::cout << "\n\nMining stopped." << std::endl;
    std::cout << "Total hashes: " << g_total_hashes.load() << std::endl;
    std::cout << "Blocks found: " << g_blocks_found.load() << std::endl;

    curl_global_cleanup();
    return 0;
}
