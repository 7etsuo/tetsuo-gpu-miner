# TETSUO GPU Miner

High-performance CUDA GPU miner for the TETSUO blockchain. Supports multi-GPU mining with automatic work distribution.

## Features

- **Multi-GPU Support**: Automatically uses all available GPUs with optimized work distribution
- **SHA256d Mining**: Optimized double SHA256 implementation for proof-of-work
- **Custom Coinbase**: Mine directly to your wallet address
- **Modern CUDA**: Supports Ampere, Ada/Lovelace, Hopper, and Blackwell architectures
- **Clean Architecture**: Modular codebase with proper error handling

## Requirements

- CUDA Toolkit 12.0 or later
- CMake 3.24 or later
- libcurl development headers
- OpenSSL development headers
- C++17 compatible compiler

### Supported GPUs

| Architecture | Compute Capability | Example GPUs |
|--------------|-------------------|--------------|
| Ampere | 80, 86 | A100, RTX 30xx |
| Ada/Lovelace | 89 | RTX 40xx |
| Hopper | 90 | H100 |
| Blackwell | 100, 120 | B100, B200 |

## Building

```bash
# Clone the repository
git clone https://github.com/tetsuo/tetsuo-gpu-miner.git
cd tetsuo-gpu-miner

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# The binary is at build/tetsuo-miner
```

### Build Options

```bash
# Specify CUDA architectures
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="86;89"

# Debug build with verbose output
cmake -B build -DCMAKE_BUILD_TYPE=Debug
```

## Usage

```bash
# Basic usage (uses node's coinbase)
./tetsuo-miner -o http://127.0.0.1:8337 -u rpcuser -p rpcpass

# Mine to your address
./tetsuo-miner -a TYourAddressHere -o http://127.0.0.1:8337

# Use specific GPUs
./tetsuo-miner -d 0,2 -a TYourAddressHere

# Full example
./tetsuo-miner \
    --address TRJvAZFxtoJEjR6ipC15NuQ4LFq7gS5Qyu \
    --url http://127.0.0.1:8337 \
    --user myuser \
    --pass mypass
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `-a, --address ADDR` | Mining address (rewards sent here) |
| `-o, --url URL` | RPC URL (default: http://127.0.0.1:8337) |
| `-u, --user USER` | RPC username |
| `-p, --pass PASS` | RPC password |
| `-d, --device IDs` | CUDA device IDs, comma-separated (default: all) |
| `-b, --block-size N` | CUDA block size (default: 256) |
| `-v, --verbose` | Verbose output |
| `-h, --help` | Show help |

## Node Configuration

Your TETSUO node must be running with RPC enabled:

```ini
# ~/.tetsuo/tetsuo.conf
server=1
rpcuser=myuser
rpcpassword=mypass
rpcallowip=127.0.0.1
rpcport=8337
```

Start the node:
```bash
tetsuod -daemon
```

## Performance

Expected hashrates (approximate):

| GPU | Hashrate |
|-----|----------|
| RTX 4090 | ~8 GH/s |
| RTX 4080 | ~6 GH/s |
| RTX 3090 | ~5 GH/s |
| RTX 3080 | ~4 GH/s |

Multi-GPU setups scale linearly.

## Architecture

```
tetsuo-gpu-miner/
├── CMakeLists.txt
├── LICENSE
├── README.md
├── include/
│   ├── sha256.cuh           # CUDA SHA256 kernel
│   └── tetsuo/
│       ├── config.hpp       # Configuration
│       ├── cuda_utils.cuh   # CUDA error handling
│       ├── mining.hpp       # Mining structures
│       ├── rpc.hpp          # RPC client
│       └── utils.hpp        # Utility functions
└── src/
    ├── main.cu              # Main miner entry point
    ├── mining.cpp           # Mining functions
    ├── rpc.cpp              # RPC implementation
    └── utils.cpp            # Utility implementations
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
