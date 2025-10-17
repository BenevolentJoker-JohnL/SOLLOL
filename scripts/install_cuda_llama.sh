#!/bin/bash
# SOLLOL - Install CUDA and build llama.cpp with GPU support
# Run this script with: bash /tmp/install_cuda_and_build_llama.sh

set -e

echo "========================================"
echo "SOLLOL: CUDA + llama.cpp Installation"
echo "========================================"
echo ""

# Install CUDA keyring
echo "📦 Installing CUDA repository..."
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update

# Install CUDA toolkit (minimal, ~3GB)
echo "📦 Installing CUDA toolkit (this will take a few minutes)..."
sudo apt-get install -y cuda-toolkit-12-6

# Add CUDA to PATH
echo "🔧 Adding CUDA to PATH..."
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# Verify CUDA installation
echo ""
echo "✅ Verifying CUDA installation..."
nvcc --version

# Build llama.cpp with CUDA support
echo ""
echo "🔨 Building llama.cpp with CUDA support..."
cd /tmp/llama.cpp-gpu
rm -rf build
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release -j $(nproc)

# Install binaries to ~/.local/bin
echo ""
echo "📦 Installing binaries to ~/.local/bin..."
mkdir -p ~/.local/bin
cp build/bin/llama-server ~/.local/bin/llama-server
cp build/bin/llama-cli ~/.local/bin/llama-cli
cp build/bin/rpc-server ~/.local/bin/rpc-server

# Verify installation
echo ""
echo "✅ Verifying installation..."
~/.local/bin/llama-server --version 2>&1 | head -5
~/.local/bin/rpc-server --version 2>&1 | head -5

echo ""
echo "========================================"
echo "✅ Installation Complete!"
echo "========================================"
echo ""
echo "CUDA-enabled binaries installed:"
echo "  - llama-server (coordinator with CUDA)"
echo "  - llama-cli (inference with CUDA)"
echo "  - rpc-server (RPC backend with CUDA)"
echo ""
echo "These binaries will:"
echo "  ✅ Use CUDA on NVIDIA GPU nodes"
echo "  ✅ Fall back to CPU on non-GPU nodes"
echo ""
echo "Next steps:"
echo "1. Run: python scripts/setup_rpc_node.py"
echo "2. Start RPC server with generated command"
echo "3. Enjoy GPU-accelerated inference! 🚀"
echo ""
