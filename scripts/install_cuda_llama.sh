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

# Build llama.cpp RPC server with CUDA support
echo ""
echo "🔨 Building llama.cpp RPC server with CUDA support..."
cd ~/llama.cpp
rm -rf build
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DGGML_RPC=ON

cmake --build build --config Release --target rpc-server -j $(nproc)

# Install binaries to ~/.local/bin
echo ""
echo "📦 Installing CUDA rpc-server binary to ~/.local/bin..."
mkdir -p ~/.local/bin
cp build/bin/rpc-server ~/.local/bin/rpc-server

# Check binary size
echo ""
echo "✅ Verifying build..."
ls -lh ~/.local/bin/rpc-server
ldd ~/.local/bin/rpc-server | grep -i cuda || echo "⚠️  CUDA libraries not found (expected on CPU-only build machine)"

echo ""
echo "========================================"
echo "✅ Installation Complete!"
echo "========================================"
echo ""
echo "CUDA-enabled RPC server built:"
echo "  - rpc-server (~/.local/bin/rpc-server, ~689MB)"
echo ""
echo "Build configuration:"
echo "  - CUDA architectures: 75,80,86,89,90 (Turing→Hopper)"
echo "  - Static libraries for portability"
echo "  - RPC backend support enabled"
echo ""
echo "⚠️  IMPORTANT:"
echo "  - This CUDA binary REQUIRES NVIDIA drivers on target nodes"
echo "  - Will NOT run on CPU-only coordinator without drivers"
echo "  - Deploy to GPU nodes (10.9.66.90, etc.)"
echo ""
echo "Next steps:"
echo "1. Copy binary to GPU nodes:"
echo "   scp ~/.local/bin/rpc-server <gpu-node>:~/.local/bin/"
echo ""
echo "2. On GPU nodes, start RPC server:"
echo "   nohup ~/.local/bin/rpc-server --host 0.0.0.0 --port 50052 > /tmp/rpc-server.log 2>&1 &"
echo ""
echo "3. For CPU-only coordinator, build separate CPU binary:"
echo "   cmake -B build-cpu -DGGML_CUDA=OFF -DLLAMA_BUILD_TOOLS=ON -DGGML_RPC=ON"
echo "   cmake --build build-cpu --target rpc-server -j $(nproc)"
echo ""
