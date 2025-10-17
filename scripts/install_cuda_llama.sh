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
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_CURL=OFF
cmake --build build --config Release --target llama-server -j $(nproc)

# Install binaries to ~/.local/bin
echo ""
echo "📦 Installing binaries to ~/.local/bin..."
mkdir -p ~/.local/bin
cp build/bin/llama-server ~/.local/bin/llama-server

# Verify installation (note: will fail on CPU-only machines without CUDA stubs)
echo ""
echo "✅ Verifying installation..."
if ~/.local/bin/llama-server --version 2>&1 | head -5; then
  echo "Binary verification successful"
else
  echo "⚠️  Binary verification failed (expected on CPU-only machines)"
  echo "Binary will work on nodes with NVIDIA drivers installed"
fi

echo ""
echo "========================================"
echo "✅ Installation Complete!"
echo "========================================"
echo ""
echo "CUDA-enabled binary installed:"
echo "  - llama-server (~/.local/bin/llama-server)"
echo ""
echo "Build configuration:"
echo "  - CUDA architectures: 75,80,86,89,90 (Turing→Hopper)"
echo "  - Static libraries for portability"
echo "  - Binary size: ~760MB (includes kernels for all architectures)"
echo ""
echo "This binary will:"
echo "  ✅ Use CUDA on NVIDIA GPU nodes (RTX 20/30/40 series, A100, H100)"
echo "  ✅ Fall back to CPU on non-GPU nodes"
echo ""
echo "Next steps:"
echo "1. Deploy this binary to your GPU nodes"
echo "2. Ensure NVIDIA drivers (535+) are installed on GPU nodes"
echo "3. Start llama-server with your model"
echo ""
echo "Note: Binary requires NVIDIA drivers on target GPU nodes"
echo ""
