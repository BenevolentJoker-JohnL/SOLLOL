#!/usr/bin/env python3
"""
Setup script for llama.cpp RPC backend.

This script helps you:
1. Check if llama.cpp is installed
2. Build llama.cpp with RPC support
3. Start RPC servers
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_cmake():
    """Check if cmake is installed."""
    try:
        result = subprocess.run(["cmake", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_llama_cpp_exists(llama_dir):
    """Check if llama.cpp directory exists."""
    return Path(llama_dir).exists()


def clone_llama_cpp(install_dir):
    """Clone llama.cpp repository."""
    print("📥 Cloning llama.cpp...")
    result = subprocess.run(
        ["git", "clone", "https://github.com/ggerganov/llama.cpp", install_dir],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("✅ Cloned llama.cpp")
        print("ℹ️  Recommended: Update to build 6743+ for stability fixes")
        print("   Run: cd ~/llama.cpp && git pull")
    return result.returncode == 0


def build_llama_cpp(llama_dir):
    """Build llama.cpp with RPC support."""
    print("🔨 Building llama.cpp with RPC support...")

    build_dir = Path(llama_dir) / "build"

    # Configure
    print("   Configuring with CMake...")
    result = subprocess.run(
        ["cmake", "-B", "build", "-DGGML_RPC=ON", "-DLLAMA_CURL=OFF"],
        cwd=llama_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ CMake configuration failed:\n{result.stderr}")
        return False

    # Build
    print("   Building...")
    import multiprocessing

    nproc = multiprocessing.cpu_count()

    result = subprocess.run(
        [
            "cmake",
            "--build",
            "build",
            "--config",
            "Release",
            "--target",
            "rpc-server",
            f"-j{nproc}",
        ],
        cwd=llama_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ Build failed:\n{result.stderr}")
        return False

    print("✅ Build successful!")
    return True


def start_rpc_server(llama_dir, host="0.0.0.0", port=50052, mem_gb=None):
    """Start llama.cpp RPC server."""
    rpc_server = Path(llama_dir) / "build" / "bin" / "rpc-server"

    if not rpc_server.exists():
        print(f"❌ RPC server not found at {rpc_server}")
        print("   Please build llama.cpp first with --build")
        return False

    cmd = [str(rpc_server), "--host", host, "--port", str(port)]
    if mem_gb:
        cmd.extend(["--mem", str(mem_gb * 1024)])

    print(f"🚀 Starting RPC server on {host}:{port}...")
    print(f"   Command: {' '.join(cmd)}")
    print("   Press Ctrl+C to stop")

    try:
        subprocess.run(cmd, cwd=llama_dir)
    except KeyboardInterrupt:
        print("\n⏹️  RPC server stopped")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Setup and manage llama.cpp RPC backends for SOLLOL"
    )

    parser.add_argument(
        "--install-dir",
        default=os.path.expanduser("~/llama.cpp"),
        help="Directory to install llama.cpp (default: ~/llama.cpp)",
    )

    parser.add_argument("--clone", action="store_true", help="Clone llama.cpp repository")

    parser.add_argument("--build", action="store_true", help="Build llama.cpp with RPC support")

    parser.add_argument("--start", action="store_true", help="Start RPC server")

    parser.add_argument("--host", default="0.0.0.0", help="RPC server host (default: 0.0.0.0)")

    parser.add_argument("--port", type=int, default=50052, help="RPC server port (default: 50052)")

    parser.add_argument("--mem", type=int, help="Memory limit in GB (optional)")

    parser.add_argument(
        "--all", action="store_true", help="Clone, build, and install systemd service (full setup)"
    )

    parser.add_argument(
        "--service",
        action="store_true",
        help="Install as systemd service (recommended for production)",
    )

    parser.add_argument(
        "--gpu-monitoring",
        action="store_true",
        help="Also install GPU monitoring service (requires Redis)",
    )

    args = parser.parse_args()

    # Check cmake
    if not check_cmake():
        print("❌ cmake not found!")
        print("   Install it with: sudo apt-get install cmake build-essential")
        return 1

    print("✅ cmake found")

    install_dir = args.install_dir

    # Full setup
    if args.all:
        args.clone = True
        args.build = True
        args.service = True
        args.gpu_monitoring = True

    # Clone
    if args.clone:
        if check_llama_cpp_exists(install_dir):
            print(f"⚠️  llama.cpp already exists at {install_dir}")
            print("   Skipping clone...")
        else:
            if not clone_llama_cpp(install_dir):
                print("❌ Failed to clone llama.cpp")
                return 1
            print(f"✅ Cloned to {install_dir}")

    # Build
    if args.build:
        if not check_llama_cpp_exists(install_dir):
            print(f"❌ llama.cpp not found at {install_dir}")
            print("   Run with --clone first")
            return 1

        if not build_llama_cpp(install_dir):
            return 1

        # Copy to ~/.local/bin for easy systemd access
        import shutil

        local_bin = Path.home() / ".local/bin"
        local_bin.mkdir(parents=True, exist_ok=True)
        rpc_server_src = Path(install_dir) / "build/bin/rpc-server"
        rpc_server_dst = local_bin / "rpc-server"
        if rpc_server_src.exists():
            shutil.copy2(rpc_server_src, rpc_server_dst)
            rpc_server_dst.chmod(0o755)
            print(f"✅ Copied rpc-server to {rpc_server_dst}")

    # Install systemd service
    if args.service:
        if not check_llama_cpp_exists(install_dir):
            print(f"❌ llama.cpp not found at {install_dir}")
            print("   Run with --clone --build first")
            return 1

        # Import and run systemd installer
        try:
            from sollol.install_systemd_service import install_rpc_service

            if not install_rpc_service():
                return 1
        except ImportError:
            # Fallback if not installed as package
            print("⚠️  Installing systemd service manually...")
            script_path = Path(__file__).parent / "install_systemd_service.py"
            if script_path.exists():
                result = subprocess.run([sys.executable, str(script_path)])
                if result.returncode != 0:
                    return 1
            else:
                print("❌ install_systemd_service.py not found")
                return 1

    # Install GPU monitoring service
    if args.gpu_monitoring:
        print("\n" + "=" * 70)
        print("Installing GPU Monitoring Service")
        print("=" * 70)

        # Check Redis
        try:
            result = subprocess.run(
                ["redis-cli", "ping"], capture_output=True, text=True, timeout=2
            )
            if result.returncode != 0 or "PONG" not in result.stdout:
                print("❌ Redis not available")
                print("   Install Redis: sudo apt-get install redis-server")
                return 1
        except Exception:
            print("❌ Redis not available")
            print("   Install Redis: sudo apt-get install redis-server")
            return 1

        print("✅ Redis available")

        # Install Python dependencies
        print("📦 Installing Python dependencies (gpustat, redis)...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "gpustat", "redis"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return 1
        print("✅ Dependencies installed")

        # Run installation script
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        install_script = scripts_dir / "install-gpu-reporter-service.sh"

        if install_script.exists():
            print("🚀 Installing GPU reporter service...")
            result = subprocess.run(["bash", str(install_script)])
            if result.returncode != 0:
                print("❌ Failed to install GPU monitoring service")
                return 1
        else:
            print(f"⚠️  Installation script not found at {install_script}")
            print("   GPU monitoring service not installed")
            print("   Run manually: scripts/install-gpu-reporter-service.sh")

    # Start (interactive mode)
    if args.start:
        if not check_llama_cpp_exists(install_dir):
            print(f"❌ llama.cpp not found at {install_dir}")
            print("   Run with --clone --build first")
            return 1

        start_rpc_server(install_dir, args.host, args.port, args.mem)

    # No args - show help
    if not (
        args.clone or args.build or args.start or args.all or args.service or args.gpu_monitoring
    ):
        parser.print_help()
        print("\n" + "=" * 70)
        print("QUICK START:")
        print("=" * 70)
        print("\n1. Full setup (clone + build + systemd services):")
        print("   python setup_llama_cpp.py --all")
        print("\n2. Production setup (with GPU monitoring):")
        print("   python setup_llama_cpp.py --all --gpu-monitoring")
        print("\n3. Just start (if already built):")
        print("   python setup_llama_cpp.py --start")
        print("\n4. Custom port:")
        print("   python setup_llama_cpp.py --start --port 50053")
        print("\n5. Multi-node setup:")
        print("   # Node 1 (GPU node with monitoring):")
        print("   python setup_llama_cpp.py --all --gpu-monitoring")
        print("\n   # Node 2 (CPU-only node):")
        print("   python setup_llama_cpp.py --all")
        print("\n" + "=" * 70)
        print("RECOMMENDED BUILD:")
        print("=" * 70)
        print("llama.cpp build 6743+ includes stability fixes for distributed inference")
        print(
            "Update existing installation: cd ~/llama.cpp && git pull && python setup_llama_cpp.py --build"
        )
        print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
