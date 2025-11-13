#!/usr/bin/env python3
"""
Install SOLLOL RPC server as systemd user service
"""
import os
import subprocess
import sys
from pathlib import Path


def install_rpc_service():
    """Install RPC server as systemd user service"""

    # Get user home directory
    home = Path.home()
    user = os.environ.get("USER", os.environ.get("USERNAME"))

    # Find rpc-server binary
    rpc_server_paths = [
        home / ".local/bin/rpc-server",
        home / "llama.cpp/build/bin/rpc-server",
        Path("/usr/local/bin/rpc-server"),
    ]

    rpc_server = None
    for path in rpc_server_paths:
        if path.exists():
            rpc_server = path
            break

    if not rpc_server:
        print("❌ Error: rpc-server binary not found!")
        print(f"   Searched in: {', '.join(str(p) for p in rpc_server_paths)}")
        print("\n   Please build llama.cpp first:")
        print("   python3 -m sollol.setup_llama_cpp --all")
        return False

    # Create systemd user directory
    systemd_dir = home / ".config/systemd/user"
    systemd_dir.mkdir(parents=True, exist_ok=True)

    # Create service file (for user service, don't specify User=)
    service_content = f"""[Unit]
Description=SOLLOL llama.cpp RPC Server
After=network.target
Documentation=https://github.com/B-A-M-N/SynapticLlamas

[Service]
Type=simple
WorkingDirectory={home}
ExecStart={rpc_server} --host 0.0.0.0 --port 50052
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
LimitNOFILE=65536

[Install]
WantedBy=default.target
"""

    service_file = systemd_dir / "sollol-rpc-server.service"
    service_file.write_text(service_content)

    print(f"✅ Created service file: {service_file}")

    # Reload systemd
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    print("✅ Reloaded systemd")

    # Enable and start service
    subprocess.run(["systemctl", "--user", "enable", "sollol-rpc-server.service"], check=True)
    print("✅ Enabled sollol-rpc-server service")

    subprocess.run(["systemctl", "--user", "start", "sollol-rpc-server.service"], check=True)
    print("✅ Started sollol-rpc-server service")

    # Enable lingering
    subprocess.run(["loginctl", "enable-linger", user], check=False)
    print("✅ Enabled user lingering (service runs even when not logged in)")

    print("\n" + "=" * 70)
    print("✅ SOLLOL RPC Server installed as systemd service!")
    print("=" * 70)
    print("\nUseful commands:")
    print("  systemctl --user status sollol-rpc-server    # Check status")
    print("  systemctl --user restart sollol-rpc-server   # Restart service")
    print("  systemctl --user stop sollol-rpc-server      # Stop service")
    print("  journalctl --user -u sollol-rpc-server -f    # View logs")
    print()

    return True


def main():
    """Main entry point"""
    try:
        success = install_rpc_service()
        sys.exit(0 if success else 1)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running command: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
