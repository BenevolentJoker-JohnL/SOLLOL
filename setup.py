"""Setup script for SOLLOL - Super Ollama Load Balancer."""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else "SOLLOL - Super Ollama Load Balancer with Distributed Inference"

setup(
    name="sollol",
    version="0.9.52",
    author="B-A-M-N",
    author_email="benevolentjoker@gmail.com",
    description="Super Ollama Load Balancer with Intelligent Routing and Distributed Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/B-A-M-N/SOLLOL",
    project_urls={
        "Bug Tracker": "https://github.com/B-A-M-N/SOLLOL/issues",
        "Documentation": "https://github.com/B-A-M-N/SOLLOL/blob/main/README.md",
        "Source Code": "https://github.com/B-A-M-N/SOLLOL",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.31.0",
        "httpx>=0.24.0",
        "ipaddress>=1.0.23",
        "dask[distributed]>=2024.1.0",
        "ray[default]>=2.0.0",
        "prometheus-client>=0.12.0",
        "rich>=13.7.0",
        "flask>=2.0.0",
        "flask-cors>=4.0.0",
        "flask-sock>=0.7.0",
        "gevent>=23.9.0",
        "gevent-websocket>=0.10.1",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "starlette>=0.27.0",
        "bokeh>=3.1.0",
        "gpustat>=1.0.0",
        "redis>=5.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
        "llama-cpp": [
            # Optional: for llama.cpp distributed inference
        ],
    },
    entry_points={
        "console_scripts": [
            "sollol=sollol.cli:main",
            "sollol-setup-llama-cpp=sollol.setup_llama_cpp:main",
            "sollol-install-service=sollol.install_systemd_service:main",
        ],
    },
    include_package_data=True,
    keywords="ai llm distributed ollama load-balancing inference llama-cpp distributed-inference",
    zip_safe=False,
)
