# Contributing to SOLLOL

Thank you for your interest in contributing to SOLLOL! This document provides guidelines and best practices for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## Code of Conduct

This project follows a simple code of conduct:

- **Be respectful**: Treat all contributors with respect and professionalism
- **Be constructive**: Provide helpful feedback and suggestions
- **Be collaborative**: Work together to improve the project
- **Be patient**: Remember that everyone is learning

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/SOLLOL.git
   cd SOLLOL
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/B-A-M-N/SOLLOL.git
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.8+
- Ray (distributed computing)
- Dask (batch processing)
- FastAPI (web framework)
- Ollama (AI model inference)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (recommended)
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=sollol --cov-report=html

# Run specific test file
pytest tests/test_intelligence.py

# Run with verbose output
pytest tests/ -v
```

### Running SOLLOL Locally

```bash
# Start with default configuration
python -m sollol.cli up

# Or use custom configuration
python -m sollol.cli up --workers 4 --port 8000
```

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces
- **Quotes**: Use double quotes for strings
- **Imports**: Group in order: stdlib, third-party, local

### Code Organization

```python
"""
Module docstring explaining purpose and usage.
"""
# Standard library imports
import asyncio
from typing import Dict, List, Optional

# Third-party imports
from fastapi import FastAPI
import ray

# Local imports
from sollol.intelligence import get_router
from sollol.memory import get_best_host

# Constants
DEFAULT_PORT = 8000
MAX_RETRIES = 3

# Classes and functions
class MyClass:
    """Class docstring with description."""

    def __init__(self, param: str):
        """Initialize with parameters."""
        self.param = param

    def my_method(self) -> Dict:
        """Method docstring describing behavior."""
        pass
```

### Type Hints

Always use type hints for function parameters and return values:

```python
def process_request(
    payload: Dict,
    priority: int = 5,
    timeout: float = 30.0
) -> Dict[str, Any]:
    """Process a request with given parameters."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_score(host_meta: Dict, context: TaskContext) -> float:
    """
    Calculate routing score for a host based on context.

    Args:
        host_meta: Host metadata including performance metrics
        context: Task context with requirements and preferences

    Returns:
        Score value (higher is better)

    Raises:
        ValueError: If host_meta is invalid

    Example:
        >>> score = calculate_score(host_meta, context)
        >>> print(f"Score: {score}")
        Score: 185.3
    """
    pass
```

## Commit Message Guidelines

We follow **Conventional Commits** specification:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no logic changes)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(routing): add GPU memory weighting to scoring algorithm

Implement GPU memory consideration in host scoring to prefer
nodes with more available GPU memory for GPU-intensive tasks.

Closes #42
```

```
fix(gateway): handle connection timeout in health checks

Add explicit timeout handling for health check requests to
prevent indefinite hangs.

Fixes #38
```

```
docs(readme): add performance benchmark section

Include detailed performance benchmarks comparing SOLLOL with
round-robin and random routing strategies.
```

### Commit Message Rules

- Use present tense ("add feature" not "added feature")
- Use imperative mood ("move cursor to..." not "moves cursor to...")
- First line should be 50-72 characters
- Reference issues and pull requests when relevant
- Body should explain *what* and *why*, not *how*

## Pull Request Process

### Before Submitting

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests locally**:
   ```bash
   pytest tests/
   ```

3. **Check code style**:
   ```bash
   black src/sollol/
   flake8 src/sollol/
   mypy src/sollol/
   ```

4. **Update documentation** if needed

### Submitting a Pull Request

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub with:
   - Clear title following commit message guidelines
   - Detailed description of changes
   - Screenshots/demos if UI changes
   - Reference to related issues

3. **PR Description Template**:
   ```markdown
   ## Description
   Brief description of what this PR does.

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing
   Describe how you tested these changes.

   ## Checklist
   - [ ] Code follows project style guidelines
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] All tests pass
   - [ ] No new warnings

   ## Related Issues
   Closes #XX
   ```

### Review Process

- Maintainers will review your PR within 3-5 business days
- Address feedback by pushing new commits to your branch
- Once approved, a maintainer will merge your PR
- PRs require at least one approval before merging

## Testing Guidelines

### Test Structure

```
tests/
├── unit/
│   ├── test_intelligence.py
│   ├── test_prioritization.py
│   └── test_memory.py
├── integration/
│   ├── test_gateway.py
│   └── test_end_to_end.py
└── conftest.py  # Shared fixtures
```

### Writing Tests

```python
import pytest
from sollol.intelligence import IntelligentRouter, TaskContext

class TestIntelligentRouter:
    """Test suite for IntelligentRouter."""

    @pytest.fixture
    def router(self):
        """Create router instance for testing."""
        return IntelligentRouter()

    @pytest.fixture
    def sample_payload(self):
        """Sample request payload."""
        return {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Hello"}]
        }

    def test_detect_task_type_generation(self, router, sample_payload):
        """Test task type detection for generation tasks."""
        task_type = router.detect_task_type(sample_payload)
        assert task_type == "generation"

    def test_estimate_complexity_simple(self, router):
        """Test complexity estimation for simple requests."""
        payload = {"messages": [{"role": "user", "content": "Hi"}]}
        complexity, tokens = router.estimate_complexity(payload)
        assert complexity == "simple"
        assert tokens < 500
```

### Test Coverage

- Aim for **>80% test coverage**
- Focus on critical paths and edge cases
- Include both positive and negative test cases
- Test error handling and boundary conditions

## Documentation

### When to Update Documentation

- Adding new features
- Changing existing behavior
- Adding new API endpoints
- Modifying configuration options
- Adding new dependencies

### Documentation Locations

- **README.md**: Overview, quick start, basic usage
- **ARCHITECTURE.md**: System design, components, scaling
- **API docs**: Auto-generated from FastAPI (http://localhost:8000/docs)
- **Code comments**: Explain *why*, not *what*
- **Docstrings**: Function/class behavior and usage

### Documentation Style

- Use clear, concise language
- Include code examples
- Add diagrams for complex concepts
- Keep examples up-to-date with code changes

## Areas for Contribution

### High Priority

- **Testing**: Expand test coverage for all modules
- **Performance**: Optimize routing algorithm and metrics collection
- **Monitoring**: Enhanced Prometheus metrics and dashboards
- **Documentation**: More examples and use cases

### Feature Ideas

- ML-based routing prediction
- Cost-aware routing for cloud deployments
- Geographic/latency-aware routing
- A/B testing framework
- Auto-scaling based on queue depth
- WebSocket support for streaming responses

### Good First Issues

Look for issues tagged with `good-first-issue` in the GitHub issue tracker. These are well-documented, straightforward issues perfect for new contributors.

## Questions?

- **GitHub Discussions**: Ask questions and share ideas
- **GitHub Issues**: Report bugs or request features
- **Email**: Contact maintainers at [your-email]

---

Thank you for contributing to SOLLOL! Your efforts help make intelligent AI orchestration accessible to everyone.
