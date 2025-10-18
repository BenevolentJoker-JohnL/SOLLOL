# Publishing SOLLOL to PyPI

This guide shows you how to publish SOLLOL to the Python Package Index (PyPI).

## Prerequisites

### 1. Create PyPI Accounts

You need accounts on both TestPyPI (for testing) and PyPI (for production):

- **TestPyPI** (testing): https://test.pypi.org/account/register/
- **PyPI** (production): https://pypi.org/account/register/

### 2. Install Build Tools

```bash
pip install --upgrade pip
pip install --upgrade build twine
```

## Step-by-Step Publishing

### Step 1: Clean Previous Builds

```bash
cd /home/joker/SynapticLlamas/sollol
rm -rf dist/ build/ *.egg-info
```

### Step 2: Build the Package

```bash
python -m build
```

This creates two files in `dist/`:
- `sollol-0.3.0.tar.gz` (source distribution)
- `sollol-0.3.0-py3-none-any.whl` (wheel distribution)

### Step 3: Test on TestPyPI (RECOMMENDED)

Always test on TestPyPI first to catch any issues:

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# You'll be prompted for:
# Username: your TestPyPI username
# Password: your TestPyPI password (or API token)
```

**Test the installation:**

```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ sollol

# Test it works
python -c "from sollol import OllamaPool; print('✅ SOLLOL imported successfully!')"
```

### Step 4: Publish to PyPI

Once TestPyPI works, publish to the real PyPI:

```bash
# Upload to PyPI
python -m twine upload dist/*

# You'll be prompted for:
# Username: your PyPI username
# Password: your PyPI password (or API token)
```

### Step 5: Verify

```bash
# Install from PyPI
pip install sollol

# Test it
python -c "from sollol import OllamaPool; print('✅ SOLLOL installed from PyPI!')"
```

## Using API Tokens (RECOMMENDED)

API tokens are more secure than passwords.

### Create API Tokens

1. **TestPyPI**: https://test.pypi.org/manage/account/token/
2. **PyPI**: https://pypi.org/manage/account/token/

### Configure .pypirc

Create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-PRODUCTION-TOKEN-HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TEST-TOKEN-HERE
```

**Set permissions:**
```bash
chmod 600 ~/.pypirc
```

Now you can upload without entering credentials:

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

## Quick Publish Script

Use the included helper script:

```bash
# Test publish (TestPyPI)
python publish.py --test

# Real publish (PyPI)
python publish.py --production
```

## Updating the Package

When releasing a new version:

1. **Update version** in `setup.py` and `pyproject.toml`
2. **Clean old builds:** `rm -rf dist/ build/ *.egg-info`
3. **Build:** `python -m build`
4. **Test on TestPyPI:** `twine upload --repository testpypi dist/*`
5. **Publish to PyPI:** `twine upload dist/*`

## Common Issues

### Issue: "File already exists"

**Solution:** You can't re-upload the same version. Increment the version number.

### Issue: "Invalid credentials"

**Solution:**
- Make sure you're using the correct username/password
- Or use API tokens (recommended)
- Check your `~/.pypirc` file

### Issue: "Package name already taken"

**Solution:** The name "sollol" must be available on PyPI. Check https://pypi.org/project/sollol/

If taken, you'll need to:
- Choose a different name (e.g., "sollol-lb", "super-ollama-lb")
- Update `name` in `setup.py` and `pyproject.toml`

## Verification Checklist

Before publishing, verify:

- [ ] Version number is correct
- [ ] README.md is complete and accurate
- [ ] LICENSE file is present
- [ ] All dependencies are listed
- [ ] Package builds without errors: `python -m build`
- [ ] Tested on TestPyPI first
- [ ] Package name is available on PyPI

## After Publishing

Once published, users can install with:

```bash
pip install sollol

# Or with llama.cpp setup
pip install sollol
sollol-setup-llama-cpp --all
```

## Resources

- PyPI: https://pypi.org/
- TestPyPI: https://test.pypi.org/
- Python Packaging Guide: https://packaging.python.org/
- Twine docs: https://twine.readthedocs.io/
