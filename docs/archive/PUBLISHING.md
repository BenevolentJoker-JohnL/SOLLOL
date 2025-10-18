# Publishing SOLLOL to PyPI

Guide for publishing new versions of SOLLOL to PyPI.

## Prerequisites

1. **PyPI Account**: Create account at https://pypi.org/
2. **Test PyPI Account**: Create account at https://test.pypi.org/
3. **GitHub Secrets**: Configure trusted publishing (recommended)

## Trusted Publishing Setup (Recommended)

### 1. Configure PyPI

1. Go to https://pypi.org/manage/account/publishing/
2. Add a new pending publisher:
   - **PyPI Project Name**: `sollol`
   - **Owner**: `BenevolentJoker-JohnL`
   - **Repository name**: `SOLLOL`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi`

### 2. Configure Test PyPI

1. Go to https://test.pypi.org/manage/account/publishing/
2. Add a new pending publisher with same settings
3. Use environment name: `testpypi`

## Publishing Workflow

### Test on Test PyPI First

```bash
# 1. Update version in pyproject.toml
# Edit: version = "0.1.1"

# 2. Build locally
python -m pip install --upgrade build
python -m build

# 3. Test the package
python -m pip install --upgrade twine
twine check dist/*

# 4. Trigger Test PyPI publish via GitHub Actions
# Go to: Actions → Publish to PyPI → Run workflow
# Select: "Publish to Test PyPI instead" = true

# 5. Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ sollol
```

### Publish to Production PyPI

```bash
# 1. Create a new release on GitHub
# Go to: Releases → Draft a new release

# 2. Create a new tag (e.g., v0.1.0)
git tag v0.1.0
git push origin v0.1.0

# 3. Fill in release notes
# - What's new
# - Bug fixes
# - Breaking changes

# 4. Publish release
# This automatically triggers the publish.yml workflow
# Package is built and published to PyPI
```

## Manual Publishing (Fallback)

If automated publishing fails:

```bash
# 1. Build the package
python -m build

# 2. Upload to Test PyPI
twine upload --repository testpypi dist/*

# 3. Test installation
pip install --index-url https://test.pypi.org/simple/ sollol

# 4. Upload to PyPI
twine upload dist/*
```

## Version Numbering

Follow Semantic Versioning (SemVer):

- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.2.0): New features, backward compatible
- **PATCH** (0.1.1): Bug fixes, backward compatible

### Pre-release versions

- **Alpha**: `0.1.0a1`, `0.1.0a2`
- **Beta**: `0.1.0b1`, `0.1.0b2`
- **Release Candidate**: `0.1.0rc1`, `0.1.0rc2`

## Checklist Before Publishing

- [ ] Update version in `pyproject.toml`
- [ ] Update CHANGELOG.md with changes
- [ ] Run all tests: `pytest tests/`
- [ ] Run linting: `black src/ && flake8 src/`
- [ ] Build package: `python -m build`
- [ ] Check package: `twine check dist/*`
- [ ] Test on Test PyPI first
- [ ] Verify installation works
- [ ] Create GitHub release with notes
- [ ] Tag matches version in pyproject.toml

## Post-Publishing

1. **Verify on PyPI**: Check https://pypi.org/project/sollol/
2. **Test installation**: `pip install sollol`
3. **Update documentation**: Ensure docs reflect new version
4. **Announce**: Create GitHub discussion, tweet, etc.

## Troubleshooting

### "File already exists" error

You can't re-upload the same version. Increment version number.

### Trusted publishing not working

Fall back to API tokens:
1. Create API token on PyPI
2. Add to GitHub secrets as `PYPI_API_TOKEN`
3. Update publish.yml to use token authentication

### Build fails

```bash
# Clean dist directory
rm -rf dist/ build/ *.egg-info

# Rebuild
python -m build
```

### Dependencies missing

Ensure all dependencies are listed in `pyproject.toml`:
```toml
dependencies = [
    "dependency>=version",
]
```

## Resources

- **PyPI Project**: https://pypi.org/project/sollol/
- **Test PyPI**: https://test.pypi.org/project/sollol/
- **Packaging Guide**: https://packaging.python.org/
- **Trusted Publishing**: https://docs.pypi.org/trusted-publishers/

---

**Last Updated**: 2025-10-03
