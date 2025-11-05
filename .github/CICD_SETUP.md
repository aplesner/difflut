# GitHub Actions CI/CD Setup Guide

This guide explains the automated CI/CD workflows configured for DiffLUT.

## Workflows Overview

### 1. **tests.yml** - Continuous Testing
Runs automatically on every push and pull request.

**Features:**
- Tests all Python versions: 3.8, 3.9, 3.10, 3.11, 3.12
- CPU tests run for all versions
- GPU tests run for Python 3.10 and 3.11 on main branch and PRs
- Generates coverage reports and uploads to Codecov
- Tests both CPU and GPU build configurations

**Matrix:**
| Python | CPU | GPU | PyTorch |
|--------|-----|-----|---------|
| 3.8    | ✅  | -   | 1.13.1  |
| 3.9    | ✅  | -   | 1.13.1  |
| 3.10   | ✅  | ✅  | 2.0.1   |
| 3.11   | ✅  | ✅  | 2.0.1   |
| 3.12   | ✅  | -   | 2.1.2   |

### 2. **version-management.yml** - Version Control
Ensures version consistency and manages version bumps.

**Features:**
- Validates version consistency across:
  - `setup.py`
  - `pyproject.toml`
  - `difflut/__init__.py`
- Checks for version bumps in PRs
- Auto-syncs versions when they change on main branch
- Creates automated PRs for version synchronization

### 3. **release.yml** - Automated Releases
Creates releases and publishes to PyPI when version changes.

**Features:**
- Detects version changes on main branch
- Generates changelog from commits
- Creates GitHub Release with changelog
- Publishes to PyPI (requires `PYPI_API_TOKEN`)

**Requires:** GitHub secret `PYPI_API_TOKEN` for PyPI publishing

### 4. **pr-checks.yml** - Pull Request Validation
Runs code quality checks on every PR.

**Features:**
- **Linting:** black, flake8, isort
- **Version Reminder:** Suggests version bump if needed
- **Documentation:** Builds docs and checks for warnings
- **Code Style:** Enforces consistent formatting

## Setup Instructions

### 1. Initial Repository Setup

Push the `.github/workflows/` directory to your repository:

```bash
cd /itet-stor/sbuehrer/net_scratch/difflut
git add .github/workflows/
git commit -m "ci: add github actions workflows"
git push origin main
```

### 2. Ensure Version Files are Consistent

Before first deployment, ensure all version files are synchronized:

```bash
cd difflut

# Current version (from pyproject.toml)
VERSION=$(grep "version = " pyproject.toml | head -1 | sed 's/.*version = "//;s/".*//')
echo "Current version: $VERSION"

# Verify setup.py has matching version
grep "version=" setup.py | head -1

# Verify __init__.py has matching version
grep "__version__" difflut/__init__.py
```

### 3. Configure GitHub Secrets (Required for PyPI Publishing)

If you want automatic PyPI publishing:

1. Go to GitHub repository → Settings → Secrets and variables → Actions
2. Add new secret: `PYPI_API_TOKEN`
   - Value: Your PyPI API token (create at https://pypi.org/manage/account/tokens/)
3. Save

Without this secret, releases will be created on GitHub but not published to PyPI.

### 4. Setup Codecov Integration (Optional)

For coverage reports on pull requests:

1. Go to https://codecov.io
2. Connect your GitHub account
3. Enable codecov for your repository
4. Codecov will automatically pick up coverage reports from GitHub Actions

## Workflow: Using bump2version for Version Control

### Updating Version Locally

Use `bump2version` to manage versions:

```bash
cd difflut

# Install bump2version
pip install bump2version

# View current version
bump2version --list

# Bump patch version (bug fixes)
bump2version patch   # 1.1.2 → 1.1.3

# Bump minor version (new features)
bump2version minor   # 1.1.2 → 1.2.0

# Bump major version (breaking changes)
bump2version major   # 1.1.2 → 2.0.0
```

### Push and Release

After bumping version:

```bash
# Push to your branch
git push origin feature-branch

# Create pull request
# (GitHub web UI or gh CLI)

# Once merged to main, GitHub Actions will:
# 1. Validate version consistency
# 2. Create GitHub Release with changelog
# 3. Publish to PyPI (if PYPI_API_TOKEN is set)
```

## Understanding Test Matrices

### CPU Testing
All Python versions (3.8-3.12) are tested on CPU:
- Tests run on `ubuntu-latest`
- No GPU required
- Faster feedback for PRs

### GPU Testing
Selective Python versions (3.10-3.11) with CUDA:
- Tests run on `ubuntu-latest` with NVIDIA CUDA toolkit
- Tests build with CUDA support
- Only runs on:
  - Pull requests
  - Pushes to `main` branch
- Slower but validates GPU compatibility

## Troubleshooting

### Tests Failing

1. **Check logs:** Click on the failing workflow run in GitHub → View details
2. **Local testing:** Run tests locally before pushing
   ```bash
   cd difflut
   pytest tests/ -v
   ```
3. **Version mismatch:** Ensure versions are synchronized:
   ```bash
   bump2version --list
   ```

### PyPI Publishing Not Working

1. **Check secret:** Verify `PYPI_API_TOKEN` is set in GitHub Secrets
2. **Check token:** Ensure PyPI token hasn't expired
3. **Check logs:** View release workflow logs for error details

### GPU Tests Not Running

GPU tests only run on:
- Pull requests to `main` or `develop`
- Pushes to `main` branch

If you need GPU tests on other branches, edit `.github/workflows/tests.yml`:

```yaml
test-gpu:
  if: github.event_name == 'pull_request' || github.ref == 'refs/heads/main'
  # Change to: if: always()
```

## File Structure

```
.github/
├── workflows/
│   ├── tests.yml                  # CPU/GPU testing
│   ├── version-management.yml     # Version validation
│   ├── release.yml                # Automated releases
│   └── pr-checks.yml              # PR validation
```

## Next Steps

1. ✅ **Push workflows to GitHub**
   ```bash
   git add .github/workflows/
   git commit -m "ci: add github actions workflows"
   git push
   ```

2. ✅ **Setup PyPI token** (optional, for automatic publishing)
   - Go to GitHub Secrets and add `PYPI_API_TOKEN`

3. ✅ **Test a pull request** to verify workflows work

4. ✅ **Use bump2version** for version management:
   ```bash
   bump2version patch
   ```

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [bump2version Documentation](https://github.com/c4urself/bump2version)
- [PyPI API Tokens](https://pypi.org/manage/account/tokens/)
- [Codecov Documentation](https://docs.codecov.io/)

---

For questions or issues, see the DiffLUT documentation or contact: sbuehrer@ethz.ch
