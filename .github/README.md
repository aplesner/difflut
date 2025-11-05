# DiffLUT GitHub Configuration

This directory contains GitHub-specific configuration files for automated workflows, issue templates, and pull request templates.

## Contents

### 📋 Workflows (`.github/workflows/`)

Automated CI/CD pipelines:

- **`tests.yml`** - Comprehensive test suite
  - Runs on all Python versions (3.8-3.12)
  - Tests CPU and GPU builds
  - Generates coverage reports
  
- **`version-management.yml`** - Version control
  - Validates version consistency
  - Syncs versions across files
  - Checks for version bumps in PRs
  
- **`release.yml`** - Automated releases
  - Creates GitHub releases
  - Publishes to PyPI
  - Generates changelogs
  
- **`pr-checks.yml`** - Pull request validation
  - Linting (black, flake8, isort)
  - Documentation building
  - Version bump reminders

### 📖 Documentation

- **`CICD_SETUP.md`** - Complete CI/CD setup guide
  - Workflow overview
  - Setup instructions
  - Troubleshooting guide
  - Usage examples with bump2version

## Quick Start

### 1. Initial Setup
```bash
# These workflows are already configured
# Just ensure you have proper GitHub secrets set up
```

### 2. Configure PyPI Publishing (Optional)
```
Settings → Secrets and variables → Actions → New repository secret
Name: PYPI_API_TOKEN
Value: (your PyPI token)
```

### 3. Use bump2version for Releases
```bash
cd difflut

# Install bump2version
pip install bump2version

# Bump version (creates commit and tag)
bump2version patch    # 1.1.2 → 1.1.3
bump2version minor    # 1.1.2 → 1.2.0
bump2version major    # 1.1.2 → 2.0.0

# Push changes
git push origin develop
# Create PR to main
# Once merged, GitHub Actions automatically:
# - Validates versions
# - Creates GitHub Release
# - Publishes to PyPI
```

## Workflows Summary

| Workflow | Trigger | Purpose | Python Versions |
|----------|---------|---------|-----------------|
| tests.yml | Push/PR | Test code | 3.8, 3.9, 3.10, 3.11, 3.12 |
| version-management.yml | Version change | Validate/sync versions | - |
| release.yml | Version bump on main | Create release & PyPI | - |
| pr-checks.yml | Pull request | Lint, docs, version check | 3.11 |

## Version Management Workflow

```
1. Make changes in feature branch
   ↓
2. Bump version with bump2version
   - Updates: setup.py, pyproject.toml, __init__.py
   - Creates: version commit and git tag
   ↓
3. Create Pull Request
   - PR checks run (lint, docs, version validation)
   ↓
4. Merge to main
   - Tests run (all Python versions, CPU & GPU)
   - Version consistency validated
   ↓
5. GitHub Actions triggers release
   - Creates GitHub Release with changelog
   - Publishes to PyPI (if PYPI_API_TOKEN set)
```

## Key Features

✅ **Automated Testing**
- All Python versions (3.8-3.12)
- CPU and GPU builds
- Coverage reporting

✅ **Version Management**
- Automatic version synchronization
- Version bump reminders in PRs
- Consistent versioning across files

✅ **Automated Releases**
- GitHub Release creation
- PyPI publishing
- Changelog generation

✅ **Code Quality**
- Linting with black, flake8, isort
- Documentation building
- PR validation

## For More Information

See **[`CICD_SETUP.md`](./CICD_SETUP.md)** for:
- Detailed setup instructions
- Complete workflow documentation
- Troubleshooting guide
- Advanced configuration

## GitHub Settings Required

### Recommended Settings

**Protect main branch:**
1. Settings → Branches → Add rule for `main`
2. Require pull request reviews before merging
3. Require status checks to pass before merging
4. Require branches to be up to date

**Allow:**
- Auto-merge for pull requests
- Delete head branches on merge

These settings ensure code quality and prevent merging broken changes.

---

**Questions?** See the main [Documentation](../docs/) or contact: sbuehrer@ethz.ch
