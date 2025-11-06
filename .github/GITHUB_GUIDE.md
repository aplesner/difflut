# GitHub Guide

This guide explains the workflow for contributing to DiffLUT, including branching strategies, version management, and CI/CD pipelines.

## 📋 Table of Contents
- [Contributing Workflow](#contributing-workflow)
- [Branch Naming Conventions](#branch-naming-conventions)
- [Version Management](#version-management)
- [GitHub Actions CI/CD](#github-actions-cicd)
- [Creating Custom Workflows](#creating-custom-workflows)

---

## Contributing Workflow

### 🐛 Bug Fixes
1. **Create an Issue** - Document the bug with a clear description and reproduction steps
2. **Create a Branch** - Use the naming convention `bugfix/issue-<number>-<description>`
3. **Fix the Bug** - Make your changes and test thoroughly
4. **Pull Request** - Create a PR to `main` and add `simon-jonas-buehrer` as a reviewer
5. **Review & Merge** - Address review comments and merge after approval

**Example:**
```bash
# Create a new branch for fixing issue #42
git checkout -b bugfix/issue-42-fix-cuda-memory-leak
git push -u origin bugfix/issue-42-fix-cuda-memory-leak
```

### 🚀 Feature Development
1. **Create a Branch** - Use `feature/<description>` naming convention
2. **Develop** - Implement your feature with tests
3. **Pull Request** - Create PR to `main` with detailed description
4. **Review** - Add `simon-jonas-buehrer` as reviewer
5. **Merge** - Merge after approval and CI passes

**Example:**
```bash
# Create a new feature branch
git checkout -b feature/add-attention-node
git push -u origin feature/add-attention-node
```

### 🔧 Hotfixes
1. **Create Branch** - Use `hotfix/<description>`
2. **Quick Fix** - Make minimal changes to fix critical issue
3. **Fast-track PR** - Expedited review process
4. **Deploy** - Merge and deploy immediately

---

## Branch Naming Conventions

Follow these naming conventions for all branches:

| Branch Type | Format | Example | Use Case |
|------------|--------|---------|----------|
| **Feature** | `feature/<description>` | `feature/add-attention-node` | New features |
| **Bugfix** | `bugfix/issue-<number>-<description>` | `bugfix/issue-42-fix-memory-leak` | Bug fixes with issue tracker |
| **Bugfix (no issue)** | `bugfix/<description>` | `bugfix/fix-typo-in-docs` | Minor bug fixes |
| **Hotfix** | `hotfix/<description>` | `hotfix/critical-cuda-crash` | Critical production fixes |
| **Docs** | `docs/<description>` | `docs/update-installation-guide` | Documentation updates |
| **Refactor** | `refactor/<description>` | `refactor/simplify-registry` | Code refactoring |
| **Test** | `test/<description>` | `test/add-unit-tests-for-nodes` | Adding or fixing tests |
| **Chore** | `chore/<description>` | `chore/update-dependencies` | Maintenance tasks |

**General Rules:**
- Use lowercase letters
- Use hyphens (`-`) to separate words
- Keep descriptions short but descriptive
- Reference issue numbers when applicable

---

## Version Management

DiffLUT uses **semantic versioning** (SemVer) with automated version management via `bump2version`.

### 📦 Semantic Versioning Format: `MAJOR.MINOR.PATCH`

| Component | When to Bump | Example |
|-----------|--------------|---------|
| **MAJOR** | Breaking changes that require user code modifications | `1.2.3` → `2.0.0` |
| **MINOR** | New features that are backward compatible | `1.2.3` → `1.3.0` |
| **PATCH** | Bug fixes and minor improvements | `1.2.3` → `1.2.4` |

### 🤖 Automatic Version Bumping

The version management workflow **automatically bumps the patch version** if you forget to do it manually. This ensures every merge increments the version.

**What happens:**
1. You create a PR without bumping the version
2. CI detects version hasn't changed from `main`
3. **Warning**: PR comment alerts you about auto-bump
4. When merged to `main`, version is auto-bumped (patch)
5. New version is committed and tagged

### 🔧 Manual Version Bumping (Recommended)

Before creating a PR, manually bump the version based on your changes:

```bash
# For bug fixes and minor improvements (default)
bump2version patch   # 1.2.3 → 1.2.4

# For new features (backward compatible)
bump2version minor   # 1.2.3 → 1.3.0

# For breaking changes
bump2version major   # 1.2.3 → 2.0.0
```

**What `bump2version` does:**
- Updates version in `setup.py`, `pyproject.toml`, and `difflut/__init__.py`
- Creates a git commit with message: `Bump version: X.Y.Z → X.Y.Z+1`
- Creates a git tag: `vX.Y.Z+1`
- Ensures version consistency across all files

### 📝 Version Files

Three files must always have matching versions:
- `setup.py` - `version='1.2.3'`
- `pyproject.toml` - `version = "1.2.3"`
- `difflut/__init__.py` - `__version__ = "1.2.3"`

The `.bumpversion.cfg` file ensures all three are updated together.

---

## GitHub Actions CI/CD

DiffLUT has two main workflows in `.github/workflows/`:

```
.github/
├── GITHUB_GUIDE.md
└── workflows/
    ├── tests.yml              # Runs tests on CPU and GPU
    └── version-management.yml # Auto-bumps version if needed
```

### 🧪 Tests Workflow (`tests.yml`)

**Triggers:** Every push and pull request to `main`

**What it does:**
1. **CPU Tests** - Runs on multiple Python versions (3.10, 3.11, 3.12) and PyTorch versions (2.4-2.9)
2. **GPU Tests** - Builds CUDA extensions and tests on multiple CUDA versions (12.4, 12.6, 12.8)
3. **Coverage** - Generates coverage reports and uploads to Codecov
4. **Fast Tests** - Runs quick tests first to fail fast
5. **Full Tests** - Runs complete test suite with coverage

**Test Matrix:**
- **CPU**: 6 combinations (3 Python × 2 PyTorch versions)
- **GPU**: 6 combinations (3 Python × 2 CUDA versions)

**When tests fail:**
- Check the GitHub Actions tab for detailed logs
- Tests are marked with `@pytest.mark.gpu` or `@pytest.mark.slow`
- Fix issues before merging

### 🔢 Version Management Workflow (`version-management.yml`)

**Triggers:** Every push and pull request to `main`

**What it does:**
1. **Version Check** - Compares your branch version with `main` branch version
2. **Auto-Bump** - If versions match, automatically bumps patch version
3. **Warning** - Posts comment on PR if auto-bump occurred
4. **Commit** - Creates commit with bumped version (only on push to `main`)
5. **Tag** - Creates git tag for the new version

**Behavior:**
- **Pull Request**: Warns you if version needs bumping
- **Push to Main**: Automatically commits and pushes version bump

**Example PR Comment:**
```
⚠️ Version Bump Warning

The version was not manually bumped. Auto-bumped from `1.2.3` → `1.2.4`.

If you want a minor or major version bump instead, run:
bump2version minor  # for feature additions
bump2version major  # for breaking changes
```

---

## Creating Custom Workflows

To add your own GitHub Actions workflow:

### 1. Create Workflow File
```bash
# Create a new workflow file
touch .github/workflows/my-workflow.yml
```

### 2. Basic Workflow Template
```yaml
name: My Custom Workflow

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  my-job:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -e .[dev]
    
    - name: Run my custom task
      run: |
        echo "Add your commands here"
```

### 3. Common Workflow Patterns

**Linting & Formatting:**
```yaml
- name: Run Black formatter check
  run: |
    pip install black
    black --check difflut/ tests/

- name: Run Flake8 linting
  run: |
    pip install flake8
    flake8 difflut/ tests/ --max-line-length=100
```

**Documentation Building:**
```yaml
- name: Build Sphinx docs
  run: |
    pip install sphinx sphinx-rtd-theme
    cd docs && make html
```

**Security Scanning:**
```yaml
- name: Run Bandit security scan
  run: |
    pip install bandit
    bandit -r difflut/
```

### 4. Useful GitHub Actions

- `actions/checkout@v4` - Check out repository
- `actions/setup-python@v4` - Set up Python environment
- `codecov/codecov-action@v3` - Upload coverage reports
- `actions/upload-artifact@v3` - Upload build artifacts
- `actions/cache@v3` - Cache dependencies

### 5. Best Practices

- **Fail Fast** - Run quick checks before slow ones
- **Matrix Testing** - Test across multiple versions
- **Caching** - Cache pip dependencies to speed up CI
- **Secrets** - Use GitHub Secrets for sensitive data
- **Branch Protection** - Require CI to pass before merging

---

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [bump2version Documentation](https://github.com/c4urself/bump2version)
- [Semantic Versioning Specification](https://semver.org/)
- [PyTest Documentation](https://docs.pytest.org/)

---

## 🤝 Getting Help

If you encounter issues:
1. Check existing GitHub Issues
2. Review the CI logs for error details
3. Ask in Pull Request comments
4. Contact maintainers: `simon-jonas-buehrer`

**Happy Contributing! 🚀**