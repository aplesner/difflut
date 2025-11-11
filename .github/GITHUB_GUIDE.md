# GitHub Guide

This guide explains the workflow for contributing to DiffLUT, including branching strategies, version management, and CI/CD pipelines.

## Table of Contents
- [Contributing Workflow](#contributing-workflow)
- [Branch Naming Conventions](#branch-naming-conventions)
- [Version Management](#version-management)
- [GitHub Actions CI/CD](#github-actions-cicd)
- [Creating Custom Workflows](#creating-custom-workflows)

---

## Contributing Workflow

### Bug Fixes
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

### Feature Development
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

### Hotfixes
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

### Semantic Versioning Format: `MAJOR.MINOR.PATCH`

| Component | When to Bump | Example |
|-----------|--------------|---------|
| **MAJOR** | Breaking changes that require user code modifications | `1.2.3` → `2.0.0` |
| **MINOR** | New features that are backward compatible | `1.2.3` → `1.3.0` |
| **PATCH** | Bug fixes and minor improvements | `1.2.3` → `1.2.4` |

### Version Bump Requirement

The version management workflow **requires you to bump the version** before merging to main. If you forget, the CI will fail and warn you.

**What happens:**
1. You create a PR without bumping the version
2. CI detects version hasn't changed from `main`
3. **CI fails** with a clear warning message
4. PR comment tells you exactly what to do
5. You bump the version and push again

### How to Bump Version (Required)

Before creating a PR, **always bump the version** based on your changes:

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

**After bumping, push with tags:**
```bash
git push --follow-tags
```

### Version Files

Three files must always have matching versions:
- `setup.py` - `version='1.2.3'`
- `pyproject.toml` - `version = "1.2.3"`
- `difflut/__init__.py` - `__version__ = "1.2.3"`

The `.bumpversion.cfg` file ensures all three are updated together.

---

## GitHub Actions CI/CD

DiffLUT has multiple workflows in `.github/workflows/`:

```
.github/
├── GITHUB_GUIDE.md
└── workflows/
    ├── tests.yml              # Runs tests on CPU and GPU
    ├── version-management.yml # Warns about version bumping
    └── format.yml             # Checks code formatting (Black & isort)
```

### Tests Workflow (`tests.yml`)

**Triggers:** Every push and pull request to `main`

**What it does:**
1. **CPU Tests** - Runs on multiple Python versions (3.10, 3.11, 3.12) and PyTorch versions (2.4-2.9)
2. **GPU Tests** - Builds CUDA extensions and validates GPU setup on multiple CUDA versions (12.4, 12.6, 12.8)
3. **Coverage** - Generates coverage reports and uploads to Codecov
4. **Test Markers** - Uses pytest markers (`slow`, `gpu`, `skip_ci`, `experimental`) to organize tests

**Test Matrix:**
- **CPU**: 6 combinations (3 Python × 2 PyTorch versions)
- **GPU**: 6 combinations (3 Python × 2 CUDA versions)

**When tests fail:**
- Check the GitHub Actions tab for detailed logs
- See [Testing Guide](../docs/DEVELOPER_GUIDE/tests.md) for test marker details
- Fix issues before merging

**For comprehensive testing information:**
- 📖 **Test Structure & Organization**: See [Testing Guide - Current Test Structure](../docs/DEVELOPER_GUIDE/tests.md#current-test-structure)
- 📊 **Test Markers**: See [Testing Guide - Test Markers](../docs/DEVELOPER_GUIDE/tests.md#test-markers)
- 🔄 **CI/CD Details**: See [Testing Guide - CI/CD Workflow](../docs/DEVELOPER_GUIDE/tests.md#cicd-workflow)
- ✍️ **Writing Tests**: See [Testing Guide - Writing Tests](../docs/DEVELOPER_GUIDE/tests.md#writing-tests-for-custom-components)
- ⚙️ **Test Configuration**: See `pyproject.toml` (pytest section) and `tests/conftest.py`

### Version Management Workflow (`version-management.yml`)

**Triggers:** Every push and pull request to `main`

**What it does:**
1. **Version Check** - Compares your branch version with `main` branch version
2. **Warn if Not Bumped** - CI warns if versions are the same (doesn't fail)
3. **Comment** - Posts clear comment on PR with instructions
4. **Reminder** - Reminds you to bump version before merging

**Behavior:**
- **Pull Request**: Warns and posts comment if version not bumped
- **Push to Main**: Warns if version wasn't bumped
- **Does NOT fail** - Only provides warnings to remind developers

**Example PR Comment:**
```
⚠️ Version Bump Required

The version has not been bumped from the main branch.

Please run one of the following commands before merging:
bump2version patch  # for bug fixes (1.1.3 → 1.1.4)
bump2version minor  # for new features (1.1.3 → 1.2.0)
bump2version major  # for breaking changes (1.1.3 → 2.0.0)

Then commit and push the changes:
git push --follow-tags
```

**Why this approach?**
- Simple and predictable
- Doesn't block PRs, just reminds
- No automated commits to worry about
- Forces developers to think about version semantics
- No permission issues with GitHub Actions

### Code Formatting Workflow (`format.yml`)

**Triggers:** Every push and pull request to `main`

**What it does:**
1. **Black Check** - Verifies code is formatted with Black (88 char line length)
2. **isort Check** - Verifies imports are properly sorted
3. **Warn Only** - Posts warnings but does NOT fail CI
4. **PR Comment** - Provides clear instructions on how to fix formatting

**Tools Used:**
- **Black** - Python code formatter (PEP 8 compliant)
- **isort** - Python import sorter (compatible with Black)

**How to fix formatting locally:**
```bash
# Install formatting tools
pip install black isort

# Format your code
black .
isort .

# Commit the changes
git add .
git commit -m "style: apply code formatting"
git push
```

**Example PR Comment:**
```
⚠️ Code Formatting Issues Detected

Some files need formatting adjustments:
- ❌ Black: Code is not properly formatted
- ❌ isort: Imports are not properly sorted

🔧 How to Fix
Run these commands locally to auto-format your code:
black .
isort .

Note: This is a warning only and will not block your PR from merging.
```

**Why this approach?**
- Maintains consistent code style
- Simplifies code reviews (focus on logic, not style)
- Doesn't block PRs for style issues
- Easy to fix with one command
- Works with most Python editors (VS Code, PyCharm, etc.)

---

## Code Formatting Best Practices

DiffLUT uses **Black** and **isort** for consistent code formatting across the codebase.

### Installation

```bash
# Install formatting tools
pip install black isort

# Or install with dev dependencies
pip install -e .[dev]
```

### Running Formatters

```bash
# Format all Python files
black .

# Sort all imports
isort .

# Do both in one go
black . && isort .
```

### Editor Integration

**VS Code:**
```json
// .vscode/settings.json
{
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

**PyCharm:**
1. Go to Settings → Tools → File Watchers
2. Add Black and isort watchers
3. Enable "Format on save"

### Configuration Files (Optional)

Create `pyproject.toml` in the root (if not exists):
```toml
[tool.black]
line-length = 88
target-version = ['py310', 'py311', 'py312']

[tool.isort]
profile = "black"
line_length = 88
```

### Pre-commit Hooks (Optional)

For automatic formatting before commits:

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
EOF

# Install the hooks
pre-commit install
```

Now formatting runs automatically before each commit!

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

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [bump2version Documentation](https://github.com/c4urself/bump2version)
- [Semantic Versioning Specification](https://semver.org/)
- [PyTest Documentation](https://docs.pytest.org/)

---

## Getting Help

If you encounter issues:
1. Check existing GitHub Issues
2. Review the CI logs for error details
3. Ask in Pull Request comments
4. Contact maintainers: `simon-jonas-buehrer`

**Happy Contributing!**