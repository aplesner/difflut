## Description
<!-- Please include a summary of the changes and related context. -->

## Type of Change
<!-- Please delete options that are not relevant. -->

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to change)
- [ ] 📚 Documentation update
- [ ] 🔧 Configuration/setup change
- [ ] ♻️ Code refactoring (no functional changes)

## Related Issues
<!-- Link to any related issues: Fixes #123, Relates to #456 -->

Fixes #
Relates to #

## How Has This Been Tested?
<!-- Please describe the tests that you ran to verify your changes. -->

- [ ] Tested locally with CPU build
- [ ] Tested locally with GPU build (if applicable)
- [ ] Added new tests
- [ ] All tests passing

## Version Bump

If this PR should result in a release, please bump the version using bump2version:

```bash
cd difflut

# Install bump2version
pip install bump2version

# Choose one based on the type of change:
bump2version patch    # Bug fixes: 1.1.2 → 1.1.3
bump2version minor    # New features: 1.1.2 → 1.2.0
bump2version major    # Breaking changes: 1.1.2 → 2.0.0

# This will update:
# - setup.py
# - pyproject.toml
# - difflut/__init__.py
# And create a git commit
```

Then commit and push:
```bash
git push origin your-branch-name
```

**Note:** The version bump is **optional** for documentation, refactoring, or minor fixes that don't warrant a release. Leave the version unchanged in those cases.

## Checklist

- [ ] My code follows the project's style guidelines (black, flake8, isort)
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests passed with my changes
- [ ] Any dependent changes have been merged and published

## Additional Context
<!-- Add any other context about the pull request here. -->

---

**CI/CD Checks:**
- ✅ Tests will run automatically on all Python versions (3.8-3.12)
- ✅ Code style will be checked (black, flake8, isort)
- ✅ Documentation will be built and verified
- ✅ GPU compatibility will be validated (if applicable)

See [CI/CD Setup Guide](.github/CICD_SETUP.md) for details.
