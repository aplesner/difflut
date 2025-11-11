# Contributing to DiffLUT

Thank you for your interest in contributing to DiffLUT! This guide explains the development process, testing requirements, and how to submit pull requests.

## Getting Started

### 1. Fork and Clone

```bash
# Fork repository on GitLab
git clone https://gitlab.ethz.ch/your-username/difflut.git
cd difflut

# Add upstream remote
git remote add upstream https://gitlab.ethz.ch/disco-students/hs25/difflut.git
```

### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install additional dev tools if you need
pip install wandb
```
Or use the maintained container for reproducible HPC environments. See detailed instructions in `apptainer/README.md`.


### 3. Create Branch

```bash
# Update local repository
git fetch upstream
git checkout upstream/main

# Create feature branch
git checkout -b feature/my-new-feature
```
See `.github/GITHUB_GUIDE.md` for detailed describtion on naming conventions, versioning and other github specifcs.

## Code Standards

- **Type hints**: Use PEP 484-compatible `Optional[T]` (not `T | None`) for Python < 3.10 compatibility
- **Defaults**: Define as module-level constants and warn when applied (use `warn_default_value()`)
- **Configuration**: Use type-safe parameter passing
- **Code style**: Format with `black`, sort with `isort`
- **Docstrings**: Use NumPy style with Parameters, Returns, Raises, Examples sections
- **Commits**: Write clear, descriptive messages with bullet points

## Testing
See [Testing Documentation](tests.md)

## Community Guidelines

- Be respectful and constructive
- Assume good intent
- Help others learn
- Share knowledge
- Give credit for ideas

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Treat all contributors with respect.

## License

By contributing to DiffLUT, you agree that your contributions will be licensed under the MIT License.

## Next Steps

- Read [Creating Components](creating_components.md) to implement new features
- Check [Packaging Guide](packaging.md) for distribution info
- Review existing code in `difflut/` directory

Thank you for contributing!
