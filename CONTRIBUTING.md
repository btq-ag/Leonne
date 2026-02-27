# Contributing to Leonne

Thank you for your interest in contributing to Leonne. This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a feature branch from `main`

```bash
git checkout -b feature/your-feature-name
```

4. Install development dependencies

```bash
pip install -e ".[dev,all]"
```

## Development Workflow

### Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=. --cov-report=term-missing
```

### Code Style

- Use **camelCase** for variable and function names
- Use **PascalCase** for class names
- Include docstrings for all public functions and classes
- Keep lines under 100 characters where practical

### Commit Messages

Write clear, concise commit messages. Use the imperative mood:

- "Add quantum trust partitioning tests"
- "Fix edge assignment for degenerate sequences"
- "Update persistent homology computation"

## Project Structure

| Directory | Contents |
|-----------|----------|
| `Classical Algorithms/` | Classical trust, consensus, and graph algorithms |
| `Quantum Algorithms/` | Quantum-enhanced counterparts |
| `Documentation/` | Mathematical background and theory |
| `Extension/` | Continuous-time cobordism extensions |
| `Visualizer/` | Hub interfaces and CLI tools |
| `tests/` | Test suite |

## Submitting Changes

1. Ensure all tests pass
2. Push your branch and open a pull request against `main`
3. Describe the change and link any relevant issues

## Reporting Issues

Open an issue on GitHub with:

- A clear description of the problem
- Steps to reproduce (if applicable)
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
