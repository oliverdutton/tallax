# Agent Guidelines for tallax Development

## Development Environment Setup

When working on tallax, ensure you have the proper development dependencies installed:

### Install Development Version

According to `pyproject.toml`, tallax has optional dev dependencies that include pytest:

```bash
pip install -e ".[dev]"
```

Or if you need TPU support:

```bash
pip install -e ".[tpu,dev]"
```

The dev dependencies include:
- `pytest>=7.0.0`

### Common Mistakes to Avoid

1. **Do not mock pytest imports**: Tests should assume pytest is available in the development environment. Remove any try/except blocks that mock pytest functionality.

2. **Install dev dependencies**: Before running tests, always install the `[dev]` optional dependencies as specified in `pyproject.toml`.

3. **Check README**: The README.md contains installation instructions. For development, use `pip install -e ".[dev]"` instead of just `pip install .`.

## Testing Guidelines

- All tests should use pytest directly
- Parameterized tests should use `@pytest.mark.parametrize`
- Tests should be runnable with `pytest tests/`
- Individual tests can be run directly with Python for debugging, but should still import pytest properly
