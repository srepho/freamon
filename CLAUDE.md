# Freamon Development Guidelines

## Build & Test Commands
```bash
# Install in development mode
pip install -e .

# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run a specific test file
pytest tests/test_filename.py

# Run a specific test
pytest tests/test_filename.py::test_function_name

# Run with coverage
pytest --cov=freamon
```

## Code Style Guidelines
- **Python Versions:** Target 3.10, 3.11, 3.12
- **Type Hints:** Required for all functions and methods
- **Docstrings:** Comprehensive docstrings for all public elements (with examples)
- **Naming:** Use snake_case for functions/variables, PascalCase for classes
- **Imports:** Group standard library, third-party, and local imports
- **Error Handling:** Use descriptive exceptions with clear error messages
- **Validation:** Input validation at function boundaries
- **Method Chaining:** Support fluent interfaces where appropriate
- **Performance:** Use vectorized operations over loops, optimize memory use
- **Testing:** All functionality requires test coverage

## Development Workflow
- Create feature branches from main
- Submit PRs for code review before merging
- Follow semantic versioning
- Add doctests in docstrings for examples