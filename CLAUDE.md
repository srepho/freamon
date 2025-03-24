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

## Key Features

### Deduplication Tracking
- Use `IndexTracker` to maintain bidirectional mappings between original and current indices
- Update mappings after any operation that changes dataset size
- Map results back to original dataset with `create_full_result_df()`
- Consider the `DeduplicationTracker` for advanced visualization and tracking

### Markdown Reports
- Generate reports using `generate_markdown_report()` or `EDAAnalyzer.generate_report(format="markdown")`
- Convert to HTML with `convert_to_html=True` parameter
- Use high-quality formatting and structured sections for readability
- Keep file sizes small for version control compatibility