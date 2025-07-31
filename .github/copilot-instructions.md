# Copilot Instructions for Orbital Development

This document contains instructions for GitHub Copilot when working on the Orbital project.

## Python Environment and Testing

### Running Tests
- Python tests should be run with `uv run pytest`
- Valid pytest options can be added as needed (e.g., `uv run pytest -v`, `uv run pytest tests/specific_test.py`)
- Use `uv run pytest` instead of plain `pytest` to ensure the correct environment

### Python Commands
- All commands that need to run within a valid Python environment should be performed with `uv run COMMAND`
- Examples:
  - `uv run python script.py`
  - `uv run mypy src/`
  - `uv run ruff format .`
  - `uv run ruff check .`

## Quality Assurance Workflow

### Standard Testing
1. Run the standard test suite: `uv run pytest`
2. Ensure all tests pass before proceeding

### Final Validation
After standard tests have passed and before considering work complete:

1. **Run example validation**: Execute `uv run examples/test_examples.sh` to check for bugs in example scripts
   - **Important**: Always run with `uv run` prefix, do NOT modify the `test_examples.sh` script itself
   - The script internally uses `python` which will resolve to the correct environment when run via `uv run`
2. **Run pre-commit checks**: Execute `pre-commit run -a` and ensure it passes
3. **Format tests directory**: Execute `uv run ruff format tests/` (this won't be performed by pre-commit)

### Work Completion Checklist
- [ ] Standard tests pass (`uv run pytest`)
- [ ] Example validation passes (`uv run examples/test_examples.sh`)
- [ ] Pre-commit checks pass (`pre-commit run -a`)
- [ ] Tests directory formatted (`uv run ruff format tests/`)

Only when all four items are complete should work be considered fully done.

## Development Best Practices

- Always use `uv run` prefix for Python-related commands to maintain environment consistency
- Run the full validation workflow before submitting pull requests
- If any step in the quality assurance workflow fails, address the issues before proceeding

## Code Quality and Documentation

### Docstring Style
- Use Sphinx-style docstrings for documenting Python function arguments
- Function parameters should be documented using `:param NAME: description` format
- Document returned values in the docstring itself, do not use `:return:` directive.
- Do not document exceptions, as the maintenance cost of keeping them up-to-date is high.
- Example:
  ```python
  def example_function(name: str, age: int) -> str:
      """Example function with Sphinx-style docstring.

      Returns a formatted string with the person's information.
      
      :param name: The person's name
      :param age: The person's age in years
      """
      if age < 0:
          raise ValueError("Age cannot be negative")
      return f"{name} is {age} years old"
  ```

### Meaningful Comments
- Comments should explain **why** something is done, not **what** is being done
- The code itself should be readable and self-explanatory through good naming and structure
- Only add comments when they provide meaningful context about:
  - Business logic or domain-specific requirements
  - Non-obvious design decisions or trade-offs
  - Limitations or special cases that aren't apparent from the code
  - Purpose of the example or specific behavior being demonstrated
  - **Machine learning algorithm specifics**: When dealing with ML algorithms, explain the mathematical or algorithmic requirements that may not be obvious to someone unfamiliar with the specific algorithm or ML in general
- Avoid comments that simply restate what the code does (e.g., "# Create a list" for `items = []`)
- Example good comments:
  - `# Binary classification: expensive vs affordable houses`
  - `# SQL column names cannot start with digits`
  - `# Binary classification should produce exactly 2 probability columns`
  - `# ONNX treats binary classification as a special case with only one weight`
  - `# Tree ensemble requires post-transform before threshold check`
- Example bad comments:
  - `# Split target from features` (obvious from variable names)
  - `# Fill missing values` (obvious from function call)
