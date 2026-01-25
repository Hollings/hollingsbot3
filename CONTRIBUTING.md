# Contributing to hollingsbot3

## Development Setup

### Prerequisites
- Python 3.10+
- Git
- System dependencies: `tesseract-ocr`, `libcairo2` (for image processing)

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/Hollings/hollingsbot3.git
   cd hollingsbot3
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   make install-dev
   # Or manually:
   pip install -r requirements.txt
   pip install ruff pytest pytest-asyncio pytest-cov pre-commit mypy bandit
   ```

4. Set up pre-commit hooks:
   ```bash
   make pre-commit
   # Or manually:
   pre-commit install
   ```

5. Copy the example environment file and configure:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

## Development Workflow

### Available Commands

Run `make help` to see all available commands:

| Command | Description |
|---------|-------------|
| `make lint` | Run linter (Ruff) |
| `make format` | Auto-format code |
| `make test` | Run tests |
| `make test-cov` | Run tests with coverage |
| `make typecheck` | Run type checker (mypy) |
| `make security` | Run security scan (bandit) |
| `make check` | Run all checks |
| `make clean` | Clean up generated files |

### Code Style

This project uses:
- **Ruff** for linting and formatting (replaces black, isort, flake8)
- **120 character line length**
- **Double quotes** for strings

The pre-commit hooks will automatically format your code on commit.

### Running Tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run specific test file
PYTHONPATH=src pytest tests/test_url_metadata.py -v
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. They will:
- Format your code with Ruff
- Check for linting errors
- Check for trailing whitespace
- Validate YAML/JSON/TOML files
- Scan for security issues

To run hooks manually on all files:
```bash
make pre-commit-all
```

### Pull Request Guidelines

1. Create a feature branch from `main`
2. Make your changes
3. Ensure all checks pass: `make check`
4. Write/update tests if applicable
5. Submit a pull request

## Project Structure

```
hollingsbot3/
├── src/hollingsbot/       # Main bot code
│   ├── cogs/              # Discord cogs (commands/features)
│   ├── text_generators/   # AI text generation (Anthropic, OpenAI, etc.)
│   ├── image_generators/  # Image generation (Replicate, etc.)
│   └── tools/             # Bot tools and utilities
├── tests/                 # Test files
├── scripts/               # Utility scripts
├── config/                # Configuration files
└── docker/                # Docker-related files
```

## Questions?

Open an issue on GitHub if you have questions or run into problems.
