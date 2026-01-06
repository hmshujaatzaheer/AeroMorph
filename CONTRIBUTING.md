# Contributing to AeroMorph

Thank you for your interest in contributing to AeroMorph! This document provides guidelines for contributions.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/AeroMorph.git
cd AeroMorph
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Code Style

- Follow PEP 8 guidelines
- Use Black for formatting: `black aeromorph/`
- Use type hints where appropriate
- Write docstrings for all public functions

## Testing

Run tests before submitting:
```bash
pytest tests/ -v
```

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes with clear commit messages
3. Add tests for new functionality
4. Update documentation as needed
5. Submit a pull request with a clear description

## Research Contributions

For research-related contributions:
- Ensure mathematical formulations are clearly documented
- Provide references to relevant literature
- Include validation experiments where possible

## Questions?

Contact: shujabis@gmail.com
