# Contributing to Car Features and MSRP Prediction

First off, thank you for considering contributing to this project! Your help is essential for keeping it great.

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

- **Use the GitHub issue tracker** — Check if the bug has already been reported by searching on GitHub under [Issues](https://github.com/yourusername/car-features-msrp/issues).
- **Use the bug report template** — When you create a new issue, you'll see a template that guides you through the essential components of a bug report.
- **Include detailed steps to reproduce** — The more detail you provide, the easier it is to reproduce and the faster it can be fixed.

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion, including completely new features and minor improvements to existing functionality.

- **Use the GitHub issue tracker** — Check if the enhancement has already been suggested by searching on GitHub under [Issues](https://github.com/yourusername/car-features-msrp/issues).
- **Use the feature request template** — When you create a new issue, select the feature request template which will guide you through the essential information we need.
- **Describe the enhancement in detail** — Explain why this enhancement would be useful to most users.

### Pull Requests

- **Fill out the pull request template** — It's designed to prompt you for all the information we need to properly review your PR.
- **Make small, focused commits** — Break large changes into smaller, logical commits for easier review.
- **Follow the coding style** — Match the existing code style and conventions.
- **Add tests for new functionality** — This ensures the new code works as expected and won't break in the future.
- **Update documentation** — Make sure any new features or changes are reflected in the documentation.

## Development Process

### Setting Up a Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/car-features-msrp.git
   cd car-features-msrp
   ```

3. Create a branch for local development:
   ```bash
   git checkout -b name-of-your-feature
   ```

4. Set up your environment:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

### Making Changes

1. Make your changes locally
2. Run tests to ensure your changes don't break existing functionality
3. Commit your changes:
   ```bash
   git add .
   git commit -m "Your detailed description of your changes"
   ```

4. Push your branch to GitHub:
   ```bash
   git push origin name-of-your-feature
   ```

5. Submit a pull request through the GitHub website

## Style Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use docstrings for all functions, classes, and modules
- Include type hints where appropriate
- Keep lines to a maximum of 100 characters
- Use meaningful variable and function names

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests after the first line

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.
