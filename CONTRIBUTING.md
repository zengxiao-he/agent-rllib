# Contributing to Agent-RLlib

Thank you for your interest in contributing to Agent-RLlib! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/agent-rllib.git
   cd agent-rllib
   ```

2. **Set up development environment**
   ```bash
   make install-dev
   ```

3. **Run tests to ensure everything works**
   ```bash
   make test
   ```

## 🏗️ Development Workflow

### Setting Up Your Environment

1. **Python Version**: Use Python 3.8 or higher
2. **Virtual Environment**: Recommended to use `venv` or `conda`
3. **Dependencies**: Install with `make install-dev`
4. **Pre-commit Hooks**: Automatically installed with dev dependencies

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Follow the existing code style
   - Add tests for new functionality

3. **Test your changes**
   ```bash
   make test
   make lint
   make type-check
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Code Style Guidelines

### Python Code Style

- **Formatting**: We use `black` for code formatting
- **Import Sorting**: We use `isort` for import organization
- **Linting**: We use `flake8` for style checking
- **Type Hints**: Use type hints for all public functions and methods

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

**Examples:**
```
feat: add multi-agent negotiation environment
fix: resolve memory leak in PPO training loop
docs: update API documentation for tool registry
test: add comprehensive tests for SupportBot environment
```

## 🧪 Testing Guidelines

### Test Structure

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Environment Tests**: Test RL environments thoroughly
- **End-to-End Tests**: Test complete workflows

### Writing Tests

1. **Test File Naming**: `test_*.py`
2. **Test Function Naming**: `test_<functionality_being_tested>`
3. **Use Fixtures**: For setup and teardown
4. **Mock External Dependencies**: Use `unittest.mock` or `pytest-mock`

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest src/tests/test_env.py -v

# Run with coverage
make test

# Run tests in parallel
pytest -n auto
```

## 🏗️ Architecture Guidelines

### Adding New Environments

1. **Inherit from `gym.Env`**
2. **Implement required methods**: `reset()`, `step()`, `render()`, `close()`
3. **Define observation and action spaces**
4. **Add comprehensive docstrings**
5. **Include example usage**

### Adding New Agents

1. **Follow the existing agent interface**
2. **Implement `get_action()` and `update()` methods**
3. **Support both training and inference modes**
4. **Add configuration options**
5. **Include model checkpointing**

### Adding New Tools

1. **Inherit from `BaseTool`**
2. **Implement `execute()` method**
3. **Add parameter validation**
4. **Handle errors gracefully**
5. **Register with `ToolRegistry`**

## 📚 Documentation

### Code Documentation

- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Include type hints for all parameters and return values
- **Examples**: Include usage examples in docstrings

### API Documentation

- **Sphinx**: We use Sphinx for generating documentation
- **Auto-generation**: Documentation is auto-generated from docstrings
- **Build Docs**: Use `make docs` to build documentation locally

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment Information**
   - Python version
   - Package versions (`pip list`)
   - Operating system

2. **Reproduction Steps**
   - Minimal code example
   - Expected behavior
   - Actual behavior

3. **Error Messages**
   - Full stack traces
   - Log files if applicable

## 💡 Feature Requests

When requesting features:

1. **Use Case**: Describe the problem you're trying to solve
2. **Proposed Solution**: Suggest how it might be implemented
3. **Alternatives**: Consider alternative approaches
4. **Impact**: Estimate the benefit to the community

## 🔍 Code Review Process

### For Contributors

1. **Self Review**: Review your own code before submitting
2. **Test Coverage**: Ensure adequate test coverage
3. **Documentation**: Update documentation as needed
4. **Breaking Changes**: Clearly mark and document breaking changes

### For Reviewers

1. **Be Constructive**: Provide helpful feedback
2. **Ask Questions**: Clarify unclear code or decisions
3. **Suggest Improvements**: Offer specific suggestions
4. **Approve When Ready**: Don't delay unnecessarily

## 🏆 Recognition

Contributors will be recognized in:

- **README.md**: Contributors section
- **CHANGELOG.md**: Release notes
- **GitHub**: Contributor graphs and statistics

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Discord**: [Join our Discord server](https://discord.gg/agent-rllib)
- **Email**: maintainer@agent-rllib.com

## 📜 License

By contributing to Agent-RLlib, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You

Thank you for contributing to Agent-RLlib! Your contributions help make this project better for everyone.

---

*This document is based on best practices from the open-source community and is continuously updated based on project needs and contributor feedback.*
