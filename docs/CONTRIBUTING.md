# Contributing to Darija Transliteration

Thank you for your interest in contributing! This project aims to improve Moroccan Darija NLP tools.

## How to Contribute

### 1. Report Bugs

Found a bug? Please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, GPU/CPU)

### 2. Suggest Features

Have an idea? Open an issue with:
- Feature description
- Use case / motivation
- Proposed implementation (if you have ideas)

### 3. Submit Code

#### Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Test thoroughly
5. Commit with clear messages: `git commit -m "Add: feature description"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Open a Pull Request

#### Code Style

- Follow PEP 8 for Python code
- Add comments for complex logic
- Include docstrings for functions
- Keep functions focused and modular

#### Example:

```python
def process_text(text):
    """
    Process and clean input text.
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())
```

### 4. Add Data

High-quality Darija data is valuable! You can contribute by:

- Adding new sentence pairs to datasets
- Validating existing translations
- Providing domain-specific examples (medical, legal, technical, etc.)

**Data Format:**
```csv
latin_text,arabic_text
salam cv,سلام صافي
```

### 5. Improve Documentation

- Fix typos
- Add examples
- Clarify confusing sections
- Translate documentation to other languages

## Areas Needing Help

### High Priority
- [ ] More diverse training data (different regions, age groups, topics)
- [ ] Evaluation metrics implementation
- [ ] Web interface for easy access
- [ ] Model quantization for faster inference

### Medium Priority
- [ ] Support for other Maghrebi dialects
- [ ] Sentiment analysis integration
- [ ] Named entity recognition
- [ ] Better handling of code-switching

### Low Priority
- [ ] Mobile app
- [ ] Browser extension
- [ ] API service
- [ ] Batch processing tools

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/darija-transliteration.git
cd darija-transliteration

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/darija-transliteration.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8
```

## Testing

Before submitting a PR:

```bash
# Run basic tests
python test_model.py

# Check code style
flake8 *.py

# Format code
black *.py
```

## Pull Request Guidelines

- **One feature per PR**: Keep changes focused
- **Clear description**: Explain what and why
- **Link issues**: Reference related issues
- **Update docs**: If you change functionality, update README
- **Test**: Ensure your changes don't break existing functionality

## Code Review Process

1. Maintainer reviews your PR
2. Feedback/changes requested (if needed)
3. You make updates
4. PR is approved and merged

## Community Guidelines

- Be respectful and inclusive
- Help others learn
- Give constructive feedback
- Celebrate contributions

## Questions?

- Open a discussion on GitHub
- Check existing issues and PRs
- Read the documentation

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for helping improve Darija NLP! 🙏
