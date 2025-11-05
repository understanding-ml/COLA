# Building COLA Documentation

This guide explains how to build the COLA documentation locally and deploy it to Read the Docs.

## Prerequisites

- Python 3.7+
- pip
- COLA source code

## Local Build

### 1. Install Documentation Dependencies

```bash
# From the COLA root directory
pip install -r docs/requirements-docs.txt

# Or install with extras
pip install -e ".[docs]"
```

### 2. Build HTML Documentation

```bash
cd docs
make html
```

The built documentation will be in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser.

### 3. Live Preview (Auto-reload on Changes)

For development, use auto-build mode:

```bash
cd docs
make livehtml
```

This will start a local server at `http://127.0.0.1:8000` that automatically rebuilds when you save changes.

### 4. Build Other Formats

```bash
# PDF (requires LaTeX)
make latexpdf

# ePub
make epub

# Clean build artifacts
make clean
```

## Windows Users

Use `make.bat` instead of `make`:

```bash
cd docs
make.bat html
```

## Sphinx Extensions Used

COLA documentation uses the following Sphinx extensions:

- **sphinx.ext.autodoc** - Auto-generate docs from docstrings
- **sphinx.ext.napoleon** - Support Google/NumPy docstring styles
- **sphinx.ext.viewcode** - Link to source code
- **sphinx.ext.intersphinx** - Link to other projects' docs
- **sphinx.ext.mathjax** - Render math equations
- **sphinx_copybutton** - Copy button for code blocks
- **myst_parser** - Markdown support

## Theme

We use the **Furo** theme for a modern, clean look. Configuration is in `docs/conf.py`.

## Read the Docs Deployment

### Automatic Deployment

COLA documentation is automatically built and deployed to Read the Docs when you push to GitHub.

Configuration: `.readthedocs.yaml`

### Manual Setup

1. Go to https://readthedocs.org/
2. Sign in with GitHub
3. Import the COLA repository
4. The build will start automatically

### Webhook

Read the Docs automatically creates a webhook in your GitHub repository to trigger builds on push.

## Documentation Structure

```
docs/
├── conf.py                 # Sphinx configuration
├── index.rst              # Main page
├── Makefile               # Build commands (Linux/Mac)
├── make.bat               # Build commands (Windows)
├── requirements-docs.txt  # Documentation dependencies
│
├── images/                # Images used in docs
│
├── tutorials/             # Tutorial guides
│   ├── README.md
│   └── 01_basic_tutorial.md
│
├── user_guide/            # User guides (to be created)
│   ├── data_interface.rst
│   ├── models.rst
│   └── ...
│
└── api/                   # API reference (to be created)
    ├── cola.rst
    ├── data.rst
    └── ...
```

## Writing Documentation

### Adding a New Page

1. Create a new `.rst` or `.md` file in the appropriate directory
2. Add it to the `toctree` in `index.rst` or the parent page

Example:

```rst
.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/new_page
```

### Markdown vs reStructuredText

- **Markdown (.md)**: Easier to write, good for tutorials and guides
- **reStructuredText (.rst)**: More powerful, better for API docs

Both are supported via `myst_parser`.

### Docstring Style

Use Google or NumPy style docstrings:

```python
def example_function(param1, param2):
    """Brief description.

    Detailed description here.

    Args:
        param1 (str): Description of param1
        param2 (int): Description of param2

    Returns:
        bool: Description of return value

    Example:
        >>> example_function("test", 42)
        True
    """
    return True
```

### Cross-References

```rst
:ref:`section-label`
:doc:`other_page`
:class:`xai_cola.COLA`
:func:`xai_cola.data.COLAData.add_counterfactuals`
:mod:`xai_cola.ce_generator`
```

## Troubleshooting

### "WARNING: document isn't included in any toctree"

Add the document to a `toctree` directive in `index.rst` or a parent page.

### "ImportError" during build

Ensure all dependencies are installed:

```bash
pip install -e ".[docs]"
```

### Images not showing

Check that:
- Image paths are correct relative to the document
- Images are in the `docs/images/` directory
- Image paths in `html_static_path` are correct in `conf.py`

### Math not rendering

Ensure `sphinx.ext.mathjax` is in the extensions list in `conf.py`.

## Best Practices

1. **Build often**: Test your changes frequently with `make html`
2. **Use live preview**: `make livehtml` for faster iteration
3. **Check warnings**: Fix all Sphinx warnings before committing
4. **Test cross-references**: Ensure all links work
5. **Preview multiple formats**: Test PDF/ePub if users will need them
6. **Mobile-friendly**: Furo theme is responsive, test on different sizes

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Furo Theme](https://pradyunsg.me/furo/)
- [Read the Docs Guide](https://docs.readthedocs.io/)
- [MyST Parser (Markdown)](https://myst-parser.readthedocs.io/)
- [Docstring Examples](https://sphinxcontrib-napoleon.readthedocs.io/)

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing to documentation.
