# Configuration file for the Sphinx documentation builder.
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'COLA'
copyright = '2024, Lei You, Lin Zhu'
author = 'Lei You, Lin Zhu'

# The full version, including alpha/beta/rc tags
with open('../VERSION', 'r') as f:
    release = f.read().strip()
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings
extensions = [
    'sphinx.ext.autodoc',        # Auto-generate docs from docstrings
    'sphinx.ext.napoleon',       # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',       # Add links to highlighted source code
    'sphinx.ext.intersphinx',    # Link to other project's documentation
    'sphinx.ext.mathjax',        # Render math via JavaScript
    'sphinx.ext.githubpages',    # Create .nojekyll file for GitHub Pages
    'sphinx_copybutton',         # Add copy button to code blocks
    'myst_parser',               # Support for Markdown files
]

# Add any paths that contain templates here, relative to this directory
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages
html_theme = 'furo'  # Modern, clean theme

# Theme options
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2962FF",
        "color-brand-content": "#2962FF",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# Add any paths that contain custom static files (such as style sheets)
html_static_path = ['_static']

# The name of an image file (relative to this directory) to place at the top
# of the sidebar
html_logo = 'images/problem.png'

# The name of an image file (within the static path) to use as favicon
# html_favicon = '_static/favicon.ico'

# If true, links to the reST sources are added to the pages
html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer
html_show_copyright = True

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# -- Options for Napoleon (Google/NumPy docstring style) --------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for MyST (Markdown) --------------------------------------------

myst_enable_extensions = [
    "colon_fence",      # ::: fence for directives
    "deflist",          # Definition lists
    "dollarmath",       # $...$ for math
    "fieldlist",        # Field lists
    "html_admonition",  # HTML-style admonitions
    "html_image",       # HTML images
    "linkify",          # Auto-detect URLs
    "replacements",     # Text replacements
    "smartquotes",      # Smart quotes
    "strikethrough",    # ~~strikethrough~~
    "substitution",     # Variable substitution
    "tasklist",         # Task lists
]

myst_heading_anchors = 3  # Auto-generate anchors for headings

# -- Options for copybutton --------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
