# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Temporal Mapper'
copyright = '2024, Kaleb D Ruscitti'
author = 'Kaleb D Ruscitti'
release = '1.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
    'nbsphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary'
]

templates_path = ['_templates']
exclude_patterns = ['**.ipynb_checkpoints']

# -- Autosummary --
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_static_path = ['_static']


import plotly.io as pio
pio.renderers.default = 'sphinx_gallery'
