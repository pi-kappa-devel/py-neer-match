# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

from src import neer_match

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Neer Match'
copyright = '2024, Pantelis Karapanagiotis, Marius Liebald'
author = 'Pantelis Karapanagiotis, Marius Liebald'
release = neer_match.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

html_logo = "_static/img/logo.png"
html_theme_options = {
    'logo_only': True,
    'github_user': 'pi-kappa-devel',
    'github_repo': 'neer-match',
    'github_banner': True,
    'github_button': True,
    'github_type': 'star',
    'github_count': True,
    'github_url': 'https://github.com/pi-kappa-devel/neer-match',
    'show_powered_by': True,
    'show_related': True,
    'note_bg': '#ffffff',
    'note_border': '#c8c8c8',
    'fixed_sidebar': True,
    'extra_nav_links': {
        '📚': 'https://neer-match.pikappa.eu/',
        '📦': 'https://pypi.org/project/neer-match/',
    }
}


html_favicon = "_static/img/favicon/favicon.ico"

