import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from src import neer_match


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Neer Match"
copyright = "2024, Pantelis Karapanagiotis, Marius Liebald"
author = "Pantelis Karapanagiotis, Marius Liebald"
release = neer_match.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


extlinks = {
    "ltn-tut-2c": (
        "https://nbviewer.org/github/logictensornetworks/logictensornetworks/blob/"
        "master/tutorials/2b-operators_and_gradients.ipynb%s",
        None,
    )
}


def _skip(app, what, name, obj, would_skip, options):
    if name in ["__getitem__", "__init__", "__iter__", "__len__", "__str__"]:
        return False
    return would_skip


def setup(app):
    """Set up the Sphinx application."""
    app.connect("autodoc-skip-member", _skip)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

html_logo = "_static/img/logo.png"
html_theme_options = {
    "github_user": "pi-kappa-devel",
    "github_repo": "neer-match",
    "github_banner": True,
    "github_button": True,
    "github_type": "star",
    "github_count": True,
    "show_powered_by": True,
    "show_related": True,
    "note_bg": "#ffffff",
    "note_border": "#c8c8c8",
    "fixed_sidebar": True,
    "extra_nav_links": {
        "📦 PyPi Package": "https://pypi.org/project/neer-match/",
        "📦 R Package": "https://github.com/pi-kappa-devel/r-neer-match",
        "📖 R Docs": "https://r-neer-match.pikappa.eu",
    },
}


html_favicon = "_static/img/favicon/favicon.ico"
