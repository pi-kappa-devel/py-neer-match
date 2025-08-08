# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= python -m sphinx.cmd.build
SOURCEDIR     = source
BUILDDIR      = build
PACKAGEDIR    = $(SOURCEDIR)/../..
EXAMPLEDIR    = $(SOURCEDIR)/_static/examples

EXAMPLE_MD := $(patsubst $(EXAMPLEDIR)/%.qmd, $(SOURCEDIR)/%.md, $(wildcard $(EXAMPLEDIR)/*.qmd))

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

$(SOURCEDIR)/LICENSE.md: $(PACKAGEDIR)/LICENSE.txt
	cp "$(PACKAGEDIR)/LICENSE.txt" "$(SOURCEDIR)/LICENSE.md"

$(SOURCEDIR)/README.md: $(SOURCEDIR)/_static/README.qmd
	QUARTO_PYTHON=$(QUARTO_PYTHON) quarto render "$<" -t gfm --output-dir ../
	cp "$@" ../README.md
	sed -i 's/\[MIT license\](LICENSE.txt)/<a href="LICENSE.html">MIT license<\/a>/g' "$@"
	sed -i 's/docs\/source\///g' "$@"

$(SOURCEDIR)/%.md: $(EXAMPLEDIR)/%.qmd
	QUARTO_PYTHON=$(QUARTO_PYTHON) quarto render "$<" -t gfm
	mv "$(EXAMPLEDIR)/$*.md" "$(SOURCEDIR)/$*.md"
	sed -i "s/$*_files/_static\/examples\/$*_files/g" "$(SOURCEDIR)/$*.md"

examples: $(EXAMPLE_MD)

html: $(SOURCEDIR)/LICENSE.md $(SOURCEDIR)/README.md examples
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
