# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	cp "$(SOURCEDIR)/../../LICENSE.txt" "$(SOURCEDIR)/LICENSE.md"
	quarto render "$(SOURCEDIR)/_static/README.qmd" -t gfm --output-dir ../
	cp "$(SOURCEDIR)/README.md" ../README.md
	sed -i 's/\[MIT license\](LICENSE.txt)/<a href="LICENSE.html">MIT license<\/a>/g' "$(SOURCEDIR)/README.md"
	sed -i 's/docs\/source\///g' "$(SOURCEDIR)/README.md"
	quarto render "$(SOURCEDIR)/_static/examples/dl-matching.qmd" -t gfm
	mv "$(SOURCEDIR)/_static/examples/dl-matching.md" "$(SOURCEDIR)/dl-matching.md"
	sed -i 's/dl-matching_files/_static\/examples\/dl-matching_files/g' "$(SOURCEDIR)/dl-matching.md"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
