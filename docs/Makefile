# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
PACKAGEDIR    = "$(SOURCEDIR)/../.."
EXAMPLEDIR    = "$(SOURCEDIR)/_static/examples"

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	cp "$(PACKAGEDIR)/LICENSE.txt" "$(SOURCEDIR)/LICENSE.md"
	QUARTO_PYTHON=python quarto render "$(SOURCEDIR)/_static/README.qmd" -t gfm --output-dir ../
	cp "$(SOURCEDIR)/README.md" ../README.md
	sed -i 's/\[MIT license\](LICENSE.txt)/<a href="LICENSE.html">MIT license<\/a>/g' \
		"$(SOURCEDIR)/README.md"
	sed -i 's/docs\/source\///g' "$(SOURCEDIR)/README.md"
	for path in $(EXAMPLEDIR)/*.qmd; do \
		file=$$(basename "$$path"); \
		QUARTO_PYTHON=python quarto render "$(EXAMPLEDIR)/$${file}" -t gfm; \
		mv "$(EXAMPLEDIR)/$${file%.qmd}.md" "$(SOURCEDIR)/$${file%.qmd}.md"; \
		sed -i "s/$${file%.qmd}_files/_static\/examples\/$${file%.qmd}_files/g" \
			"$(SOURCEDIR)/$${file%.qmd}.md"; \
	done
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
