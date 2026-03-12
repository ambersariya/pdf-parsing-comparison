# ---------------------------------------------------------------------------
# PDF Extraction Bake-off — Makefile
# ---------------------------------------------------------------------------
# Common targets:
#   make install          Install all dependencies via uv
#   make fixtures         Generate robustness test PDFs in docs/
#   make run PDF=my.pdf   Run all parsers against a single PDF
#   make run-all          Run all parsers against every PDF in docs/
#   make run-libs         Run a subset of libraries (LIBS="pdfplumber pymupdf")
#   make clean            Remove all generated results
#   make clean-results    Remove results/ only
#   make help             Print this help

PYTHON     ?= .venv/bin/python
UV         ?= uv
RESULTS    ?= results
DOCS       ?= docs
TIMEOUT    ?= 300
WORKERS    ?=
LIBS       ?=
PDF        ?=

# Build optional CLI flags from variables
_WORKERS   := $(if $(WORKERS),--workers $(WORKERS),)
_LIBS      := $(if $(LIBS),--libraries $(LIBS),)
_TIMEOUT   := --timeout $(TIMEOUT)

.PHONY: help install fixtures run run-all run-libs clean clean-results

# ---------------------------------------------------------------------------
help:
	@echo ""
	@echo "  make install               Install all dependencies via uv"
	@echo "  make fixtures              Generate test fixture PDFs in docs/"
	@echo "  make run PDF=<path>        Run all parsers against one PDF"
	@echo "  make run-all               Run all parsers against every PDF in docs/"
	@echo "  make run-libs PDF=<path>   Run a subset: LIBS=\"pdfplumber pymupdf\""
	@echo "  make clean                 Remove results/ and __pycache__"
	@echo "  make clean-results         Remove results/ only"
	@echo ""
	@echo "  Options (override with make VAR=value):"
	@echo "    PDF=<path>     PDF file for 'run' and 'run-libs'  (default: none)"
	@echo "    LIBS=<names>   Space-separated library names       (default: all)"
	@echo "    TIMEOUT=<sec>  Per-library timeout in seconds      (default: 300)"
	@echo "    WORKERS=<n>    Parallel workers per PDF            (default: all)"
	@echo "    RESULTS=<dir>  Output directory                    (default: results)"
	@echo ""

# ---------------------------------------------------------------------------
install:
	$(UV) sync

# ---------------------------------------------------------------------------
fixtures:
	$(PYTHON) scripts/create_test_fixtures.py

# ---------------------------------------------------------------------------
run:
ifndef PDF
	$(error PDF is not set. Usage: make run PDF=docs/my.pdf)
endif
	$(PYTHON) run_comparison.py $(PDF) \
		--results-dir $(RESULTS) \
		$(_TIMEOUT) $(_WORKERS) $(_LIBS)

# ---------------------------------------------------------------------------
run-all:
	$(PYTHON) run_comparison.py $(DOCS)/*.pdf \
		--results-dir $(RESULTS) \
		$(_TIMEOUT) $(_WORKERS) $(_LIBS)

# ---------------------------------------------------------------------------
run-libs:
ifndef PDF
	$(error PDF is not set. Usage: make run-libs PDF=docs/my.pdf LIBS="pdfplumber pymupdf")
endif
ifndef LIBS
	$(error LIBS is not set. Usage: make run-libs PDF=docs/my.pdf LIBS="pdfplumber pymupdf")
endif
	$(PYTHON) run_comparison.py $(PDF) \
		--results-dir $(RESULTS) \
		--libraries $(LIBS) \
		$(_TIMEOUT) $(_WORKERS)

# ---------------------------------------------------------------------------
clean-results:
	rm -rf $(RESULTS)

clean: clean-results
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"     -delete 2>/dev/null || true
