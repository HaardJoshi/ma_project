#!/bin/bash
# Compile all three chapters with their individual bibliographies
set -e

echo "=== Compiling Literature Review ==="
typst compile report/revised_lit_review/compile-litreview.typ report/revised_lit_review/02-lit-review-ver3.pdf 2>&1
echo "  -> Done: report/revised_lit_review/02-lit-review-ver3.pdf"

echo ""
echo "=== Compiling Methodology ==="
typst compile report/methodology_draft/compile-methodology.typ report/methodology_draft/revised-methodology-ver4.pdf 2>&1
echo "  -> Done: report/methodology_draft/revised-methodology-ver4.pdf"

echo ""
echo "=== Compiling Findings ==="
typst compile report/findings_draft/compile-findings.typ report/findings_draft/04-findings-ver3.pdf 2>&1
echo "  -> Done: report/findings_draft/04-findings-ver3.pdf"

echo ""
echo "=== All compilations complete ==="
