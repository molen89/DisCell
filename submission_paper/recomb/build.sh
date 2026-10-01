#!/usr/bin/env bash
# Build main.pdf and supplement.pdf (cross-referenced via xr-hyper) and report the main-text length.
# Usage: ./build.sh            (from any directory)
set -uo pipefail
cd "$(dirname "$0")"

LMK=(latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error)
run() { "${LMK[@]}" "${@:2}" "$1" > "build_${1%.tex}.out" 2>&1 || { echo "FAILED: $1 (see build_${1%.tex}.out, ${1%.tex}.log)"; tail -30 "build_${1%.tex}.out"; exit 1; }; }

# Two rounds, so that each document sees the other's final .aux; the second round is forced (-g)
# because latexmk does not track the other document's .aux when it was missing at the first run.
run supplement.tex; run main.tex
run supplement.tex -g; run main.tex -g
run main.tex -g   # once more, to embed the final supplement.pdf after the references

# Main-text pages: \mainpages is written at the page where the bibliography starts
# (page numbering restarts at 1 after the title page, which is excluded).
mainpages=$(sed -n 's/.*\\gdef *\\mainpages *{\([0-9]*\)}.*/\1/p' main.aux | tail -1)
total=$(pdfinfo main.pdf | awk '/^Pages:/{print $2}')
supp=$(pdfinfo supplement.pdf | awk '/^Pages:/{print $2}')
echo "main.pdf: ${total} pages in all (1 title page + main text + bibliography)"
echo "main text: ${mainpages} pages (limit 10)$( [ "${mainpages:-0}" -gt 10 ] && echo '  ** OVER THE LIMIT **')"
echo "supplement.pdf: ${supp} pages"

# Undefined references and citations.
for d in main supplement; do
  n=$(grep -c "Reference .* undefined" "$d.log")
  c=$(grep -c "Citation .* undefined" "$d.log")
  echo "$d: $n undefined reference warnings, $c undefined citation warnings"
  grep -o "Reference \`[^']*' on page [0-9]* undefined" "$d.log" | sed "s/Reference \`\([^']*\)'.*/\1/" | sort -u | tr '\n' ' ' | sed 's/^./  undefined: &/'; [ "$n" -gt 0 ] && echo
done

# Label namespaces must be disjoint (both documents import the other's labels without a prefix).
dups=$(comm -12 <(grep -o '^\\newlabel{[^}@]*}' main.aux | sort -u) <(grep -o '^\\newlabel{[^}@]*}' supplement.aux | sort -u))
[ -n "$dups" ] && { echo "WARNING: labels defined in both main and supplement:"; echo "$dups" | sed 's/\\newlabel/  /'; }
exit 0
