#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

build=true
if [ "$#" -gt 1 ]; then
  echo "usage: $0 [--build|--no-build]" >&2
  exit 2
fi
case "${1:-}" in
  ""|--build) ;;
  --no-build) build=false ;;
  *)
    echo "usage: $0 [--build|--no-build]" >&2
    exit 2
    ;;
esac

for paper in 01_theory_formalization 02_empirical_validation; do
  en_dir="$repo_root/papers/submission/$paper/en/sections"
  ko_dir="$repo_root/papers/submission/$paper/ko/sections"

  en_files=$(find "$en_dir" -maxdepth 1 -name '*.tex' -printf '%f\n' | sort)
  ko_files=$(find "$ko_dir" -maxdepth 1 -name '*.tex' -printf '%f\n' | sort)
  if [ "$en_files" != "$ko_files" ]; then
    echo "$paper: EN/KO section file lists differ" >&2
    exit 1
  fi

  for file in $en_files; do
    en_file="$en_dir/$file"
    ko_file="$ko_dir/$file"

    for env in definition assumption lemma proposition theorem corollary; do
      en_count=$(rg -c "\\begin\{$env\}" "$en_file" || true)
      ko_count=$(rg -c "\\begin\{$env\}" "$ko_file" || true)
      [ -n "$en_count" ] || en_count=0
      [ -n "$ko_count" ] || ko_count=0
      if [ "$en_count" != "$ko_count" ]; then
        echo "$paper/$file: $env count EN=$en_count KO=$ko_count" >&2
        exit 1
      fi
    done

    en_nums=$(rg -o '[0-9]+\.[0-9]+' "$en_file" | sort || true)
    ko_nums=$(rg -o '[0-9]+\.[0-9]+' "$ko_file" | sort || true)
    if [ "$en_nums" != "$ko_nums" ]; then
      echo "$paper/$file: decimal literal multisets differ" >&2
      diff -u <(printf '%s\n' "$en_nums") <(printf '%s\n' "$ko_nums") || true
      exit 1
    fi
  done
done

section_dirs="$repo_root/papers/submission/01_theory_formalization/en/sections
$repo_root/papers/submission/01_theory_formalization/ko/sections
$repo_root/papers/submission/02_empirical_validation/en/sections
$repo_root/papers/submission/02_empirical_validation/ko/sections"

if rg -n '\$[^$]*\\\\[[:alpha:]]' $section_dirs; then
  echo 'double-escaped LaTeX command found' >&2
  exit 1
fi

if rg -n -U '\\begin\{(theorem|proposition|definition|assumption|lemma|corollary)\}(\[[^]]*\])?\r?\n[[:space:]]*\r?\n' $section_dirs; then
  echo 'blank line immediately after theorem-like environment opening' >&2
  exit 1
fi

if rg -n \
  -e 'StrTrack\(\\mathcal S\)' \
  -e 'first, second, and fourth' \
  -e "Paper 3'supplies" \
  -e 'area and unit survival' \
  -e 'Structural Imagination Theory' \
  -e 'two effects should not be conflated' \
  -e 'that places generator relations in the simplicial one-skeleton' \
  $section_dirs; then
  echo 'stale manuscript wording found' >&2
  exit 1
fi

for main in \
  "$repo_root/papers/submission/01_theory_formalization/en/main.tex" \
  "$repo_root/papers/submission/01_theory_formalization/ko/main.tex" \
  "$repo_root/papers/submission/02_empirical_validation/en/main.tex" \
  "$repo_root/papers/submission/02_empirical_validation/ko/main.tex"; do
  if ! rg -q '^\\date\{' "$main"; then
    echo "$main: missing explicit date" >&2
    exit 1
  fi
done

require_phrase() {
  local file="$1"
  local phrase="$2"
  if ! rg -q -F "$phrase" "$file"; then
    echo "$file: required bilingual content anchor missing: $phrase" >&2
    exit 1
  fi
}

require_phrase \
  "$repo_root/papers/submission/01_theory_formalization/en/sections/abstract.tex" \
  'quantitative capacity bound on the relational defect'
require_phrase \
  "$repo_root/papers/submission/01_theory_formalization/ko/sections/abstract.tex" \
  'relational defect의 정량적 상한'
require_phrase \
  "$repo_root/papers/submission/02_empirical_validation/en/sections/conclusion.tex" \
  'a positive TSI-minus-dense contrast occurs only in the Paper 4 family'
require_phrase \
  "$repo_root/papers/submission/02_empirical_validation/ko/sections/conclusion.tex" \
  '양의 TSI-minus-dense 대비는 Paper 4 family에서만 나타난다'

en_ledger="$repo_root/papers/submission/02_empirical_validation/en/evidence_ledger.md"
ko_ledger="$repo_root/papers/submission/02_empirical_validation/ko/evidence_ledger.md"
en_rows=$(rg -c '^\| Paper [34]' "$en_ledger")
ko_rows=$(rg -c '^\| Paper [34]' "$ko_ledger")
if [ "$en_rows" != "$ko_rows" ]; then
  echo "evidence ledger row count differs: EN=$en_rows KO=$ko_rows" >&2
  exit 1
fi
en_artifacts=$(rg -o '[A-Za-z0-9_./-]+\.json' "$en_ledger" | sed 's#^.*/##' | sort)
ko_artifacts=$(rg -o '[A-Za-z0-9_./-]+\.json' "$ko_ledger" | sed 's#^.*/##' | sort)
if [ "$en_artifacts" != "$ko_artifacts" ]; then
  echo 'evidence ledger artifact inventory differs between EN and KO' >&2
  diff -u <(printf '%s\n' "$en_artifacts") <(printf '%s\n' "$ko_artifacts") || true
  exit 1
fi

if [ "$build" = true ]; then
  for dir in \
    "$repo_root/papers/submission/01_theory_formalization/en" \
    "$repo_root/papers/submission/01_theory_formalization/ko" \
    "$repo_root/papers/submission/02_empirical_validation/en" \
    "$repo_root/papers/submission/02_empirical_validation/ko"; do
    latexmk -cd -xelatex -interaction=nonstopmode -halt-on-error "$dir/main.tex"
    txt=$(mktemp)
    pdftotext "$dir/main.pdf" "$txt"
    if rg -n 'Deltamath|inf ty|Every map The discrete|Undefined control sequence' "$txt"; then
      echo "$dir/main.pdf: broken rendered text found" >&2
      exit 1
    fi
    rm -f "$txt"
  done
fi

echo 'Bilingual manuscript parity checks passed.'
