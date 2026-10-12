#!/usr/bin/env bash
# Fail if any tracked (or, before the first commit, any) text file still carries internal
# identifiers, machine paths or credentials. Usage: bash scripts/release/scrub_check.sh
cd "$(dirname "$0")/../.."
PATTERN='igYh|p7rr|HqJz|[Rr]ebuttal|REBUTTAL|ICLR|camready|CAMERA_READY|/mnt/md0_raid|/opt/conda|/home/jupyter|/home/moshiur|moshiur22|hypertrip-accv|HYPERTRIP_BASE|signal_use_vlm|[Cc]laude|[Aa]nthropic|Fable|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}'
ALLOW='scripts/release/scrub_check.sh|scripts/release/scrub_result_paths.py|hra/knockout/hypervis_adapter.py'
if git rev-parse --is-inside-work-tree >/dev/null 2>&1 && [ -n "$(git ls-files 2>/dev/null | head -1)" ]; then
  FILES=$(git ls-files)
else
  FILES=$(find . -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' -o -name '*.txt' -o -name '*.toml' -o -name '*.yml' -o -name '*.yaml' -o -name '*.cff' -o -name '*.tex' -o -name '*.json' -o -name 'Makefile' -o -name '.gitignore' \) -not -path './.git/*' -not -path './runs/*' -not -path './checkpoints/*' -not -path './data/*' -not -path './models/*' | sed 's|^\./||')
fi
HITS=$(echo "$FILES" | grep -vE "$ALLOW" | xargs -r grep -nE "$PATTERN" 2>/dev/null || true)
if [ -n "$HITS" ]; then
  echo "$HITS" | head -n 200
  echo "---- $(echo "$HITS" | wc -l) hit(s); scrub_check FAILED"
  exit 1
fi
echo "scrub_check: clean"
