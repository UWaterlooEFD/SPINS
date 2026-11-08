#!/usr/bin/env bash
# Golden-output regression: short runs of upstream cases whose diagnostics must match stored
# references (<name>.ref here) within a stated relative tolerance, so that performance work
# (compiler flags, FFTW planning, solver changes) cannot change answers unnoticed.
#
# Usage: run_golden.sh <build dir> [name ...]
#   <build dir>      CMake build tree holding src/cases/<case>/<case>.x
#   name ...         run only these configurations (default: all in configs.txt)
#   GOLDEN_NP        MPI ranks (default 1; the references hold on 1-4 ranks)
#   GOLDEN_OUT       root of the run directories (default ./golden_runs)
#   GOLDEN_UPDATE=1  write new references from these runs instead of comparing
# Exit status: the number of configurations that failed.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=$(cd "$here/../../.." && pwd)
build=$(cd "${1:?usage: run_golden.sh <build dir> [name ...]}" && pwd)
shift
np=${GOLDEN_NP:-1}
out=${GOLDEN_OUT:-$PWD/golden_runs}
fail=0
while read -r name case_name rtol scales overrides; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    if [ $# -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$name"; then continue; fi
    d="$out/$name"
    rm -rf "$d"
    mkdir -p "$d"
    cp "$src/src/cases/$case_name/spins.conf" "$d/spins.conf"
    for kv in $overrides; do # key=value, replacing the key's line (or appending it)
        key="${kv%%=*}" val="${kv#*=}"
        pattern="^[[:space:]]*#?[[:space:]]*${key}[[:space:]]*="
        if grep -qE "$pattern" "$d/spins.conf"; then
            sed -i -E "s|${pattern}.*|${key} = ${val}|" "$d/spins.conf"
        else
            echo "${key} = ${val}" >> "$d/spins.conf"
        fi
    done
    # stdin from /dev/null: mpirun reads stdin and would swallow the rest of configs.txt
    if ! (cd "$d" && mpirun -np "$np" "$build/src/cases/$case_name/$case_name.x" < /dev/null > run.log 2>&1); then
        echo "golden $name: FAIL (the run failed; see $d/run.log)"
        fail=$((fail + 1))
        continue
    fi
    if [ -n "${GOLDEN_UPDATE:-}" ]; then
        awk -f "$here/thin.awk" "$d/diagnostics.txt" > "$here/$name.ref"
        echo "golden $name: reference written ($(($(wc -l < "$here/$name.ref") - 1)) rows)"
    else
        awk -v name="$name" -v rtol="$rtol" -v scales="$scales" -f "$here/compare.awk" "$here/$name.ref" "$d/diagnostics.txt" || fail=$((fail + 1))
    fi
done < "$here/configs.txt"
exit $fail
