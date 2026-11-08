# Compare a run's diagnostics.txt with a golden reference (both comma-separated, header first).
# Usage: awk -v name=<config> -v rtol=<tolerance> -v scales=<col:scalecol,...|-> \
#            -f compare.awk <reference> <diagnostics.txt>
# Rows are matched by Iter. Every reference column except Iter must agree within
# rtol x scale, where scale is the column's largest |value| in the reference (or that of the column
# named in scales, for a column that sits near zero, e.g. Min_density:Max_density), 1 if that is 0.
# Every out-of-tolerance column is reported before failing; NaN, Inf, a missing row or a missing
# column fail at once. Exit status 0 = pass.
BEGIN {
    FS = ","
    nsc = split(scales == "-" ? "" : scales, sc, ",")
    for (i = 1; i <= nsc; i++) { split(sc[i], p, ":"); scale_of[p[1]] = p[2] }
}
FNR == 1 {
    for (i = 1; i <= NF; i++) { h = $i; gsub(/[ \t\r]/, "", h); if (FILENAME == ARGV[1]) refname[i] = h; else runcol[h] = i }
    if (FILENAME == ARGV[1]) nref = NF
    next
}
FILENAME == ARGV[1] {
    it = $1 + 0; nrows++; order[nrows] = it
    for (i = 1; i <= NF; i++) {
        ref[it, i] = $i + 0
        if (abs($i + 0) > big[refname[i]]) big[refname[i]] = abs($i + 0)
    }
    next
}
{
    it = $runcol["Iter"] + 0
    if (!((it, 1) in ref)) next
    seen[it] = 1
    for (i = 1; i <= nref; i++) if (refname[i] in runcol) run[it, i] = $runcol[refname[i]]
}
function abs(x) { return x < 0 ? -x : x }
END {
    bad = 0; hard = ""
    for (i = 2; i <= nref; i++) if (!(refname[i] in runcol)) hard = hard " missing column " refname[i] ";"
    for (r = 1; r <= nrows; r++) {
        it = order[r]
        if (!(it in seen)) { hard = hard " missing row Iter " it ";"; continue }
        for (i = 2; i <= nref; i++) {
            if (!(refname[i] in runcol)) continue
            v = run[it, i]
            if (tolower(v) ~ /nan|inf/) { hard = hard " " refname[i] " = " v " at Iter " it ";"; continue }
            s = big[(refname[i] in scale_of) ? scale_of[refname[i]] : refname[i]]
            if (s == 0) s = 1
            e = abs(v - ref[it, i]) / s
            if (e > worst[i]) { worst[i] = e; worst_it[i] = it }
        }
    }
    wi = 2
    for (i = 2; i <= nref; i++) {
        if (worst[i] > worst[wi]) wi = i
        if (worst[i] > rtol) { bad++; detail = detail sprintf(" %s %.2e (Iter %d);", refname[i], worst[i], worst_it[i]) }
    }
    if (hard != "") { printf "golden %s: FAIL:%s\n", name, hard; exit 1 }
    if (bad) { printf "golden %s: FAIL, %d column(s) over rtol %g:%s\n", name, bad, rtol, detail; exit 1 }
    printf "golden %s: PASS (%d rows; worst %s %.2e, rtol %g)\n", name, nrows, refname[wi], worst[wi], rtol
}
