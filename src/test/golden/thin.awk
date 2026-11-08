# Make a golden reference from a run's diagnostics.txt: the header and about 40 rows (every k-th
# and the last), without Clock_time (wall-clock time, which no rerun reproduces).
# Usage: awk -f thin.awk diagnostics.txt > <name>.ref
BEGIN { FS = "," }
{ line[NR] = $0 }
NR == 1 { for (i = 1; i <= NF; i++) { h = $i; gsub(/[ \t\r]/, "", h); if (h == "Clock_time") skip = i } }
END {
    k = int((NR - 2) / 40) + 1
    for (r = 1; r <= NR; r++) {
        if (r > 1 && (r - 2) % k != 0 && r != NR) continue
        n = split(line[r], f, ",")
        out = ""
        for (i = 1; i <= n; i++) if (i != skip) out = out (out == "" ? "" : ",") f[i]
        print out
    }
}
