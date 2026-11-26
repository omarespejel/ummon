"""Draw the crossover plot of Appendix D from fig2_prove_once_verify_n.csv, and print the crossover table.

Total machine time to prove one proof and check it N times: prover seconds per request + N × verifier seconds.
The two drawn lines (line=yes) are each system as run; the other rows give the crossover under other accounting
choices, shown as a band. The sum adds seconds on different machines (a cloud GPU host for proving, one laptop
for verifying) and is not a cost in money. Standard library only.

Usage: python3 paper/figures/fig2_plot.py [csv] [svg]
"""
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "fig2_prove_once_verify_n.csv"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "fig2_prove_once_verify_n.svg"

W, H = 820, 520
LEFT, RIGHT, TOP, BOTTOM = 96, 190, 48, 70
XLOG, YLOG = (0, 5), (2, 6.5)
OURS, GREY = "#2a78d6", "#8c8b86"
INK, MUTED, GRID, SURFACE, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#ece9e2"


def crossover(a, b):
    """Checks at which a's total equals b's; a proves slower and verifies faster than b."""
    return (a["p"] - b["p"]) / (b["v"] - a["v"])


def table(rows):
    ours = [r for r in rows if r["system"] == "Ummon"]
    theirs = [r for r in rows if r["system"] == "FlashProve"]
    out = []
    for a in ours:
        for b in theirs:
            out.append((a["accounting"], b["accounting"], crossover(a, b)))
    return out


def main():
    rows = [dict(r, p=float(r["prover_seconds"]), v=float(r["verify_seconds"]))
            for r in csv.DictReader(open(SRC, encoding="utf-8"))]
    pairs = table(rows)
    for a, b, n in pairs:
        print(f"{n:8.1f}  Ummon {a} | FlashProve {b}")
    lo, hi = min(n for *_, n in pairs), max(n for *_, n in pairs)
    drawn = [r for r in rows if r["line"] == "yes"]
    n_run = crossover(*sorted(drawn, key=lambda r: r["system"] != "Ummon"))
    print(f"as run: {n_run:.1f} checks; range over the accounting choices: {lo:.1f} to {hi:.1f}")

    pw, ph = W - LEFT - RIGHT, H - TOP - BOTTOM
    sx = lambda n: LEFT + (math.log10(n) - XLOG[0]) / (XLOG[1] - XLOG[0]) * pw
    sy = lambda t: TOP + ph - (math.log10(t) - YLOG[0]) / (YLOG[1] - YLOG[0]) * ph
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="Helvetica, Arial, sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
           f'<rect x="{sx(lo):.1f}" y="{TOP}" width="{sx(hi) - sx(lo):.1f}" height="{ph}" fill="{BAND}"/>',
           f'<text x="{(sx(lo) + sx(hi)) / 2:.1f}" y="{TOP - 8}" text-anchor="middle" fill="{MUTED}">'
           f'crossover over the accounting choices: {lo:.0f} to {hi:.0f} checks</text>']
    for e in range(XLOG[0], XLOG[1] + 1):
        x = sx(10 ** e)
        out.append(f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{TOP + ph}" stroke="{GRID}"/>')
        out.append(f'<text x="{x:.1f}" y="{TOP + ph + 18}" text-anchor="middle" fill="{MUTED}">{10 ** e:,}</text>')
    for e in range(math.ceil(YLOG[0]), math.floor(YLOG[1]) + 1):
        y = sy(10 ** e)
        out.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + pw}" y2="{y:.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{LEFT - 8}" y="{y + 4:.1f}" text-anchor="end" fill="{MUTED}">{10 ** e:,} s</text>')
    out.append(f'<text x="{LEFT + pw / 2:.1f}" y="{H - 24}" text-anchor="middle" fill="{INK}">'
               'Checks of one proof, N (log scale)</text>')
    out.append(f'<text transform="translate(22 {TOP + ph / 2:.1f}) rotate(-90)" text-anchor="middle" fill="{INK}">'
               'Prover time per request + N × verifier time (seconds, log scale)</text>')
    for r in drawn:
        colour = OURS if r["system"] == "Ummon" else GREY
        pts = [(sx(n), sy(r["p"] + n * r["v"])) for n in (10 ** (k / 20) for k in range(XLOG[0] * 20, XLOG[1] * 20 + 1))]
        pts = [(x, y) for x, y in pts if TOP <= y <= TOP + ph]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out.append(f'<polyline points="{d}" fill="none" stroke="{colour}" stroke-width="2.5"/>')
        x, y = pts[-1]
        out.append(f'<text x="{x + 8:.1f}" y="{y - 6:.1f}" fill="{INK}" font-weight="bold">{r["system"]}</text>')
        out.append(f'<text x="{x + 8:.1f}" y="{y + 9:.1f}" fill="{MUTED}">{r["p"]:,.1f} s + N × {r["v"] * 1000:,.2f} ms</text>')
    t_run = drawn[0]["p"] + n_run * drawn[0]["v"]
    out.append(f'<circle cx="{sx(n_run):.1f}" cy="{sy(t_run):.1f}" r="4.5" fill="{INK}"/>')
    out.append(f'<text x="{sx(n_run) + 8:.1f}" y="{sy(t_run) + 18:.1f}" fill="{INK}">as run: {n_run:.0f} checks</text>')
    out.append("</svg>")
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
