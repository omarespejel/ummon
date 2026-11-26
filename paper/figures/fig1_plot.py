"""Draw Figure 1 of the paper from fig1_verify_vs_size.csv.

One horizontal bar per system: verification time of one proof, in seconds on a linear axis, slowest first. Ummon's
bar is coloured and the others grey; a filled bar was measured by us, an outlined bar is reported by the system's own
source. A reported range is drawn to its upper end, with a mark at its lower end. Systems slower than SLOW seconds get no bar; an inset lists them with their reported times, so the linear axis
stays readable. Only rows with bar=yes are drawn. A multiple of our time is printed only for rows measured on the same machine
as ours (same_machine_as_ours=yes). Standard library only.

Usage: python3 paper/figures/fig1_plot.py [csv] [svg]
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "fig1_verify_vs_size.csv"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "fig1_verify_vs_size.svg"

W = 820
LEFT, RIGHT, TOP = 170, 210, 56
ROW, BAR = 58, 26
XMAX = 14.0
SLOW = XMAX
OURS, GREY = "#2a78d6", "#8c8b86"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def fmt(s):
    if s >= 3600:
        return f"{s / 3600:.1f} h"
    if s >= 100:
        return f"{s:.2f}".rstrip("0").rstrip(".") + " s"
    return f"{s * 1000:.2f} ms" if s < 0.1 else (f"{s * 1000:.1f} ms" if s < 1 else f"{s:.2f} s")


def main():
    allrows = [r for r in csv.DictReader(open(SRC, encoding="utf-8")) if r["bar"] == "yes"]
    slow = sorted([r for r in allrows if float(r["verify_seconds"]) > SLOW], key=lambda r: -float(r["verify_seconds"]))
    rows = [r for r in allrows if float(r["verify_seconds"]) <= SLOW]
    rows.sort(key=lambda r: -float(r["verify_seconds"]))
    ours = next(float(r["verify_seconds"]) for r in rows if r["system"] == "Ummon")
    plot_w = W - LEFT - RIGHT
    H = TOP + ROW * len(rows) + 48
    sx = lambda v: LEFT + v / XMAX * plot_w
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="Helvetica, Arial, sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>']
    base = TOP + ROW * len(rows)
    for t in range(0, int(XMAX) + 1, 2):
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" y1="{TOP - 8}" x2="{x:.1f}" y2="{base}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{base + 16}" text-anchor="middle" fill="{MUTED}">{t} s</text>')
    out.append(f'<text x="{LEFT + plot_w / 2:.1f}" y="{base + 36}" text-anchor="middle" fill="{INK}">'
               'Verification time of one proof (seconds, linear scale)</text>')
    for k, r in enumerate(rows):
        y = TOP + k * ROW + (ROW - BAR) / 2
        v = float(r["verify_seconds"])
        top = float(r["verify_seconds_high"]) if r["verify_seconds_high"] else v
        cut = top > XMAX
        w = max(sx(min(top, XMAX)) - LEFT, 2.0)
        colour = OURS if r["system"] == "Ummon" else GREY
        measured = r["provenance"] == "measured by us"
        fill = colour if measured else SURFACE
        rx = 3 if w > 8 else 0
        out.append(f'<rect x="{LEFT}" y="{y:.1f}" width="{w:.1f}" height="{BAR}" rx="{rx}" fill="{fill}" '
                   f'stroke="{colour}" stroke-width="{0 if measured else 2}"/>')
        out.append(f'<text x="{LEFT - 12}" y="{y + 11:.1f}" text-anchor="end" fill="{INK}" font-weight="bold">'
                   f'{r["system"]}</text>')
        out.append(f'<text x="{LEFT - 12}" y="{y + 25:.1f}" text-anchor="end" fill="{MUTED}">{r["params_label"]}</text>')
        if top > v:
            tx = sx(v)
            out.append(f'<line x1="{tx:.1f}" y1="{y:.1f}" x2="{tx:.1f}" y2="{y + BAR:.1f}" stroke="{colour}" stroke-width="2"/>')
        if cut:
            bx = LEFT + w - 14
            out.append(f'<rect x="{bx - 3:.1f}" y="{y - 2:.1f}" width="9" height="{BAR + 4}" fill="{SURFACE}"/>')
            for dx in (0, 6):
                out.append(f'<line x1="{bx + dx - 3:.1f}" y1="{y + BAR + 3:.1f}" x2="{bx + dx + 3:.1f}" y2="{y - 3:.1f}" '
                           f'stroke="{colour}" stroke-width="1.5"/>')
        hi = r["verify_seconds_high"]
        label = r.get("label") or (f"{fmt(v)[:-2]} to {fmt(float(hi))}" if hi else fmt(v))
        if cut:
            label += ", past the axis"
        if r["same_machine_as_ours"] == "yes":
            label += f", {v / ours:.1f} times ours"
        out.append(f'<text x="{LEFT + w + 8:.1f}" y="{y + BAR / 2 + 4:.1f}" fill="{INK}">{label}</text>')
    lx, ly = LEFT, 22
    out.append(f'<rect x="{lx}" y="{ly - 10}" width="18" height="12" fill="{OURS}"/>')
    out.append(f'<text x="{lx + 24}" y="{ly}" fill="{INK}">Ummon, measured by us</text>')
    out.append(f'<rect x="{lx + 190}" y="{ly - 10}" width="18" height="12" fill="{GREY}"/>')
    out.append(f'<text x="{lx + 214}" y="{ly}" fill="{INK}">others, measured by us</text>')
    out.append(f'<rect x="{lx + 390}" y="{ly - 10}" width="18" height="12" fill="{SURFACE}" stroke="{GREY}" stroke-width="2"/>')
    out.append(f'<text x="{lx + 414}" y="{ly}" fill="{INK}">reported by its source</text>')
    if slow:
        ix, iy = LEFT + plot_w * 0.5, TOP + ROW * 3.3
        out.append(f'<text x="{ix:.1f}" y="{iy:.1f}" fill="{INK}" font-weight="bold">Past the axis, as reported:</text>')
        for k, r in enumerate(slow):
            out.append(f'<text x="{ix:.1f}" y="{iy + 18 * (k + 1):.1f}" fill="{INK}">{r["system"]} ({r["params_label"]}): '
                       f'{r.get("label") or fmt(float(r["verify_seconds"]))}</text>')
    out.append("</svg>")
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(rows)} bars; ours is {ours / max(float(r['verify_seconds']) for r in rows) * 100:.2f}% of the longest)")


if __name__ == "__main__":
    main()
