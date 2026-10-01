"""RECOMB variant of fig_tradeoff: same data and layout, RECOMB vocabulary on the axes.

"Composition leakage of z" becomes "residual niche signal in z" (STYLE.md: "leakage" is kept
for transcript spill-over only). Runs fig_tradeoff.py with the labels and output name swapped,
so the two figures cannot drift apart. Outputs ../recomb_tradeoff.pdf (+ .png).
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "fig_tradeoff.py").read_text()
for old, new in [
    ('OUT = HERE.parent / "fig_tradeoff"', 'OUT = HERE.parent / "recomb_tradeoff"'),
    (r'XLAB = r"composition leakage of $\mathbf{z}$ (\%)"',
     r'XLAB = r"residual niche signal in $\mathbf{z}$ (\%)"'),
    (r'r"low leakage, keeps cycle state"', r'r"little niche signal, keeps cycle state"'),
]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
exec(compile(src, str(HERE / "fig_tradeoff.py"), "exec"), {"__file__": str(HERE / "fig_tradeoff.py"), "__name__": "__main__"})
