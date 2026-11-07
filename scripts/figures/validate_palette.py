"""Python port of the dataviz skill's palette checks (OKLab, Machado-2009 CVD simulation):
lightness band, chroma floor, adjacent-pair CVD Delta E (protan/deutan; target 8, floor 6),
normal-vision floor 15, WCAG contrast >= 3 vs the surface.   python validate_palette.py "#hex,#hex,..." """
import math, sys
BAND, CHROMA, TARGET, FLOOR, NORMAL, CONTRAST = (0.43, 0.77), 0.10, 8.0, 6.0, 15.0, 3.0
M = {"protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
     "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
     "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]}
def lin(h):
    c = [int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
def oklab(r):
    l, m, s = (math.copysign(abs(v) ** (1 / 3), v) for v in (0.4122214708 * r[0] + 0.5363325363 * r[1] + 0.0514459929 * r[2],
              0.2119034982 * r[0] + 0.6806995451 * r[1] + 0.1073969566 * r[2], 0.0883024619 * r[0] + 0.2817188376 * r[1] + 0.6299787005 * r[2]))
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s, 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)
def sim(h, k):
    r = lin(h); return [max(0, min(1, sum(M[k][i][j] * r[j] for j in range(3)))) for i in range(3)]
def dE(a, b, k=None):
    x, y = oklab(sim(a, k) if k else lin(a)), oklab(sim(b, k) if k else lin(b))
    return 100 * math.dist(x, y)
def lum(h):
    r = lin(h); return 0.2126 * r[0] + 0.7152 * r[1] + 0.0722 * r[2]
def main(pal, surface="#fcfcfb"):
    ok = True
    for h in pal:
        L, a, b = oklab(lin(h)); C = math.hypot(a, b); cr = (lum(surface) + 0.05) / (lum(h) + 0.05)
        f = [n for n, bad in (("L", not BAND[0] <= L <= BAND[1]), ("C", C < CHROMA), ("contrast", cr < CONTRAST)) if bad]
        print(f"{h}  L={L:.3f} C={C:.3f} contrast={cr:.2f} {'FAIL ' + ','.join(f) if f else 'ok'}"); ok &= not f
    for a, b in zip(pal, pal[1:]):
        cvd = min(dE(a, b, "protan"), dE(a, b, "deutan")); nv = dE(a, b)
        st = "ok" if cvd >= TARGET and nv >= NORMAL else ("WARN (needs secondary encoding)" if cvd >= FLOOR and nv >= NORMAL else "FAIL")
        print(f"{a}-{b}: CVD {cvd:.1f}  normal {nv:.1f}  tritan {dE(a, b, 'tritan'):.1f}  {st}"); ok &= st != "FAIL"
    print("PASS" if ok else "FAIL")
if __name__ == "__main__":
    main(sys.argv[1].split(","))
