"""Clip/cut events in CholecT50: how many frames, which targets, per video."""
import json
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "cholect50" / "CholecT50" / "labels"
tot, per_vid, first = Counter(), {}, {}
for f in sorted(D.glob("VID*.json")):
    d = json.loads(f.read_text()); cat = d["categories"]
    tgt = cat["target"]; v = int(d["video"]); c = Counter()
    for fr, rows in d["annotations"].items():
        for r in rows:
            inst, verb, t = r[1], r[7], r[8]
            if inst == 4 and verb == 4:                      # clipper, clip
                c[("clip", tgt[str(t)])] += 1
                if t in (2, 3):
                    first.setdefault(v, int(fr))
            if inst == 3 and verb == 5:                      # scissors, cut
                c[("cut", tgt[str(t)])] += 1
    per_vid[v] = c; tot.update(c)
print("videos", len(per_vid), "| total clip/cut frames by target:")
for k, n in tot.most_common(): print("  ", k, n)
cl = [sum(n for (a, _), n in c.items() if a == "clip") for c in per_vid.values()]
print("clip frames per video: min", min(cl), "median", sorted(cl)[len(cl) // 2], "max", max(cl))
print("videos with a duct/artery clip:", len(first), " ids:", sorted(per_vid))
