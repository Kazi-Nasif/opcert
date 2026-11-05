"""What the Endoscapes-BBox201 anatomy annotations contain, per split."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[2] / "data" / "endoscapes"
for s in ["train", "val", "test"]:
    d = json.load(open(D / s / "annotation_coco.json"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    per = collections.defaultdict(list)
    for a in d["annotations"]:
        per[a["image_id"]].append(a)
    det = [i for i in d["images"] if i.get("is_det_keyframe")]
    print(f"== {s}: images {len(d['images'])}  det-keyframes {len(det)}  annotated {len(per)}  anns {len(d['annotations'])}")
    print("   instances", dict(collections.Counter(cats[a["category_id"]] for a in d["annotations"])))
    nb = [len(per[i["id"]]) for i in det]
    print(f"   boxes/frame mean {np.mean(nb):.1f} max {max(nb)}")
    kinds = collections.Counter()
    for i in det:
        c = collections.Counter(cats[a["category_id"]] for a in per[i["id"]])
        kinds[(c["cystic_duct"], c["cystic_artery"])] += 1
    print("   (#duct, #artery) per frame:", dict(sorted(kinds.items())))
    both = [i for i in det if {"cystic_duct", "cystic_artery"} <= {cats[a["category_id"]] for a in per[i["id"]]}]
    print(f"   frames with both: {len(both)}  CVS-rated: {sum(bool(i.get('ds')) for i in both)}  "
          f"CVS achieved (mean>=0.5 on all): {sum(all(x >= 0.5 for x in i['ds']) for i in both if i.get('ds'))}")
    print("   ds on det frames:", sum(bool(i.get('ds')) for i in det), "/", len(det))
    a = d["annotations"][0]
    print("   ann keys:", {k: (v if k != 'segmentation' else '...') for k, v in a.items()})
