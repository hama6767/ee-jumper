#!/usr/bin/env python3
"""Recreate the minimal, pinned upstream asset subset. Run from the repo root."""

import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COMMIT = "61d065219fca767f3142c8f10aff59eae5a5a004"
BASE = f"https://raw.githubusercontent.com/KingKongRobotics/jumper/{COMMIT}/"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "src/ee_jumper/assets/jumper"


def fetch(path):
    return urllib.request.urlopen(BASE + path, timeout=60).read()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    original = fetch("assets/jumper/jumper.xml")
    tree = ET.fromstring(original)
    # Render collision geometry only; preserve original kinematics and inertials.
    for parent in tree.iter():
        for child in list(parent):
            if child.tag == "camera" or (child.tag == "geom" and child.get("class") == "visual"):
                parent.remove(child)
    used = {g.get("mesh") for g in tree.findall(".//worldbody//geom") if g.get("mesh")}
    assets = tree.find("asset")
    downloads = []
    for mesh in list(assets):
        if mesh.get("name") not in used:
            assets.remove(mesh)
            continue
        source = "assets/jumper/" + mesh.get("file")
        local = "meshes/" + Path(mesh.get("file")).name
        mesh.set("file", local)
        downloads.append((source, local))

    def save(item):
        source, local = item
        data = fetch(source)
        path = DEST / local
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {
            "source": source,
            "local": local,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(save, downloads))
    ET.indent(tree)
    xml = ET.tostring(tree, encoding="unicode")
    xml = (
        "<!-- Derived from KingKongRobotics/jumper; Copyright 2026 KingKong Robotics.\n     Apache-2.0. Modified: collision-only rendering, unused meshes/cameras removed,\n     mesh paths flattened. Original joints, masses and inertias retained. -->\n"
        + xml
        + "\n"
    )
    (DEST / "robot.xml").write_text(xml)
    (ROOT / "LICENSE").write_bytes(fetch("LICENSE"))
    (ROOT / "docs/UPSTREAM_NOTICE.txt").write_bytes(fetch("NOTICE"))
    manifest = {
        "repository": "https://github.com/KingKongRobotics/jumper",
        "commit": COMMIT,
        "source_xml_sha256": hashlib.sha256(original).hexdigest(),
        "derived_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "meshes": records,
    }
    (DEST / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Vendored {len(records)} meshes: {sum(r['bytes'] for r in records):,} bytes; commit {COMMIT}"
    )


if __name__ == "__main__":
    main()
