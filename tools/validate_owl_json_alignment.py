#!/usr/bin/env python3
"""Validate OWL ↔ JSON rule alignment for Context-Sync.

Usage:
    python tools/validate_owl_json_alignment.py ontology/news_rules_1024.json ontology/context_sync_app_centered_ontology_1024.owl

Exit code 0: valid
Exit code 1: suspicious or structural mismatch found
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import rdflib

NS = rdflib.Namespace("http://www.context-sync.com/ontology/news-app#")
DIM_PREFIX = "dim_"
FRAME_PREFIX = "frame_"
VALUE_PREFIX = "val_"
SCHEMA_PREFIX = "schema_"


def local_name(uri: Any) -> str:
    text = str(uri)
    if "#" in text:
        text = text.rsplit("#", 1)[1]
    return text


def strip_prefix(name: str, prefix: str) -> str:
    return name[len(prefix):] if name.startswith(prefix) else name


def load_owl_maps(owl_path: Path):
    g = rdflib.Graph()
    g.parse(str(owl_path), format="xml")

    schema_map = {}
    for s in g.subjects(NS.measuresDimension, None):
        sid = strip_prefix(local_name(s), SCHEMA_PREFIX)
        schema_map[sid] = {
            "operational_dimensions": sorted(strip_prefix(local_name(o), DIM_PREFIX) for o in g.objects(s, NS.measuresDimension)),
            "frames": sorted(strip_prefix(local_name(o), FRAME_PREFIX) for o in g.objects(s, NS.constrainsFrame)),
            "anchors": sorted(strip_prefix(local_name(o), VALUE_PREFIX) for o in g.objects(s, NS.anchoredByValue)),
        }

    anchor_dims = defaultdict(set)
    for s in g.subjects(NS.calibratesDimension, None):
        aid = strip_prefix(local_name(s), VALUE_PREFIX)
        for o in g.objects(s, NS.calibratesDimension):
            anchor_dims[aid].add(strip_prefix(local_name(o), DIM_PREFIX))

    return schema_map, {k: sorted(v) for k, v in anchor_dims.items()}


def classify_rule(rule: dict, schema_map: dict, anchor_dims: dict) -> tuple[str, list[str]]:
    errors = []
    schema_id = str(rule.get("schema_id", ""))
    dim = str(rule.get("dimension", ""))
    frame = str(rule.get("target_frame", ""))
    anchor = str(rule.get("value_anchor", ""))

    schema = schema_map.get(schema_id)
    if not schema:
        errors.append(f"unknown_schema:{schema_id}")
        return "suspicious", errors

    operational = set(schema["operational_dimensions"])
    normative = set(anchor_dims.get(anchor, []))

    if frame not in schema["frames"]:
        errors.append(f"frame_mismatch:{frame}!={schema['frames']}")
    if anchor not in schema["anchors"]:
        errors.append(f"anchor_mismatch:{anchor}!={schema['anchors']}")
    if not normative:
        errors.append(f"unknown_anchor:{anchor}")

    schema_ok = dim in operational
    anchor_ok = dim in normative

    if schema_ok and anchor_ok:
        status = "aligned"
    elif schema_ok and not anchor_ok:
        status = "cross_mapping"
    elif anchor_ok and not schema_ok:
        status = "anchor_only_suspicious"
    else:
        status = "suspicious"

    if status in {"anchor_only_suspicious", "suspicious"}:
        errors.append(f"dimension_mismatch:json={dim}, operational={sorted(operational)}, normative={sorted(normative)}")

    return status, errors


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip())
        return 2
    json_path = Path(sys.argv[1])
    owl_path = Path(sys.argv[2])

    rules_data = json.loads(json_path.read_text(encoding="utf-8"))
    rules = rules_data.get("rules", [])
    schema_map, anchor_dims = load_owl_maps(owl_path)

    counts = Counter()
    errors_by_rule = {}
    by_schema = Counter()
    for rule in rules:
        status, errors = classify_rule(rule, schema_map, anchor_dims)
        counts[status] += 1
        by_schema[(rule.get("schema_id"), status)] += 1
        if errors:
            errors_by_rule[rule.get("rule_id", "<missing>")] = errors

        # Optional embedded metadata consistency checks
        if rule.get("operational_dimension") and rule["operational_dimension"] not in schema_map.get(rule.get("schema_id"), {}).get("operational_dimensions", []):
            errors_by_rule.setdefault(rule.get("rule_id", "<missing>"), []).append("metadata_operational_dimension_mismatch")
        nd = rule.get("normative_dimension")
        if nd:
            nd_set = set(nd if isinstance(nd, list) else [nd])
            expected = set(anchor_dims.get(rule.get("value_anchor", ""), []))
            if nd_set != expected:
                errors_by_rule.setdefault(rule.get("rule_id", "<missing>"), []).append(f"metadata_normative_dimension_mismatch:{sorted(nd_set)}!={sorted(expected)}")

    print(f"ruleset: {rules_data.get('ruleset_id', '-')}")
    print(f"version: {rules_data.get('version', '-')}")
    print(f"rules: {len(rules)}")
    print("status counts:")
    for key in ["aligned", "cross_mapping", "anchor_only_suspicious", "suspicious"]:
        print(f"  {key}: {counts.get(key, 0)}")

    print("schema/status counts:")
    for (schema_id, status), count in sorted(by_schema.items()):
        if status != "aligned":
            print(f"  {schema_id} / {status}: {count}")

    fatal_statuses = counts.get("anchor_only_suspicious", 0) + counts.get("suspicious", 0)
    if fatal_statuses or errors_by_rule:
        print("\nFAIL ❌")
        for rid, errs in list(errors_by_rule.items())[:40]:
            print(f"  {rid}: {', '.join(errs)}")
        if len(errors_by_rule) > 40:
            print(f"  ... {len(errors_by_rule) - 40} more")
        return 1

    print("\nPASS ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
