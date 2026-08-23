import json
import logging
import math
import re
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("promote")

app = FastAPI()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CANONICAL_ID_RE = re.compile(r"^[1-9][0-9]*$")

TIMESTAMP_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})"
    r"(\.\d{1,3})?"
    r"(Z|[+-]\d{2}:\d{2})$"
)


def is_canonical_version_id(v: Any) -> bool:
    if not isinstance(v, str) or not CANONICAL_ID_RE.match(v):
        return False
    try:
        return int(v) <= 2**53 - 1
    except ValueError:
        return False


def parse_timestamp(s: Any) -> Optional[datetime]:
    if not isinstance(s, str):
        return None
    m = TIMESTAMP_RE.match(s)
    if not m:
        return None
    year, month, day, hour, minute, second, frac, offset = m.groups()
    try:
        base = f"{year}-{month}-{day}T{hour}:{minute}:{second}"
        if frac:
            micros = (frac[1:] + "000000")[:6]
            base += f".{micros}"
        base += "+00:00" if offset == "Z" else offset
        return datetime.fromisoformat(base)
    except ValueError:
        return None


def is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def in_unit_interval(x: Any) -> bool:
    return is_finite_number(x) and 0.0 <= float(x) <= 1.0


def is_nonneg_finite(x: Any) -> bool:
    return is_finite_number(x) and float(x) >= 0.0


def validate_policy(policy: Any) -> bool:
    if not isinstance(policy, dict):
        return False
    for k in ("maxAgeSeconds", "accuracyFloor", "maxLatencyMs", "maxSizeBytes", "minImprovement"):
        if not is_finite_number(policy.get(k)):
            return False
    if policy.get("maxAgeSeconds") < 0 or policy.get("maxLatencyMs") < 0 or policy.get("maxSizeBytes") < 0:
        return False
    if not (0.0 <= float(policy.get("accuracyFloor")) <= 1.0):
        return False
    if not (0.0 <= float(policy.get("minImprovement")) <= 1.0):
        return False
    for digest_key in ("datasetDigest", "schemaDigest"):
        d = policy.get(digest_key)
        if not isinstance(d, str) or not d:
            return False
    slices = policy.get("requiredSlices", {})
    if slices is None:
        slices = {}
    if not isinstance(slices, dict):
        return False
    for v in slices.values():
        if not in_unit_interval(v):
            return False
    return True


# ---------------------------------------------------------------------------
# Unified per-version gate accumulation
# ---------------------------------------------------------------------------

def gate_codes_for_version(
    v: dict,
    vid: Any,
    is_duplicate: bool,
    policy: dict,
    policy_valid: bool,
    as_of: Optional[datetime],
) -> set[str]:
    """Accumulate every applicable gate code for one version entry.
    Canonical/duplicate faults never suppress the other checks -- everything
    that can be checked, is checked, and codes are unioned together."""
    codes: set[str] = set()

    if not is_canonical_version_id(vid):
        codes.add("INVALID_VERSION")
    elif is_duplicate:
        codes.add("DUPLICATE_VERSION")

    if not policy_valid:
        codes.add("INVALID_POLICY")
        return codes  # nothing else is checkable without a valid policy

    evaluation = v.get("evaluation") if isinstance(v, dict) else None
    if not isinstance(evaluation, dict):
        codes.add("MISSING_EVALUATION")
        return codes

    accuracy = evaluation.get("accuracy")
    latency = evaluation.get("latencyMs")
    size = evaluation.get("sizeBytes")
    slices = evaluation.get("slices")

    accuracy_finite = is_finite_number(accuracy)
    latency_finite = is_finite_number(latency)
    size_finite = is_finite_number(size)
    if not (accuracy_finite and latency_finite and size_finite):
        codes.add("NON_FINITE")

    accuracy_in_range = accuracy_finite and in_unit_interval(accuracy)
    latency_in_range = latency_finite and is_nonneg_finite(latency)
    size_in_range = size_finite and is_nonneg_finite(size)
    if accuracy_finite and not accuracy_in_range:
        codes.add("METRIC_RANGE")
    if latency_finite and not latency_in_range:
        codes.add("METRIC_RANGE")
    if size_finite and not size_in_range:
        codes.add("METRIC_RANGE")

    created_at = parse_timestamp(evaluation.get("createdAt"))
    if as_of is None:
        codes.add("INVALID_TIMESTAMP")
    elif created_at is None:
        codes.add("INVALID_TIMESTAMP")
    else:
        max_age = policy.get("maxAgeSeconds")
        window_start = as_of - timedelta(seconds=float(max_age))
        if created_at > as_of:
            codes.add("FUTURE_EVALUATION")
        elif created_at < window_start:
            codes.add("STALE_EVALUATION")

    if evaluation.get("artifactDigest") != (v.get("artifactDigest") if isinstance(v, dict) else None):
        codes.add("ARTIFACT_MISMATCH")
    if evaluation.get("datasetDigest") != policy.get("datasetDigest"):
        codes.add("DATASET_MISMATCH")
    if evaluation.get("schemaDigest") != policy.get("schemaDigest"):
        codes.add("SCHEMA_MISMATCH")

    required_slices = policy.get("requiredSlices") or {}
    eval_slices = slices if isinstance(slices, dict) else {}
    for name, floor in required_slices.items():
        if name not in eval_slices:
            codes.add(f"MISSING_SLICE:{name}")
            continue
        val = eval_slices[name]
        val_finite = is_finite_number(val)
        if not in_unit_interval(val):
            codes.add(f"SLICE_RANGE:{name}")
        if val_finite and is_finite_number(floor) and float(val) < float(floor):
            codes.add(f"SLICE_FLOOR:{name}")

    accuracy_floor = policy.get("accuracyFloor")
    if accuracy_finite and is_finite_number(accuracy_floor) and float(accuracy) < float(accuracy_floor):
        codes.add("ACCURACY_FLOOR")

    max_latency = policy.get("maxLatencyMs")
    if latency_finite and is_finite_number(max_latency) and float(latency) > float(max_latency):
        codes.add("LATENCY_LIMIT")

    max_size = policy.get("maxSizeBytes")
    if size_finite and is_finite_number(max_size) and float(size) > float(max_size):
        codes.add("SIZE_LIMIT")

    return codes


def version_sort_key(v: dict):
    ev = v["evaluation"]
    return (
        -float(ev["accuracy"]),
        float(ev["latencyMs"]),
        float(ev["sizeBytes"]),
        int(v["version"]),
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/promote")
async def promote(request: Request):
    raw_body = await request.body()
    logger.info("REQUEST /promote: %s", raw_body.decode("utf-8", errors="replace"))

    try:
        body = json.loads(raw_body)
    except Exception:
        logger.info("RESPONSE /promote: 400 INVALID_INPUT (bad JSON)")
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    if not isinstance(body, dict):
        logger.info("RESPONSE /promote: 400 INVALID_INPUT (body not an object)")
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    policy = body.get("policy")
    versions = body.get("versions")
    champion_version = body.get("championVersion")
    as_of_raw = body.get("asOf")

    if policy is None or not isinstance(versions, list) or not isinstance(champion_version, str):
        logger.info("RESPONSE /promote: 400 INVALID_INPUT (missing policy/versions/championVersion)")
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    as_of = parse_timestamp(as_of_raw)
    policy_valid = validate_policy(policy)

    # Count occurrences of each version-id string to detect duplicates.
    id_counts: dict[str, int] = {}
    for v in versions:
        if isinstance(v, dict) and isinstance(v.get("version"), str):
            vid = v["version"]
            id_counts[vid] = id_counts.get(vid, 0) + 1

    failed_gates: dict[str, list[str]] = {}
    eligible_by_id: dict[str, dict] = {}

    for v in versions:
        if not isinstance(v, dict):
            continue
        vid = v.get("version")
        if not isinstance(vid, str):
            continue

        is_duplicate = id_counts.get(vid, 0) > 1
        codes = gate_codes_for_version(v, vid, is_duplicate, policy, policy_valid, as_of)

        existing = set(failed_gates.get(vid, []))
        failed_gates[vid] = sorted(existing | codes)

        # A version is eligible only if it is canonical+unique AND has
        # zero gate codes overall, AND (in case of a duplicate id) only
        # the first clean occurrence is kept as the candidate for ranking.
        if not codes and vid not in eligible_by_id:
            eligible_by_id[vid] = v

    eligible_versions_ranked = sorted(eligible_by_id.values(), key=version_sort_key)
    eligible_ids_ranked = [v["version"] for v in eligible_versions_ranked]

    champion = eligible_by_id.get(champion_version)

    if champion is None:
        resp = {
            "action": "block",
            "championVersion": champion_version,
            "selectedVersion": None,
            "eligibleVersions": eligible_ids_ranked,
            "failedGates": failed_gates,
            "aliasMutation": None,
            "evidence": None,
        }
        logger.info("RESPONSE /promote: %s", json.dumps(resp))
        return JSONResponse(status_code=200, content=resp)

    challenger = eligible_versions_ranked[0]

    champion_acc = float(champion["evaluation"]["accuracy"])
    challenger_acc = float(challenger["evaluation"]["accuracy"])
    improvement = round(challenger_acc - champion_acc, 12)
    min_improvement = float(policy["minImprovement"])

    if challenger["version"] != champion_version and improvement >= min_improvement:
        selected = challenger
        action = "promote"
        alias_mutation = {"alias": "champion", "version": selected["version"]}
    else:
        selected = champion
        action = "retain"
        alias_mutation = None

    resp = {
        "action": action,
        "championVersion": champion_version,
        "selectedVersion": selected["version"],
        "eligibleVersions": eligible_ids_ranked,
        "failedGates": failed_gates,
        "aliasMutation": alias_mutation,
        "evidence": selected["evaluation"],
    }
    logger.info("RESPONSE /promote: %s", json.dumps(resp))
    return JSONResponse(status_code=200, content=resp)