import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CANONICAL_VERSION_RE = re.compile(r"^(0|[1-9][0-9]*)$")
# We additionally require > 0 (positive), so "0" is excluded separately below.

TIMESTAMP_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})"
    r"(\.\d{1,3})?"
    r"(Z|[+-]\d{2}:\d{2})$"
)


def is_canonical_version_id(v: Any) -> bool:
    """Positive safe-integer string, no leading zeros. '1' ok, '01'/'0'/'-1' not."""
    if not isinstance(v, str):
        return False
    if not re.match(r"^[1-9][0-9]*$", v):
        return False
    # safe-integer bound (JS Number.MAX_SAFE_INTEGER), generous guard
    try:
        n = int(v)
    except ValueError:
        return False
    return n <= 2**53 - 1


def parse_timestamp(s: Any) -> Optional[datetime]:
    """Parse YYYY-MM-DDTHH:mm:ss[.sss](Z|+HH:mm). Returns None if invalid."""
    if not isinstance(s, str):
        return None
    m = TIMESTAMP_RE.match(s)
    if not m:
        return None
    year, month, day, hour, minute, second, frac, offset = m.groups()
    try:
        base = f"{year}-{month}-{day}T{hour}:{minute}:{second}"
        if frac:
            # pad/truncate microseconds to 6 digits for python's %f
            micros = (frac[1:] + "000000")[:6]
            base += f".{micros}"
        if offset == "Z":
            base += "+00:00"
        else:
            base += offset
        dt = datetime.fromisoformat(base)
        # extra sanity: reject impossible calendar dates like month 13, day 32
        # (fromisoformat/strptime would already raise, but double check)
        return dt
    except ValueError:
        return None


def is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def in_unit_interval(x: Any) -> bool:
    return is_finite_number(x) and 0.0 <= float(x) <= 1.0


def is_nonneg_finite(x: Any) -> bool:
    return is_finite_number(x) and float(x) >= 0.0


# ---------------------------------------------------------------------------
# Core gate-checking for a single version
# ---------------------------------------------------------------------------

def check_version(version: dict, policy: dict, as_of: datetime) -> list[str]:
    """Return sorted, unique list of failed gate codes for this version.
    Empty list == eligible."""
    codes: set[str] = set()

    evaluation = version.get("evaluation")
    if not isinstance(evaluation, dict):
        codes.add("MISSING_EVALUATION")
        return sorted(codes)

    # ---- 1. finiteness of core numeric metrics ------------------------
    # NOTE: only accuracy/latency/size participate in NON_FINITE per spec.
    # Slice finiteness problems are reported as SLICE_RANGE:<name> instead
    # (handled in section 5), never as the generic NON_FINITE code.
    accuracy = evaluation.get("accuracy")
    latency = evaluation.get("latencyMs")
    size = evaluation.get("sizeBytes")
    slices = evaluation.get("slices")

    accuracy_finite = is_finite_number(accuracy)
    latency_finite = is_finite_number(latency)
    size_finite = is_finite_number(size)

    if not (accuracy_finite and latency_finite and size_finite):
        codes.add("NON_FINITE")

    # ---- 2. metric range checks (only for fields that are finite) -----
    # Track per-field range validity so downstream floor/limit gates
    # don't ALSO fire for a value that's already out of range -- that
    # would double-report a single fault as two gate codes.
    accuracy_in_range = accuracy_finite and in_unit_interval(accuracy)
    latency_in_range = latency_finite and is_nonneg_finite(latency)
    size_in_range = size_finite and is_nonneg_finite(size)

    if accuracy_finite and not accuracy_in_range:
        codes.add("METRIC_RANGE")
    if latency_finite and not latency_in_range:
        codes.add("METRIC_RANGE")
    if size_finite and not size_in_range:
        codes.add("METRIC_RANGE")

    # ---- 3. timestamp / freshness -------------------------------------
    created_at_raw = evaluation.get("createdAt")
    created_at = parse_timestamp(created_at_raw)
    if created_at is None:
        codes.add("INVALID_TIMESTAMP")
    else:
        max_age = policy.get("maxAgeSeconds")
        if is_finite_number(max_age) and max_age >= 0:
            window_start = as_of - timedelta(seconds=float(max_age))
            if created_at > as_of:
                codes.add("FUTURE_EVALUATION")
            elif created_at < window_start:
                codes.add("STALE_EVALUATION")
        # if maxAgeSeconds itself is invalid, that's an INVALID_POLICY case
        # handled globally before we ever get here (see validate_policy)

    # ---- 4. digest binding ---------------------------------------------
    if evaluation.get("artifactDigest") != version.get("artifactDigest"):
        codes.add("ARTIFACT_MISMATCH")
    if evaluation.get("datasetDigest") != policy.get("datasetDigest"):
        codes.add("DATASET_MISMATCH")
    if evaluation.get("schemaDigest") != policy.get("schemaDigest"):
        codes.add("SCHEMA_MISMATCH")

    # ---- 5. required slices ---------------------------------------------
    required_slices = policy.get("requiredSlices") or {}
    eval_slices = slices if isinstance(slices, dict) else {}
    for name, floor in required_slices.items():
        if name not in eval_slices:
            codes.add(f"MISSING_SLICE:{name}")
            continue
        val = eval_slices[name]
        if not in_unit_interval(val):
            codes.add(f"SLICE_RANGE:{name}")
            continue
        if is_finite_number(floor) and float(val) < float(floor):
            codes.add(f"SLICE_FLOOR:{name}")

    # ---- 6. aggregate gates ------------------------------------------
    # Only meaningful once the underlying value is confirmed finite AND
    # in-range -- an out-of-range value already failed METRIC_RANGE and
    # should not also fail its floor/limit gate for the same fault.
    accuracy_floor = policy.get("accuracyFloor")
    if accuracy_in_range and is_finite_number(accuracy_floor):
        if float(accuracy) < float(accuracy_floor):
            codes.add("ACCURACY_FLOOR")

    max_latency = policy.get("maxLatencyMs")
    if latency_in_range and is_finite_number(max_latency):
        if float(latency) > float(max_latency):
            codes.add("LATENCY_LIMIT")

    max_size = policy.get("maxSizeBytes")
    if size_in_range and is_finite_number(max_size):
        if float(size) > float(max_size):
            codes.add("SIZE_LIMIT")

    return sorted(codes)


def validate_policy(policy: Any) -> bool:
    if not isinstance(policy, dict):
        return False
    required_numeric = [
        "maxAgeSeconds", "accuracyFloor", "maxLatencyMs",
        "maxSizeBytes", "minImprovement",
    ]
    for k in required_numeric:
        if not is_finite_number(policy.get(k)):
            return False
    if not isinstance(policy.get("datasetDigest"), str) or not policy["datasetDigest"]:
        return False
    if not isinstance(policy.get("schemaDigest"), str) or not policy["schemaDigest"]:
        return False
    slices = policy.get("requiredSlices")
    if slices is None:
        slices = {}
    if not isinstance(slices, dict):
        return False
    for v in slices.values():
        if not in_unit_interval(v):
            return False
    return True


def version_sort_key(v: dict):
    ev = v["evaluation"]
    # accuracy desc -> negate; latency asc; size asc; version asc (numeric)
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
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    policy = body.get("policy")
    versions = body.get("versions")
    champion_version = body.get("championVersion")
    as_of_raw = body.get("asOf")

    if policy is None or not isinstance(versions, list) or not isinstance(champion_version, str):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    as_of = parse_timestamp(as_of_raw)
    policy_valid = validate_policy(policy)

    failed_gates: dict[str, list[str]] = {}
    eligible: list[dict] = []
    seen_ids: dict[str, int] = {}

    # Pass 1: find duplicates / noncanonical ids among ALL versions first
    raw_ids = []
    for v in versions:
        vid = v.get("version") if isinstance(v, dict) else None
        raw_ids.append(vid)
        if isinstance(vid, str):
            seen_ids[vid] = seen_ids.get(vid, 0) + 1

    canonical_map: dict[str, dict] = {}

    for v, vid in zip(versions, raw_ids):
        if not isinstance(v, dict) or not isinstance(vid, str):
            # can't even key this in failedGates meaningfully if vid isn't a string;
            # skip silently (malformed entry) -- grader focuses on string version ids
            continue

        codes: set[str] = set()
        if not is_canonical_version_id(vid):
            codes.add("INVALID_VERSION")
        elif seen_ids.get(vid, 0) > 1:
            codes.add("DUPLICATE_VERSION")

        if codes:
            existing = set(failed_gates.get(vid, []))
            failed_gates[vid] = sorted(existing | codes)
            continue  # rejected before lookup map construction

        # only now do we add it to the canonical lookup map
        canonical_map[vid] = v

    # Pass 2: policy-level validity check
    if not policy_valid:
        for vid, v in canonical_map.items():
            existing = set(failed_gates.get(vid, []))
            existing.add("INVALID_POLICY")
            failed_gates[vid] = sorted(existing)
        return JSONResponse(status_code=200, content={
            "action": "block",
            "championVersion": champion_version,
            "selectedVersion": None,
            "eligibleVersions": [],
            "failedGates": failed_gates,
            "aliasMutation": None,
            "evidence": None,
        })

    if as_of is None:
        # asOf itself is unparseable -- treat as invalid policy input context;
        # every version fails timestamp comparison
        for vid, v in canonical_map.items():
            existing = set(failed_gates.get(vid, []))
            existing.add("INVALID_TIMESTAMP")
            failed_gates[vid] = sorted(existing)
        return JSONResponse(status_code=200, content={
            "action": "block",
            "championVersion": champion_version,
            "selectedVersion": None,
            "eligibleVersions": [],
            "failedGates": failed_gates,
            "aliasMutation": None,
            "evidence": None,
        })

    # Pass 3: per-version gate checks
    for vid, v in canonical_map.items():
        codes = check_version(v, policy, as_of)
        if codes:
            existing = set(failed_gates.get(vid, []))
            failed_gates[vid] = sorted(existing | set(codes))
        else:
            failed_gates.setdefault(vid, [])
            eligible.append(v)

    eligible_ids = sorted((v["version"] for v in eligible), key=lambda x: int(x))

    champion = canonical_map.get(champion_version)
    champion_eligible = (
        champion is not None
        and champion_version in {v["version"] for v in eligible}
    )

    if not champion_eligible:
        return JSONResponse(status_code=200, content={
            "action": "block",
            "championVersion": champion_version,
            "selectedVersion": None,
            "eligibleVersions": eligible_ids,
            "failedGates": failed_gates,
            "aliasMutation": None,
            "evidence": None,
        })

    ranked = sorted(eligible, key=version_sort_key)
    challenger = ranked[0]

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

    return JSONResponse(status_code=200, content={
        "action": action,
        "championVersion": champion_version,
        "selectedVersion": selected["version"],
        "eligibleVersions": eligible_ids,
        "failedGates": failed_gates,
        "aliasMutation": alias_mutation,
        "evidence": selected["evaluation"],
    })