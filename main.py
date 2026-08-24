import json
import logging
import math
import re
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logger = logging.getLogger("promote")

app = FastAPI()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


##############################
#-------------q3--------------
##############################

CANONICAL_ID_RE = re.compile(r"^[1-9][0-9]*$")

TIMESTAMP_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})"
    r"(\.\d{1,3})?"
    r"(Z|[+-]\d{2}:\d{2})$"
)


def version_key(raw: Any) -> Optional[str]:
    """Turn any JSON-scalar version id into a stable string key for the
    failedGates/eligibility maps -- even if it's the wrong type (e.g. a
    JSON number) and therefore doomed to fail canonical-id validation.
    A malformed version must still be REPORTED, not silently dropped."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, bool):
        return json.dumps(raw)
    if isinstance(raw, (int, float)):
        return json.dumps(raw)
    if raw is None:
        return json.dumps(raw)
    return None  # unkeyable (list/dict) -- nothing sane to report under


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

    # Count occurrences of each version-id key (including malformed/non-string
    # ids, so duplicates of a bad id are still detected) to flag duplicates.
    id_counts: dict[str, int] = {}
    for v in versions:
        if isinstance(v, dict):
            key = version_key(v.get("version"))
            if key is not None:
                id_counts[key] = id_counts.get(key, 0) + 1

    failed_gates: dict[str, list[str]] = {}
    eligible_by_id: dict[str, dict] = {}

    for v in versions:
        if not isinstance(v, dict):
            continue
        raw_vid = v.get("version")
        vid = version_key(raw_vid)
        if vid is None:
            continue

        is_duplicate = id_counts.get(vid, 0) > 1
        codes = gate_codes_for_version(v, raw_vid, is_duplicate, policy, policy_valid, as_of)

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


################################
#------------q4-----------------
################################

# main.py


INTERVENTION_ORDER = ["prompt_only", "retrieval", "lora", "qlora"]
VALID_ROLES = {"system", "user", "assistant"}
# EXPECTED_ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors"]
REQUIRED_CHECKPOINT_KEYS = {"model", "optimizer", "scheduler", "step", "rng", "dataPosition"}


def is_hex(s: str, length: int) -> bool:
    if not isinstance(s, str) or len(s) != length:
        return False
    return bool(re.fullmatch(r"[0-9a-f]{" + str(length) + "}", s))


def is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def is_positive_int(x) -> bool:
    return isinstance(x, int) and x > 0


def validate_tokens(tokens):
    if not isinstance(tokens, list) or len(tokens) == 0:
        return False
    for t in tokens:
        if not isinstance(t, dict):
            return False
        if set(t.keys()) != {"id", "role", "padding", "text"}:
            return False
        if not isinstance(t["id"], int) or t["id"] < 0:
            return False
        if t["role"] not in VALID_ROLES:
            return False
        if not isinstance(t["padding"], bool):
            return False
        if not isinstance(t["text"], str):
            return False
    return True


def compute_labels(tokens):
    labels = []
    for t in tokens:
        if t["role"] == "assistant" and not t["padding"]:
            labels.append(t["id"])
        else:
            labels.append(-100)
    return labels


def validate_parameters(parameters, allowed_targets):
    if not isinstance(parameters, list) or len(parameters) == 0:
        return False
    if not isinstance(allowed_targets, list) or len(allowed_targets) == 0:
        return False

    # allowedTargets must be unique strings
    if len(allowed_targets) != len(set(allowed_targets)):
        return False
    for t in allowed_targets:
        if not isinstance(t, str):
            return False

    names = set()
    for p in parameters:
        if not isinstance(p, dict):
            return False
        if set(p.keys()) != {"name", "target", "numel"}:
            return False
        name = p["name"]
        target = p["target"]
        numel = p["numel"]
        if not isinstance(name, str) or not isinstance(target, str):
            return False
        if not isinstance(numel, int) or isinstance(numel, bool) or numel <= 0:
            return False
        if name in names:
            return False  # names must be unique
        names.add(name)

    return True



def compute_trainable_params(parameters, allowed_targets):
    allowed_set = set(allowed_targets)
    trainable = []
    total_numel = 0
    for p in parameters:
        if p["target"] in allowed_set:
            name = p["name"]
            if name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight"):
                trainable.append(name)
                total_numel += p["numel"]
    trainable = sorted(trainable)
    return trainable, total_numel


def is_unique_nonempty_str_list(lst):
    if not isinstance(lst, list) or len(lst) == 0:
        return False
    if not all(isinstance(x, str) for x in lst):
        return False
    return len(lst) == len(set(lst))


def handle_choose(body: dict):
    policy = body.get("policy")
    candidates = body.get("candidates")

    if not isinstance(policy, dict) or not isinstance(candidates, list):
        return None

    required_policy_keys = [
        "minQuality",
        "freshnessRequired",
        "maxLatencyMs",
        "maxMemoryMb",
        "maxLabeledExamples",
        "maxTotalCost",
        "horizonRequests",
    ]
    for k in required_policy_keys:
        if k not in policy:
            return None

    min_quality = float(policy["minQuality"])
    freshness_required = bool(policy["freshnessRequired"])
    max_latency = float(policy["maxLatencyMs"])
    max_memory = float(policy["maxMemoryMb"])
    max_labeled = int(policy["maxLabeledExamples"])
    max_total_cost = float(policy["maxTotalCost"])
    horizon = float(policy["horizonRequests"])

    cand_map = {}
    for c in candidates:
        if not isinstance(c, dict) or "name" not in c:
            return None
        cand_map[c["name"]] = c

    for name in INTERVENTION_ORDER:
        if name not in cand_map:
            return None

    eligible = []
    total_costs = {}
    reason_codes = {}

    for name in INTERVENTION_ORDER:
        c = cand_map[name]
        try:
            quality = float(c["quality"])
            latency = float(c["latencyMs"])
            memory = float(c["memoryMb"])
            labeled = int(c["labeledExamples"])
            one_time = float(c["oneTimeCost"])
            recurring = float(c["recurringCost"])
            available = bool(c["available"])
            freshness = bool(c["freshness"])
        except Exception:
            return None

        codes = []

        if not available:
            codes.append("UNAVAILABLE")
        if quality < min_quality:
            codes.append("QUALITY_FLOOR")
        if freshness_required and not freshness:
            codes.append("FRESHNESS_REQUIRED")
        if latency > max_latency:
            codes.append("LATENCY_LIMIT")
        if memory > max_memory:
            codes.append("MEMORY_LIMIT")
        if labeled > max_labeled:
            codes.append("DATA_LIMIT")

        total_cost = one_time + horizon * recurring
        total_cost = round(total_cost, 12)
        total_costs[name] = total_cost

        if total_cost > max_total_cost:
            codes.append("COST_LIMIT")

        codes = sorted(set(codes))
        reason_codes[name] = codes

        if len(codes) == 0:
            eligible.append(name)

    selected = eligible[0] if eligible else None

    return {
        "selected": selected,
        "eligible": eligible,
        "totalCosts": total_costs,
        "reasonCodes": reason_codes,
    }


def handle_repair(body: dict):
    reason_codes = []

    # Tokens & labels
    tokens = body.get("tokens")
    if not validate_tokens(tokens):
        if isinstance(tokens, list) and all(isinstance(t, dict) for t in tokens):
            labels = [-100] * len(tokens)
        else:
            labels = []
        reason_codes.append("INVALID_TOKEN")
    else:
        labels = compute_labels(tokens)

    # Template applications
    template_applications = body.get("templateApplications")
    template_pass = template_applications == 1
    if not template_pass:
        reason_codes.append("CHAT_TEMPLATE_COUNT")

    # Parameters & trainable
    parameters = body.get("parameters")
    allowed_targets = body.get("allowedTargets")
    artifact_files = body.get("artifactFiles")

    peft_config_pass = True
    reason_codes = []

    if not validate_parameters(parameters, allowed_targets):
        peft_config_pass = False
        reason_codes.append("INVALID_PARAMETER")
        trainable_params = []
        trainable_count = 0
    else:
        trainable_params, trainable_count = compute_trainable_params(parameters, allowed_targets)
        if len(trainable_params) == 0:
            peft_config_pass = False
            reason_codes.append("INVALID_PARAMETER")
            trainable_params = []
            trainable_count = 0
        else:
            peft_config_pass = True

    # Adapter files
    EXPECTED_ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors"]

    if not isinstance(artifact_files, list):
        peft_config_pass = False
        reason_codes.append("ADAPTER_FILE_SET")
        adapter_files = []
    else:
        if sorted(artifact_files) != sorted(EXPECTED_ADAPTER_FILES):
            peft_config_pass = False
            reason_codes.append("ADAPTER_FILE_SET")
            adapter_files = sorted(EXPECTED_ADAPTER_FILES)
        else:
            adapter_files = sorted(artifact_files)

    # Inference mode & adapter files
    inference_mode = body.get("inferenceMode")
    dropout_active = body.get("dropoutActiveDuringEval")
    artifact_files = body.get("artifactFiles")

    if inference_mode is not False:
        peft_config_pass = False
        reason_codes.append("INFERENCE_MODE")

    if not isinstance(artifact_files, list):
        peft_config_pass = False
        reason_codes.append("ADAPTER_FILE_SET")
        adapter_files = []
    else:
        if sorted(artifact_files) != sorted(EXPECTED_ADAPTER_FILES):
            peft_config_pass = False
            reason_codes.append("ADAPTER_FILE_SET")
            adapter_files = sorted(EXPECTED_ADAPTER_FILES)
        else:
            adapter_files = sorted(artifact_files)

    # Checkpoint
    checkpoint = body.get("checkpoint")
    if not isinstance(checkpoint, dict) or not REQUIRED_CHECKPOINT_KEYS.issubset(checkpoint.keys()):
        checkpoint_complete = False
        reason_codes.append("INCOMPLETE_CHECKPOINT")
    else:
        checkpoint_complete = True

    # Lineage
    base_rev = body.get("baseRevision")
    dataset_digest = body.get("datasetDigest")
    code_digest = body.get("codeDigest")
    config_digest = body.get("configDigest")
    expected_digests = body.get("expectedDigests") or {}

    lineage_pass = True
    if not is_hex(base_rev, 40):
        lineage_pass = False
        reason_codes.append("MUTABLE_BASE_REVISION")
    if not (
        is_hex(dataset_digest, 64)
        and is_hex(code_digest, 64)
        and is_hex(config_digest, 64)
    ):
        lineage_pass = False
        reason_codes.append("LINEAGE_MISMATCH")
    
    else:
        # Check against expectedDigests if present
        if expected_digests.get("datasetDigest") != dataset_digest:
            lineage_pass = False
            reason_codes.append("LINEAGE_MISMATCH")
        if expected_digests.get("codeDigest") != code_digest:
            lineage_pass = False
            reason_codes.append("LINEAGE_MISMATCH")
        if expected_digests.get("configDigest") != config_digest:
            lineage_pass = False
            reason_codes.append("LINEAGE_MISMATCH")


    # Batch factors
    micro_batch = body.get("microBatch")
    grad_accum = body.get("gradientAccumulation")
    replicas = body.get("replicas")
    expected_eff_batch = body.get("expectedEffectiveBatch")

    batch_ok = (
        is_positive_int(micro_batch)
        and is_positive_int(grad_accum)
        and is_positive_int(replicas)
        and is_positive_int(expected_eff_batch)
        and (micro_batch * grad_accum * replicas == expected_eff_batch)
    )
    if not batch_ok:
        reason_codes.append("EFFECTIVE_BATCH_MISMATCH")

    # Eval isolation & determinism
    train_ids = body.get("trainRowIds")
    eval_ids = body.get("evalRowIds")

    eval_isolated = True
    evaluation_deterministic = True

    if not (is_unique_nonempty_str_list(train_ids) and is_unique_nonempty_str_list(eval_ids)):
        eval_isolated = False
        reason_codes.append("EVAL_LEAKAGE")
    elif set(train_ids) & set(eval_ids):
        eval_isolated = False
        reason_codes.append("EVAL_LEAKAGE")

    if dropout_active is not False:
        evaluation_deterministic = False
        reason_codes.append("EVAL_DROPOUT_ACTIVE")

    # Resume
    uw = body.get("uninterruptedWeights")
    rw = body.get("resumedWeights")
    tol = body.get("resumeTolerance")

    resume_pass = True
    if (
        not isinstance(uw, list)
        or not isinstance(rw, list)
        or len(uw) == 0
        or len(rw) == 0
        or len(uw) != len(rw)
        or not all(is_finite_number(x) for x in uw)
        or not all(is_finite_number(x) for x in rw)
        or not is_finite_number(tol)
        or tol < 0
    ):
        resume_pass = False
        reason_codes.append("RESUME_DIVERGENCE")
    else:
        max_diff = max(abs(a - b) for a, b in zip(uw, rw))
        if max_diff > tol:
            resume_pass = False
            reason_codes.append("RESUME_DIVERGENCE")

    reason_codes = sorted(set(reason_codes))

    return {
        "labels": labels,
        "templatePass": template_pass,
        "trainableParams": trainable_params,
        "trainableCount": trainable_count,
        "peftConfigPass": peft_config_pass,
        "adapterFiles": adapter_files,
        "checkpointComplete": checkpoint_complete,
        "lineagePass": lineage_pass,
        "evalIsolated": eval_isolated,
        "evaluationDeterministic": evaluation_deterministic,
        "resumePass": resume_pass,
        "reasonCodes": reason_codes,
    }


@app.post("/q4/adapt")

async def adapt_endpoint(request: Request):
    try:
        raw_body = await request.body()
        body = await request.json()  # <-- add await here
        if isinstance(body, dict):
            logger.info("REQUEST /adapt: %s", json.dumps(body, ensure_ascii=False))
        else:
            logger.info("REQUEST /adapt (non-dict): %s", raw_body.decode("utf-8", errors="replace"))
    except Exception as e:
        logger.info("REQUEST /adapt (parse error): %s", e)
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    if not isinstance(body, dict):
        logger.info("REQUEST /adapt (invalid top-level): %s", body)
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    operation = body.get("operation")

    if operation == "choose":
        result = handle_choose(body)
        if result is None:
            logger.info("RESPONSE /adapt (choose): 400 INVALID_INPUT")
            return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
        logger.info("RESPONSE /adapt (choose): %s", json.dumps(result, ensure_ascii=False))
        return JSONResponse(status_code=200, content=result)

    elif operation == "repair":
        result = handle_repair(body)
        if result is None:
            logger.info("RESPONSE /adapt (repair): 400 INVALID_INPUT")
            return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
        logger.info("RESPONSE /adapt (repair): %s", json.dumps(result, ensure_ascii=False))
        return JSONResponse(status_code=200, content=result)

    else:
        logger.info("RESPONSE /adapt (unknown operation): 400 INVALID_INPUT")
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
    
###############################
#------------q5----------------
###############################
"""
Q5 - Quantize & Admit: complete /quantize endpoint (v2).

Changes from v1:
- Boolean predictions (True/False) are now correctly rejected, not silently
  treated as 1/0.
- When predictions are invalid, `slices` returns {slice_name: None, ...}
  for every required slice, not an empty {}.
- Policy numeric fields (maxBytes, aggregateFloor, requiredSlices values,
  maxLatencyMs) are now checked for actual finiteness, not just >= 0
  (infinity/NaN previously slipped through).
- maxBytes is checked against the JS/JSON "safe integer" upper bound.
- Individual per-candidate latency values (not just the maxLatencyMs
  ceiling) are now validated as finite non-negative numbers.
- totalBytes/packageDigest are recomputed from each candidate's own
  inventory rather than read directly off the submitted dict, and
  INVALID_MANIFEST fires if that recomputation doesn't match the
  candidate's own claim (defense-in-depth; in practice rarely reachable
  since strict lineage equality already catches most tampering - flagging
  that honestly rather than pretending otherwise).
- Global lineage/policy failures now fill `slices` with {slice_name: None}
  per required slice instead of {}.

Merge into your existing app: copy the storage dicts, every function, and
the single @app.post("/quantize") route into your Q3 app file.
"""

import hashlib
import json
import math
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()  # remove if merging into your existing app instance

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
FREEZE_REQUESTS: dict[str, dict] = {}
FREEZE_RESPONSES: dict[str, dict] = {}

SAFE_INT_MAX = 2**53 - 1


# ---------------------------------------------------------------------------
# Numeric validation helpers
# ---------------------------------------------------------------------------
def is_finite_number(x, allow_bool=False) -> bool:
    if isinstance(x, bool) and not allow_bool:
        return False
    if not isinstance(x, (int, float)):
        return False
    return math.isfinite(x)


def is_safe_nonneg_integer(x) -> bool:
    if isinstance(x, bool):
        return False
    if not isinstance(x, int):
        return False
    return 0 <= x <= SAFE_INT_MAX


# ---------------------------------------------------------------------------
# FREEZE phase
# ---------------------------------------------------------------------------
def build_inventory_digest(files: dict) -> tuple[list, int, str]:
    records = []
    for name, content in files.items():
        content_bytes = content.encode("utf-8")
        byte_length = len(content_bytes)
        sha256_hash = hashlib.sha256(content_bytes).hexdigest()
        records.append({"name": name, "bytes": byte_length, "sha256": sha256_hash})
    sorted_records = sorted(records, key=lambda r: r["name"])
    total_bytes = sum(r["bytes"] for r in sorted_records)
    package = json.dumps(sorted_records, separators=(",", ":"))
    package_digest = hashlib.sha256(package.encode("utf-8")).hexdigest()
    return sorted_records, total_bytes, package_digest


def get_candidate_status(candidate: dict, request: dict) -> tuple[str, list[str]]:
    if candidate.get("unsupportedReason"):
        if candidate.get("unsupportedReason") in request.get("allowedUnsupportedReasons", []):
            return ("unsupported", [])
        else:
            return ("invalid", ["UNALLOWED_UNSUPPORTED_REASON"])
    if candidate.get("loadable") == True:
        if candidate["calibrationDigest"] == request["calibrationDigest"]:
            if candidate["tokenizerDigest"] == request["tokenizerDigest"]:
                return ("frozen", [])
            return ("invalid", ["TOKENIZER_MISMATCH"])
        return ("invalid", ["CALIBRATION_MISMATCH"])
    return ("invalid", ["NOT_LOADABLE"])


def is_valid_files_dict(files) -> bool:
    if not isinstance(files, dict) or len(files) == 0:
        return False
    for k, v in files.items():
        if not isinstance(k, str) or k == "" or not isinstance(v, str):
            return False
    return True


def get_freeze_validation_errors(body: dict) -> list[str]:
    errors = []
    if not isinstance(body, dict):
        return ["body is not a JSON object"]

    freeze_id = body.get("freezeId")
    if not isinstance(freeze_id, str) or not (1 <= len(freeze_id) <= 128):
        errors.append(f"freezeId invalid: {freeze_id!r}")

    for digest_key in ("calibrationDigest", "tokenizerDigest"):
        val = body.get(digest_key)
        if not isinstance(val, str) or val == "":
            errors.append(f"{digest_key} missing/empty: {val!r}")

    allowed_reasons = body.get("allowedUnsupportedReasons")
    if not isinstance(allowed_reasons, list):
        errors.append(f"allowedUnsupportedReasons missing or not a list: {allowed_reasons!r}")
    else:
        if any(not isinstance(r, str) or r == "" for r in allowed_reasons):
            errors.append("allowedUnsupportedReasons contains empty/non-string entries")
        if len(set(allowed_reasons)) != len(allowed_reasons):
            errors.append("allowedUnsupportedReasons has duplicates")

    candidates = body.get("candidates")
    if not isinstance(candidates, list) or len(candidates) == 0:
        errors.append(f"candidates missing/empty: {candidates!r}")
    else:
        seen_names = set()
        for i, c in enumerate(candidates):
            if not isinstance(c, dict):
                errors.append(f"candidates[{i}] is not an object")
                continue
            name = c.get("name")
            if not isinstance(name, str) or name == "":
                errors.append(f"candidates[{i}].name missing/empty: {name!r}")
            elif name in seen_names:
                errors.append(f"candidates[{i}].name duplicate: {name!r}")
            else:
                seen_names.add(name)
            # files validity is per-candidate, handled in build_freeze_response,
            # NOT a request-level rejection.

    return errors


def build_freeze_response(body: dict) -> dict:
    freeze_id = body.get("freezeId")
    results = []

    for candidate in body["candidates"]:
        files = candidate.get("files")

        if not is_valid_files_dict(files):
            results.append({
                "name": candidate.get("name"),
                "status": "invalid",
                "inventory": [],
                "totalBytes": None,
                "packageDigest": None,
                "reasonCodes": ["INVALID_INPUT"],
            })
            continue

        status_str, reason_codes = get_candidate_status(candidate, body)
        sorted_records, total_bytes, package_digest = build_inventory_digest(files)

        results.append({
            "name": candidate["name"],
            "status": status_str,
            "inventory": sorted_records,
            "totalBytes": total_bytes,
            "packageDigest": package_digest,
            "reasonCodes": reason_codes,
        })

    results_sorted = sorted(results, key=lambda r: r["name"])
    return {"freezeId": freeze_id, "candidates": results_sorted}


# ---------------------------------------------------------------------------
# SELECT phase
# ---------------------------------------------------------------------------
def compute_aggregate_accuracy(rows: list, candidate_name: str) -> float | None:
    matches = 0
    row_count = 0
    for row in rows:
        row_count += 1
        prediction = row["predictions"].get(candidate_name)
        if isinstance(prediction, bool) or prediction not in (0, 1):
            return None
        if prediction == row["label"]:
            matches += 1
    if row_count == 0:
        return None
    return round(matches / row_count, 12)


def compute_slice_accuracies(rows: list, candidate_name: str, slice_names: list) -> dict:
    result = {}
    for slice_name in slice_names:
        slice_rows = [r for r in rows if r.get("slice") == slice_name]
        if len(slice_rows) == 0:
            result[slice_name] = None
        else:
            result[slice_name] = compute_aggregate_accuracy(slice_rows, candidate_name)
    return result


def validate_select_input(body: dict) -> bool:
    if not isinstance(body, dict):
        return False
    if not isinstance(body.get("candidates"), list):
        return False
    if not isinstance(body.get("rows"), list):
        return False
    if not isinstance(body.get("policy"), dict):
        return False
    return True


def get_select_validation_errors(body: dict) -> list[str]:
    errors = []
    if not isinstance(body, dict):
        return ["body is not a JSON object"]
    if not isinstance(body.get("candidates"), list):
        errors.append(f"candidates missing or not a list: {body.get('candidates')!r}")
    if not isinstance(body.get("rows"), list):
        errors.append(f"rows missing or not a list: {body.get('rows')!r}")
    if not isinstance(body.get("policy"), dict):
        errors.append(f"policy missing or not an object: {body.get('policy')!r}")
    return errors


def validate_policy(policy: dict, candidate_names: set) -> bool:
    if not is_safe_nonneg_integer(policy.get("maxBytes")):
        return False

    agg_floor = policy.get("aggregateFloor")
    if not is_finite_number(agg_floor) or not (0 <= agg_floor <= 1):
        return False

    required_slices = policy.get("requiredSlices", {})
    if not isinstance(required_slices, dict):
        return False
    for v in required_slices.values():
        if not is_finite_number(v) or not (0 <= v <= 1):
            return False

    if not is_finite_number(policy.get("maxLatencyMs")) or policy["maxLatencyMs"] < 0:
        return False

    candidate_order = policy.get("candidateOrder")
    if not isinstance(candidate_order, list):
        return False
    if len(set(candidate_order)) != len(candidate_order):
        return False
    if set(candidate_order) != candidate_names:
        return False

    return True


def check_lineage(submitted_candidates: list, stored_freeze_response: dict | None) -> bool:
    if stored_freeze_response is None:
        return False
    return submitted_candidates == stored_freeze_response["candidates"]


def recompute_manifest(candidate: dict) -> tuple[int | None, str | None, bool]:
    inventory = candidate.get("inventory")
    if not isinstance(inventory, list):
        return None, None, False

    sorted_inv = sorted(inventory, key=lambda r: r["name"])
    total_bytes = sum(r["bytes"] for r in sorted_inv)
    package = json.dumps(sorted_inv, separators=(",", ":"))
    package_digest = hashlib.sha256(package.encode("utf-8")).hexdigest()

    matches_claim = (
        total_bytes == candidate.get("totalBytes")
        and package_digest == candidate.get("packageDigest")
    )
    return total_bytes, package_digest, matches_claim


def evaluate_candidate(frozen_candidate: dict, policy: dict, rows: list, latencies: dict) -> dict:
    name = frozen_candidate["name"]
    codes = []

    is_frozen = frozen_candidate["status"] == "frozen"
    if not is_frozen:
        codes.append("NOT_FROZEN")

    recomputed_bytes, recomputed_digest, manifest_ok = recompute_manifest(frozen_candidate)
    if not manifest_ok:
        codes.append("INVALID_MANIFEST")
    total_bytes = recomputed_bytes

    required_slice_names = list(policy.get("requiredSlices", {}).keys())

    aggregate = compute_aggregate_accuracy(rows, name)
    predictions_valid = aggregate is not None

    if predictions_valid:
        slices = compute_slice_accuracies(rows, name, required_slice_names)
    else:
        codes.append("INVALID_PREDICTIONS")
        slices = {slice_name: None for slice_name in required_slice_names}

    if predictions_valid:
        if aggregate < policy["aggregateFloor"]:
            codes.append("AGGREGATE_FLOOR")
        for slice_name, floor in policy.get("requiredSlices", {}).items():
            slice_acc = slices.get(slice_name)
            if slice_acc is None:
                codes.append(f"MISSING_SLICE:{slice_name}")
            elif slice_acc < floor:
                codes.append(f"SLICE_FLOOR:{slice_name}")

    if total_bytes is None or total_bytes > policy["maxBytes"]:
        codes.append("SIZE_LIMIT")

    raw_latency = latencies.get(name)
    if is_finite_number(raw_latency) and raw_latency >= 0:
        latency_ms = raw_latency
    else:
        latency_ms = None
    if latency_ms is None or latency_ms > policy["maxLatencyMs"]:
        codes.append("LATENCY_LIMIT")

    codes = sorted(set(codes))
    admitted = is_frozen and manifest_ok and len(codes) == 0

    return {
        "name": name,
        "aggregate": aggregate,
        "slices": slices,
        "totalBytes": total_bytes,
        "latencyMs": latency_ms,
        "admitted": admitted,
        "reasonCodes": codes,
    }


def pick_winner(results: list, candidate_order: list) -> str | None:
    admitted = [r for r in results if r["admitted"]]
    if not admitted:
        return None
    def sort_key(r):
        return (r["totalBytes"], r["latencyMs"], candidate_order.index(r["name"]))
    admitted.sort(key=sort_key)
    return admitted[0]["name"]


def build_select_response(body: dict, freeze_id: str, frozen_response: dict | None) -> dict:
    submitted_candidates = body["candidates"]
    rows = body["rows"]
    policy = body["policy"]
    latencies = body.get("latencies", {})

    candidate_names = {c.get("name") for c in submitted_candidates}
    lineage_ok = check_lineage(submitted_candidates, frozen_response)
    policy_ok = validate_policy(policy, candidate_names) if lineage_ok else False

    required_slice_names = (
        list(policy.get("requiredSlices", {}).keys())
        if isinstance(policy.get("requiredSlices"), dict) else []
    )

    results = []
    for candidate in submitted_candidates:
        name = candidate.get("name")

        if not lineage_ok:
            results.append({
                "name": name, "aggregate": None,
                "slices": {s: None for s in required_slice_names},
                "totalBytes": None, "latencyMs": None,
                "admitted": False, "reasonCodes": ["INVALID_LINEAGE"],
            })
            continue

        if not policy_ok:
            results.append({
                "name": name, "aggregate": None,
                "slices": {s: None for s in required_slice_names},
                "totalBytes": None, "latencyMs": None,
                "admitted": False, "reasonCodes": ["INVALID_POLICY"],
            })
            continue

        results.append(evaluate_candidate(candidate, policy, rows, latencies))

    candidate_order = policy.get("candidateOrder") if policy_ok else None
    if candidate_order:
        results_sorted = sorted(results, key=lambda r: candidate_order.index(r["name"]))
        selected = pick_winner(results, candidate_order)
    else:
        results_sorted = sorted(results, key=lambda r: r["name"])
        selected = None

    package_manifest = None
    if selected is not None:
        for c in frozen_response["candidates"]:
            if c["name"] == selected:
                package_manifest = c
                break

    return {
        "freezeId": freeze_id,
        "selected": selected,
        "results": results_sorted,
        "packageManifest": package_manifest,
    }


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------
@app.post("/quantize")
async def quantize(request: Request):
    body = await request.json()
    phase = body.get("phase") if isinstance(body, dict) else None

    if phase == "freeze":
        freeze_id = body.get("freezeId") if isinstance(body, dict) else None

        if isinstance(freeze_id, str) and freeze_id in FREEZE_REQUESTS:
            if FREEZE_REQUESTS[freeze_id] == body:
                return JSONResponse(FREEZE_RESPONSES[freeze_id])
            else:
                return JSONResponse({"error": "FREEZE_ID_CONFLICT"}, status_code=409)

        if get_freeze_validation_errors(body):
            return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

        response = build_freeze_response(body)
        FREEZE_REQUESTS[freeze_id] = body
        FREEZE_RESPONSES[freeze_id] = response
        return JSONResponse(response)

    elif phase == "select":
        if get_select_validation_errors(body):
            return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

        freeze_id = body.get("freezeId")
        frozen_response = FREEZE_RESPONSES.get(freeze_id)
        response = build_select_response(body, freeze_id, frozen_response)
        return JSONResponse(response)

    else:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)


############################
#-----------q6--------------
#############################

# ----------------------------
# Constants
# ----------------------------
NODES = ["verify_data", "prepare", "train", "evaluate", "register", "publish"]

REQUIRED_INPUT_KEYS = [
    "generation",
    "checksum",
    "canonicalData",
    "prepareCode",
    "prepareConfig",
    "trainCode",
    "trainConfig",
    "runtime",
    "evaluateCode",
    "evaluateConfig",
    "schemaDigest",
    "publishConfig",
]

VALID_STATUSES = {"started", "succeeded", "retryable_failed", "terminal_failed"}

EVENT_FIELDS = [
    "eventId",
    "revision",
    "node",
    "attempt",
    "status",
    "key",
    "artifactDigest",
    "receiptId",
]

# ----------------------------
# In-memory session store
# ----------------------------
sessions: dict[str, dict[str, Any]] = {}


# ----------------------------
# Helpers
# ----------------------------
def compact_json(obj: Any) -> str:
    """Compact JSON: no spaces, preserve order, UTF-8."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(value_list: list[Any]) -> str:
    """Lowercase SHA-256 over UTF-8 compact JSON array."""
    s = compact_json(value_list)
    return hashlib.sha256(s.encode("utf-8")).hexdigest().lower()


def init_node_state() -> dict[str, Any]:
    return {
        "key": None,  # current cache key (str or None)
        "artifact_digest": None,  # first success artifact for this key
        "status": None,  # None | "started" | "succeeded" | "retryable_failed" | "terminal_failed"
        "attempt": None,  # int or None
        "success_event_id": None,  # event ID that first made this node succeed (for this key)
        "start_event_id": None,  # event ID that started the current attempt
        "terminal_event_id": None,  # event ID that caused terminal failure
    }


def get_or_create_session(session_id: str, revision: int, inputs: dict[str, Any]):
    """
    Get or create session state.
    Returns (session_dict, error_code_or_None).
    Handles revision logic and input snapshot.
    """
    logger.debug("get_or_create_session: session=%s revision=%s", session_id, revision)

    if session_id not in sessions:
        logger.info("Creating new session: %s", session_id)
        sessions[session_id] = {
            "current_revision": revision,
            "inputs_snapshot": inputs,  # full inputs dict for this revision
            "nodes": {node: init_node_state() for node in NODES},
            "event_ids": {},  # eventId -> canonical JSON string
        }
        return sessions[session_id], None

    sess = sessions[session_id]
    current_rev = sess["current_revision"]

    if revision < current_rev:
        logger.debug("Older revision request: session=%s req_rev=%s curr_rev=%s", session_id, revision, current_rev)
        # Older revision: we still allow the request, but events from this revision will be ignored later.
        # Do NOT change inputs_snapshot or state.
        return sess, None

    if revision > current_rev:
        logger.info("New revision: session=%s old_rev=%s new_rev=%s", session_id, current_rev, revision)
        sess["current_revision"] = revision
        sess["inputs_snapshot"] = inputs

        # Clear attempt/terminal state, keep succeeded cache entries
        for node in NODES:
            n = sess["nodes"][node]
            if n["status"] != "succeeded":
                logger.debug("Clearing non-succeeded state for node=%s", node)
                n["status"] = None
                n["attempt"] = None
                n["start_event_id"] = None
                n["terminal_event_id"] = None
            # key, artifact_digest, success_event_id remain if succeeded

        return sess, None

    # Same revision: check inputs equality (including extra metadata)
    if compact_json(inputs) != compact_json(sess["inputs_snapshot"]):
        logger.warning("REVISION_CONFLICT: session=%s revision=%s inputs changed", session_id, revision)
        return None, "REVISION_CONFLICT"

    logger.debug("Same revision and inputs: session=%s revision=%s", session_id, revision)
    return sess, None


def validate_request(body: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """
    Validate top-level request structure.
    Returns (parsed_data, error_code_or_None).
    parsed_data = {session, revision, inputs, events}
    """
    logger.debug("Validating request body")

    # Basic structure
    if not isinstance(body, dict):
        logger.warning("INVALID_REQUEST: body is not a dict")
        return None, "INVALID_REQUEST"

    session = body.get("session")
    revision = body.get("revision")
    inputs = body.get("inputs")
    events = body.get("events")

    # session: non-empty string
    if not isinstance(session, str) or session == "":
        logger.warning("INVALID_REQUEST: session missing or not a non-empty string")
        return None, "INVALID_REQUEST"

    # revision: positive safe integer
    if not isinstance(revision, int) or revision <= 0:
        logger.warning("INVALID_REQUEST: revision must be a positive integer")
        return None, "INVALID_REQUEST"

    # inputs: dict
    if not isinstance(inputs, dict):
        logger.warning("INVALID_REQUEST: inputs must be a dict")
        return None, "INVALID_REQUEST"

    # Check required input keys: non-empty strings
    for k in REQUIRED_INPUT_KEYS:
        v = inputs.get(k)
        if not isinstance(v, str) or v == "":
            logger.warning("INVALID_REQUEST: input '%s' missing or not a non-empty string", k)
            return None, "INVALID_REQUEST"

    # events: list
    if not isinstance(events, list):
        logger.warning("INVALID_REQUEST: events must be a list")
        return None, "INVALID_REQUEST"

    parsed = {
        "session": session,
        "revision": revision,
        "inputs": inputs,
        "events": events,
    }
    logger.debug("Request validation passed")
    return parsed, None


def compute_cache_key(node: str, inputs: dict[str, Any], nodes_state: dict[str, dict[str, Any]]) -> str:
    """
    Compute cache key for a node given inputs and upstream artifact digests.
    """
    if node == "verify_data":
        arr = [inputs["generation"], inputs["checksum"]]
    elif node == "prepare":
        arr = [inputs["canonicalData"], inputs["prepareCode"], inputs["prepareConfig"]]
    elif node == "train":
        prepare_art = nodes_state["prepare"]["artifact_digest"]
        arr = [prepare_art, inputs["trainCode"], inputs["trainConfig"], inputs["runtime"]]
    elif node == "evaluate":
        train_art = nodes_state["train"]["artifact_digest"]
        arr = [train_art, inputs["canonicalData"], inputs["evaluateCode"], inputs["evaluateConfig"]]
    elif node == "register":
        evaluate_art = nodes_state["evaluate"]["artifact_digest"]
        arr = [evaluate_art, inputs["schemaDigest"]]
    elif node == "publish":
        register_art = nodes_state["register"]["artifact_digest"]
        arr = [register_art, inputs["publishConfig"]]
    else:
        raise ValueError(f"Unknown node: {node}")

    return sha256_hex(arr)


def validate_event_structure(event: dict[str, Any]) -> str | None:
    """
    Validate event structure and types.
    Returns error_code or None.
    """
    if not isinstance(event, dict):
        return "INVALID_EVENT"

    # Exactly 8 fields
    if set(event.keys()) != set(EVENT_FIELDS):
        logger.warning("INVALID_EVENT: event fields mismatch: %s", event.keys())
        return "INVALID_EVENT"

    # eventId: non-empty string
    if not isinstance(event["eventId"], str) or event["eventId"] == "":
        return "INVALID_EVENT"

    # revision: positive int
    if not isinstance(event["revision"], int) or event["revision"] <= 0:
        return "INVALID_EVENT"

    # node: one of NODES
    if event["node"] not in NODES:
        return "INVALID_EVENT"

    # attempt: positive int
    if not isinstance(event["attempt"], int) or event["attempt"] <= 0:
        return "INVALID_EVENT"

    # status: one of VALID_STATUSES
    if event["status"] not in VALID_STATUSES:
        return "INVALID_EVENT"

    # key: string (allow empty? treat as string)
    if not isinstance(event["key"], str):
        return "INVALID_EVENT"

    # artifactDigest: string or null
    ad = event["artifactDigest"]
    if ad is not None and not isinstance(ad, str):
        return "INVALID_EVENT"

    # receiptId: string or null
    rid = event["receiptId"]
    if rid is not None and not isinstance(rid, str):
        return "INVALID_EVENT"

    # Artifact/receipt rules
    status = event["status"]
    node = event["node"]
    key = event["key"]

    if status == "succeeded":
        # artifactDigest must be non-empty string
        if not isinstance(ad, str) or ad == "":
            logger.warning("INVALID_EVENT: succeeded event must have non-empty artifactDigest")
            return "INVALID_EVENT"

        # receipt rules
        if node in ("register", "publish"):
            expected_receipt = f"receipt:{node}:{key}"
            if rid != expected_receipt:
                logger.warning(
                    "INVALID_EVENT: receiptId mismatch for node=%s key=%s expected=%s got=%s",
                    node, key, expected_receipt, rid,
                )
                return "INVALID_EVENT"
        else:
            if rid is not None:
                logger.warning("INVALID_EVENT: non-register/publish succeeded event must have null receiptId")
                return "INVALID_EVENT"
    else:
        # non-succeeded: artifactDigest and receiptId must be null
        if ad is not None:
            logger.warning("INVALID_EVENT: non-succeeded event must have null artifactDigest")
            return "INVALID_EVENT"
        if rid is not None:
            logger.warning("INVALID_EVENT: non-succeeded event must have null receiptId")
            return "INVALID_EVENT"

    return None


def process_events(
    sess: dict[str, Any],
    inputs: dict[str, Any],
    events: list[dict[str, Any]],
    request_revision: int,
) -> tuple[list[str], list[str], str | None]:
    """
    Process a batch of events.
    Returns (accepted_event_ids, ignored_event_ids, error_code_or_None).
    If error_code is not None, no state changes should be applied.
    """
    logger.debug("Processing %d events for session=%s revision=%s", len(events), sess.get("current_revision"), request_revision)

    accepted_ids = []
    ignored_ids = []

    nodes_state = sess["nodes"]
    current_rev = sess["current_revision"]
    event_ids_store = sess["event_ids"]

    # We'll apply changes tentatively, then commit if no error.
    # To keep it simple, we'll apply directly but rollback on error by restoring a deep copy.
    import copy
    sess_snapshot = copy.deepcopy(sess)

    def rollback():
        logger.warning("Rolling back session state due to error")
        sess.clear()
        sess.update(sess_snapshot)

    for idx, event in enumerate(events):
        logger.debug("Processing event %d: %s", idx, event["eventId"])

        # Structure validation
        err = validate_event_structure(event)
        if err:
            logger.warning("Event invalid: %s", event["eventId"])
            rollback()
            return [], [], err

        ev_id = event["eventId"]
        ev_rev = event["revision"]
        ev_node = event["node"]
        ev_attempt = event["attempt"]
        ev_status = event["status"]
        ev_key = event["key"]
        ev_art = event["artifactDigest"]

        # Revision filter
        if ev_rev < current_rev:
            logger.debug("Ignoring event from older revision: %s", ev_id)
            ignored_ids.append(ev_id)
            continue

        if ev_rev > current_rev:
            # Should not normally happen; treat as ignore
            logger.debug("Ignoring event from future revision: %s", ev_id)
            ignored_ids.append(ev_id)
            continue

        # Compute expected key for this node
        expected_key = compute_cache_key(ev_node, inputs, nodes_state)
        if ev_key != expected_key:
            logger.debug("Ignoring event with wrong key: node=%s ev_key=%s expected=%s", ev_node, ev_key, expected_key)
            ignored_ids.append(ev_id)
            continue

        # Event ID uniqueness within session
        if ev_id in event_ids_store:
            existing_canonical = event_ids_store[ev_id]
            new_canonical = compact_json(event)
            if new_canonical != existing_canonical:
                logger.warning("EVENT_ID_CONFLICT: eventId=%s", ev_id)
                rollback()
                return [], [], "EVENT_ID_CONFLICT"
            else:
                logger.debug("Exact replay of event: %s", ev_id)
                ignored_ids.append(ev_id)
                continue
        else:
            # Store canonical JSON for this event ID
            event_ids_store[ev_id] = compact_json(event)

        # Apply transition rules
        node_state = nodes_state[ev_node]
        cur_status = node_state["status"]
        cur_attempt = node_state["attempt"]

        logger.debug(
            "Node=%s cur_status=%s cur_attempt=%s incoming_status=%s incoming_attempt=%s",
            ev_node, cur_status, cur_attempt, ev_status, ev_attempt,
        )

        def accept_event():
            # Update node state based on event
            if ev_status == "started":
                node_state["status"] = "started"
                node_state["attempt"] = ev_attempt
                node_state["start_event_id"] = ev_id
            elif ev_status == "succeeded":
                node_state["status"] = "succeeded"
                node_state["attempt"] = ev_attempt
                node_state["artifact_digest"] = ev_art
                node_state["success_event_id"] = ev_id
            elif ev_status == "retryable_failed":
                node_state["status"] = "retryable_failed"
                node_state["attempt"] = ev_attempt
            elif ev_status == "terminal_failed":
                node_state["status"] = "terminal_failed"
                node_state["attempt"] = ev_attempt
                node_state["terminal_event_id"] = ev_id

        # Transition logic per spec
        if cur_status is None:
            if ev_status == "started" and ev_attempt == 1:
                accept_event()
                accepted_ids.append(ev_id)
                continue
            else:
                logger.debug("Ignoring event: no prior state, not started(1)")
                # Remove the event ID we just stored? Spec: "Ignored events do not consume their IDs."
                # So we should NOT have stored it. We need to undo storing.
                del event_ids_store[ev_id]
                ignored_ids.append(ev_id)
                continue

        elif cur_status == "started":
            n = cur_attempt
            if ev_status in ("succeeded", "retryable_failed"):
                if ev_attempt == n:
                    accept_event()
                    accepted_ids.append(ev_id)
                    continue
                else:
                    logger.debug("Ignoring event: started(%s) but attempt mismatch %s", n, ev_attempt)
                    del event_ids_store[ev_id]
                    ignored_ids.append(ev_id)
                    continue
            else:
                logger.warning("STATUS_CONFLICT: started -> %s", ev_status)
                rollback()
                return [], [], "STATUS_CONFLICT"

        elif cur_status == "retryable_failed":
            n = cur_attempt
            if ev_status == "started" and ev_attempt == n + 1:
                accept_event()
                accepted_ids.append(ev_id)
                continue
            else:
                logger.warning("STATUS_CONFLICT: retryable_failed -> %s attempt=%s", ev_status, ev_attempt)
                rollback()
                return [], [], "STATUS_CONFLICT"

        elif cur_status == "succeeded":
            # Already cached for this key
            if ev_status == "succeeded":
                if ev_art != node_state["artifact_digest"]:
                    logger.warning("EVIDENCE_CONFLICT: node=%s different artifact", ev_node)
                    rollback()
                    return [], [], "EVIDENCE_CONFLICT"
                else:
                    # Same artifact: treat as replay? But event ID is new, so this is a conflict per spec:
                    # "succeeded/current cache | any other new event | STATUS_CONFLICT"
                    logger.warning("STATUS_CONFLICT: succeeded node with new success event")
                    rollback()
                    return [], [], "STATUS_CONFLICT"
            else:
                logger.warning("STATUS_CONFLICT: succeeded node with non-success event")
                rollback()
                return [], [], "STATUS_CONFLICT"

        elif cur_status == "terminal_failed":
            logger.warning("STATUS_CONFLICT: terminal_failed node with new event")
            rollback()
            return [], [], "STATUS_CONFLICT"

        else:
            logger.warning("STATUS_CONFLICT: unknown state")
            rollback()
            return [], [], "STATUS_CONFLICT"

    logger.debug("Events processed: accepted=%s ignored=%s", accepted_ids, ignored_ids)
    return accepted_ids, ignored_ids, None


def compute_node_response(
    node: str,
    inputs: dict[str, Any],
    nodes_state: dict[str, dict[str, Any]],
    upstream_blocked: tuple[bool, str] | None,
) -> dict[str, Any]:
    """
    Compute response for a single node.
    upstream_blocked = (is_blocked, reason) or None.
    reason in {"UPSTREAM_TERMINAL", "UPSTREAM_PENDING"}
    """
    n = nodes_state[node]
    key = compute_cache_key(node, inputs, nodes_state)

    dep_digests = {}
    # Populate dependency digests based on node
    if node == "verify_data":
        dep_digests = {
            "generation": inputs["generation"],
            "checksum": inputs["checksum"],
            "cacheKey": key,
        }
    elif node == "prepare":
        dep_digests = {
            "canonicalData": inputs["canonicalData"],
            "prepareCode": inputs["prepareCode"],
            "prepareConfig": inputs["prepareConfig"],
            "cacheKey": key,
        }
    elif node == "train":
        dep_digests = {
            "prepareArtifact": nodes_state["prepare"]["artifact_digest"],
            "trainCode": inputs["trainCode"],
            "trainConfig": inputs["trainConfig"],
            "runtime": inputs["runtime"],
            "cacheKey": key,
        }
    elif node == "evaluate":
        dep_digests = {
            "trainArtifact": nodes_state["train"]["artifact_digest"],
            "canonicalData": inputs["canonicalData"],
            "evaluateCode": inputs["evaluateCode"],
            "evaluateConfig": inputs["evaluateConfig"],
            "cacheKey": key,
        }
    elif node == "register":
        dep_digests = {
            "evaluateArtifact": nodes_state["evaluate"]["artifact_digest"],
            "schemaDigest": inputs["schemaDigest"],
            "cacheKey": key,
        }
    elif node == "publish":
        dep_digests = {
            "registerArtifact": nodes_state["register"]["artifact_digest"],
            "publishConfig": inputs["publishConfig"],
            "cacheKey": key,
        }

    triggering_event_ids = []

    # If upstream is blocked
    if upstream_blocked is not None:
        is_blocked, reason = upstream_blocked
        if is_blocked:
            return {
                "node": node,
                "action": "block",
                "reasonCodes": [reason],
                "dependencyDigests": dep_digests,
                "triggeringEventIds": [],
            }

    # Determine own state
    status = n["status"]

    if status == "succeeded":
        # Cached
        return {
            "node": node,
            "action": "reuse",
            "reasonCodes": ["CACHE_HIT"],
            "dependencyDigests": dep_digests,
            "triggeringEventIds": [n["success_event_id"]] if n["success_event_id"] else [],
        }

    if status == "started":
        return {
            "node": node,
            "action": "block",
            "reasonCodes": ["RUNNING"],
            "dependencyDigests": dep_digests,
            "triggeringEventIds": [n["start_event_id"]] if n["start_event_id"] else [],
        }

    if status == "terminal_failed":
        return {
            "node": node,
            "action": "block",
            "reasonCodes": ["TERMINAL_FAILURE"],
            "dependencyDigests": dep_digests,
            "triggeringEventIds": [n["terminal_event_id"]] if n["terminal_event_id"] else [],
        }

    if status == "retryable_failed":
        return {
            "node": node,
            "action": "rerun",
            "reasonCodes": ["RETRYABLE_FAILURE"],
            "dependencyDigests": dep_digests,
            "triggeringEventIds": [],
        }

    # No state / cache miss
    return {
        "node": node,
        "action": "rerun",
        "reasonCodes": ["CACHE_MISS"],
        "dependencyDigests": dep_digests,
        "triggeringEventIds": [],
    }


def build_response(
    revision: int,
    accepted_ids: list[str],
    ignored_ids: list[str],
    inputs: dict[str, Any],
    nodes_state: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    logger.debug("Building response for revision=%s", revision)

    nodes_response = []
    upstream_blocked = None  # (is_blocked, reason)

    for node in NODES:
        # Determine upstream blocking
        if node == "verify_data":
            ub = None
        else:
            # Find immediate upstream
            upstream_map = {
                "prepare": "verify_data",
                "train": "prepare",
                "evaluate": "train",
                "register": "evaluate",
                "publish": "register",
            }
            up = upstream_map[node]
            up_state = nodes_state[up]
            up_status = up_state["status"]

            if up_status == "terminal_failed":
                ub = (True, "UPSTREAM_TERMINAL")
            elif up_status in ("started", "retryable_failed", None):
                # If upstream is not succeeded, downstream is pending/blocked
                # But if upstream is None (no state), and not terminal, we treat as pending
                ub = (True, "UPSTREAM_PENDING")
            else:
                ub = None

        node_resp = compute_node_response(node, inputs, nodes_state, ub)
        nodes_response.append(node_resp)

        # Propagate blocking
        if node_resp["action"] == "block":
            reason = node_resp["reasonCodes"][0]
            if reason == "TERMINAL_FAILURE":
                upstream_blocked = (True, "UPSTREAM_TERMINAL")
            elif reason in ("RUNNING", "UPSTREAM_PENDING", "UPSTREAM_TERMINAL"):
                upstream_blocked = (True, "UPSTREAM_PENDING")

    response = {
        "revision": revision,
        "acceptedEventIds": accepted_ids,
        "ignoredEventIds": ignored_ids,
        "nodes": nodes_response,
    }
    logger.debug("Response built: %s", compact_json(response))
    return response


def handle_pipeline_request(body: dict[str, Any]) -> dict[str, Any]:
    logger.info("Handling /pipeline request")

    # Validate request
    parsed, err = validate_request(body)
    if err:
        logger.warning("Request validation failed: %s", err)
        return {"error": err}

    session_id = parsed["session"]
    revision = parsed["revision"]
    inputs = parsed["inputs"]
    events = parsed["events"]

    # Get/create session
    sess, err = get_or_create_session(session_id, revision, inputs)
    if err:
        logger.warning("Session/revision error: %s", err)
        return {"error": err}

    # Process events
    accepted_ids, ignored_ids, err = process_events(sess, inputs, events, revision)
    if err:
        logger.warning("Event processing error: %s", err)
        return {"error": err}

    # Build response
    response = build_response(revision, accepted_ids, ignored_ids, inputs, sess["nodes"])
    logger.info("Request handled successfully")
    return response


@app.post("/q6/pipeline")
async def pipeline_endpoint(request: Request):
    logger.info("Received POST /pipeline")
    try:
        body = await request.json()
    except Exception as e:
        logger.warning("Failed to parse JSON: %s", e)
        return JSONResponse(
            status_code=409,
            content={"error": "INVALID_REQUEST"},
        )

    result = handle_pipeline_request(body)
    if "error" in result:
        logger.warning("Returning 409 error: %s", result["error"])
        return JSONResponse(
            status_code=409,
            content={"error": result["error"]},
        )

    logger.debug("Returning success response")
    return JSONResponse(status_code=200, content=result)
