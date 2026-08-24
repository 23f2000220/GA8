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
Q5 - Quantize & Admit: FREEZE phase implementation.

"""


import hashlib
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse



# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
FREEZE_REQUESTS: dict[str, dict] = {}
FREEZE_RESPONSES: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# FREEZE phase helpers
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
 
            files = c.get("files")
            if not isinstance(files, dict) or len(files) == 0:
                errors.append(f"candidates[{i}] ({name}).files missing/empty: {files!r}")
            elif any(not isinstance(v, str) for v in files.values()):
                errors.append(f"candidates[{i}] ({name}).files has non-string values")
 
    return errors
 
 
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
 
 
# ---------------------------------------------------------------------------
# Route changes: replace your existing validation calls with these
# ---------------------------------------------------------------------------
"""
if phase == "freeze":
    errors = get_freeze_validation_errors(body)
    if errors:
        logger.info("FREEZE INVALID_INPUT for freezeId=%r: %s", body.get("freezeId"), errors)
        logger.info("Full body was: %s", body)
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    # ... rest unchanged (replay/conflict check should stay BEFORE this,
    #     exactly as in your current code - only the validate_freeze_input(body)
    #     call gets swapped for get_freeze_validation_errors(body))
 
elif phase == "select":
    errors = get_select_validation_errors(body)
    if errors:
        logger.info("SELECT INVALID_INPUT for freezeId=%r: %s", body.get("freezeId"), errors)
        logger.info("Full body was: %s", body)
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    # ... rest unchanged
"""


def build_freeze_response(body: dict) -> dict:
    freeze_id = body.get("freezeId")
    results = []

    for candidate in body["candidates"]:
        status_str, reason_codes = get_candidate_status(candidate, body)
        sorted_records, total_bytes, package_digest = build_inventory_digest(candidate["files"])

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
# SELECT phase helpers
# ---------------------------------------------------------------------------
def compute_aggregate_accuracy(rows: list, candidate_name: str) -> float | None:
    matches = 0
    row_count = 0
    for row in rows:
        row_count += 1
        prediction = row["predictions"].get(candidate_name)
        if prediction not in (0, 1):
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




def validate_policy(policy: dict, candidate_names: set) -> bool:
    max_bytes = policy.get("maxBytes")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0:
        return False

    agg_floor = policy.get("aggregateFloor")
    if not isinstance(agg_floor, (int, float)) or isinstance(agg_floor, bool):
        return False
    if not (0 <= agg_floor <= 1):
        return False

    required_slices = policy.get("requiredSlices", {})
    if not isinstance(required_slices, dict):
        return False
    for v in required_slices.values():
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0 <= v <= 1):
            return False

    max_latency = policy.get("maxLatencyMs")
    if not isinstance(max_latency, (int, float)) or isinstance(max_latency, bool) or max_latency < 0:
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


def evaluate_candidate(frozen_candidate: dict, policy: dict, rows: list, latencies: dict) -> dict:
    name = frozen_candidate["name"]
    codes = []

    is_frozen = frozen_candidate["status"] == "frozen"
    if not is_frozen:
        codes.append("NOT_FROZEN")

    total_bytes = frozen_candidate["totalBytes"]

    aggregate = compute_aggregate_accuracy(rows, name)
    predictions_valid = aggregate is not None

    slices = None
    if predictions_valid:
        slices = compute_slice_accuracies(rows, name, list(policy.get("requiredSlices", {}).keys()))
    else:
        codes.append("INVALID_PREDICTIONS")

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

    latency_ms = latencies.get(name)
    if latency_ms is None or latency_ms > policy["maxLatencyMs"]:
        codes.append("LATENCY_LIMIT")

    codes = sorted(set(codes))
    admitted = is_frozen and len(codes) == 0

    return {
        "name": name,
        "aggregate": aggregate,
        "slices": slices if slices is not None else {},
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

    results = []
    for candidate in submitted_candidates:
        name = candidate.get("name")

        if not lineage_ok:
            results.append({
                "name": name, "aggregate": None, "slices": {},
                "totalBytes": None, "latencyMs": None,
                "admitted": False, "reasonCodes": ["INVALID_LINEAGE"],
            })
            continue

        if not policy_ok:
            results.append({
                "name": name, "aggregate": None, "slices": {},
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
@app.post("/q5/quantize")
async def quantize(request: Request):
    body = await request.json()
    phase = body.get("phase") if isinstance(body, dict) else None

    if phase == "freeze":
        logger.info("FREEZE request received: %s", body)
        freeze_id = body.get("freezeId") if isinstance(body, dict) else None

        if isinstance(freeze_id, str) and freeze_id in FREEZE_REQUESTS:
            if FREEZE_REQUESTS[freeze_id] == body:
                return JSONResponse(FREEZE_RESPONSES[freeze_id])
            else:
                return JSONResponse({"error": "FREEZE_ID_CONFLICT"}, status_code=409)

        if not get_freeze_validation_errors(body):
            return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

        response = build_freeze_response(body)
        FREEZE_REQUESTS[freeze_id] = body
        FREEZE_RESPONSES[freeze_id] = response
        return JSONResponse(response)

    elif phase == "select":
        if not get_select_validation_errors(body):
            return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

        freeze_id = body.get("freezeId")
        frozen_response = FREEZE_RESPONSES.get(freeze_id)
        response = build_select_response(body, freeze_id, frozen_response)
        return JSONResponse(response)

    else:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)