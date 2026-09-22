"""Default-off object-local Comparative Evidence Reversion for Fixed ASCD.

This module intentionally leaves ASCD's attention intervention and contrastive
logits unchanged.  It only makes a same-prefix choice between the Fixed-ASCD
top token and the unmodified Vanilla top token when *both* complete a canonical
CHAIR object.  A frozen OWLv2 support-difference policy can then fall back to
Vanilla when the ASCD candidate is not sufficiently better supported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ComparativeDecision:
    selected_token_id: int
    eligible_object_position: bool
    accepts_ascd: bool
    reason: str
    ascd_object: Optional[str]
    vanilla_object: Optional[str]
    ascd_support: Optional[float]
    vanilla_support: Optional[float]
    support_delta: Optional[float]


def decide_comparative_reversion(
    *,
    ascd_token_id: int,
    vanilla_token_id: int,
    ascd_object: Optional[str],
    vanilla_object: Optional[str],
    ascd_support: Optional[float],
    vanilla_support: Optional[float],
    threshold: Optional[float],
    observe_only: bool,
) -> ComparativeDecision:
    """Apply the frozen object-local decision without modifying any logits.

    The constrained policy is deliberately limited to a comparison in which
    both candidates finish known CHAIR object words.  Ambiguous, non-object,
    and unavailable-support positions remain Fixed ASCD.  Observe-only still
    computes the prospective decision but emits the exact Fixed-ASCD token.
    """
    if int(ascd_token_id) == int(vanilla_token_id):
        return ComparativeDecision(
            selected_token_id=int(ascd_token_id),
            eligible_object_position=False,
            accepts_ascd=True,
            reason="same_candidate",
            ascd_object=ascd_object,
            vanilla_object=vanilla_object,
            ascd_support=ascd_support,
            vanilla_support=vanilla_support,
            support_delta=None,
        )
    if ascd_object is None or vanilla_object is None:
        return ComparativeDecision(
            selected_token_id=int(ascd_token_id),
            eligible_object_position=False,
            accepts_ascd=True,
            reason="non_object_or_partial_object",
            ascd_object=ascd_object,
            vanilla_object=vanilla_object,
            ascd_support=ascd_support,
            vanilla_support=vanilla_support,
            support_delta=None,
        )
    if ascd_support is None or vanilla_support is None:
        return ComparativeDecision(
            selected_token_id=int(ascd_token_id),
            eligible_object_position=False,
            accepts_ascd=True,
            reason="missing_detector_support",
            ascd_object=ascd_object,
            vanilla_object=vanilla_object,
            ascd_support=ascd_support,
            vanilla_support=vanilla_support,
            support_delta=None,
        )
    if threshold is None:
        raise ValueError("Comparative Evidence Reversion requires a frozen threshold")

    delta = float(ascd_support) - float(vanilla_support)
    accepts_ascd = bool(delta + 1e-12 >= float(threshold))
    if observe_only:
        selected, reason = int(ascd_token_id), "observe_only"
    elif accepts_ascd:
        selected, reason = int(ascd_token_id), "accept_ascd"
    else:
        selected, reason = int(vanilla_token_id), "fallback_vanilla"
    return ComparativeDecision(
        selected_token_id=selected,
        eligible_object_position=True,
        accepts_ascd=accepts_ascd,
        reason=reason,
        ascd_object=ascd_object,
        vanilla_object=vanilla_object,
        ascd_support=float(ascd_support),
        vanilla_support=float(vanilla_support),
        support_delta=delta,
    )


def build_comparative_audit_record(
    *,
    image_id: int,
    step: int,
    ascd_token_id: int,
    vanilla_token_id: int,
    ascd_token: str,
    vanilla_token: str,
    decision: ComparativeDecision,
) -> Dict[str, object]:
    record: Dict[str, object] = {
        "image_id": int(image_id),
        "step": int(step),
        "ascd_token_id": int(ascd_token_id),
        "vanilla_token_id": int(vanilla_token_id),
        "ascd_token": str(ascd_token),
        "vanilla_token": str(vanilla_token),
    }
    record.update(asdict(decision))
    return record
