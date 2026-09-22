"""Deterministic numerical verification using safe Python arithmetic for AuditRAG Phase 9."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

from auditrag.verification.models import Claim


def verify_arithmetic_calculation(claim: Claim) -> Optional[Dict[str, Any]]:
    """Deterministically check if arithmetic calculations in a claim hold in Python."""
    nums = claim.extracted_numbers
    if len(nums) < 2:
        return None

    text = claim.statement.lower()

    # 1. Check Difference / Increase / Decrease
    # Pattern: "increased by X ... from Y (to Z)" or "increased X from Y to Z"
    if any(k in text for k in ("increase", "decrease", "difference", "variance")):
        # If 3 numbers present: check if val_new - val_old == diff
        if len(nums) >= 3:
            # Sort or check permutations: a + b = c or c - b = a
            for i in range(len(nums)):
                for j in range(len(nums)):
                    if i == j:
                        continue
                    for k in range(len(nums)):
                        if k in (i, j):
                            continue
                        a, b, c = nums[i], nums[j], nums[k]
                        if math.isclose(a + b, c, rel_tol=1e-2, abs_tol=0.05):
                            return {
                                "verified": True,
                                "operation": "variance_check",
                                "expression": f"{c} - {b} = {a}",
                                "computed_value": round(c - b, 4),
                                "claimed_value": a,
                                "explanation": f"Arithmetic verified: {c} - {b} equals stated variance {a}.",
                            }
        # If 2 numbers: check if percentage or ratio
        elif len(nums) == 2:
            pass

    # 2. Check Sum / Total
    if any(k in text for k in ("total", "sum", "combined", "aggregate")):
        if len(nums) >= 3:
            # Check if largest number equals sum of other numbers
            sorted_nums = sorted(nums)
            total = sorted_nums[-1]
            parts = sorted_nums[:-1]
            computed_sum = sum(parts)
            if math.isclose(computed_sum, total, rel_tol=1e-2, abs_tol=0.05):
                parts_str = " + ".join(str(p) for p in parts)
                return {
                    "verified": True,
                    "operation": "sum_check",
                    "expression": f"{parts_str} = {total}",
                    "computed_value": round(computed_sum, 4),
                    "claimed_value": total,
                    "explanation": f"Arithmetic verified: sum of {parts_str} equals stated total {total}.",
                }

    # 3. Check Percentage Growth: ((new - old) / old) * 100
    if "%" in claim.statement or "percent" in text:
        if len(nums) >= 3:
            # Check if one of the numbers is percentage growth
            for i, pct in enumerate(nums):
                remaining = [n for idx, n in enumerate(nums) if idx != i]
                if len(remaining) >= 2:
                    old_val, new_val = remaining[0], remaining[1]
                    if old_val != 0:
                        computed_pct = ((new_val - old_val) / abs(old_val)) * 100.0
                        if math.isclose(computed_pct, pct, rel_tol=1e-2, abs_tol=0.2):
                            return {
                                "verified": True,
                                "operation": "percentage_growth_check",
                                "expression": f"(({new_val} - {old_val}) / {old_val}) * 100% = {pct}%",
                                "computed_value": round(computed_pct, 2),
                                "claimed_value": pct,
                                "explanation": f"Percentage calculation verified: growth from {old_val} to {new_val} is {pct}%.",
                            }

    return None
