"""LLM prompt templates from Appendix H."""

PLAN_GENERATION_PROMPT = """Decompose a clinical risk assessment into evidence-gathering steps over a medical knowledge graph.
Each step traces a specific clinical pathway (comorbidity cascade, drug interaction, or disease progression) from the patient's conditions toward the target disease.
Allocate budget fractions that sum to at most 1.

Inputs:
Patient codes: {patient_codes}
History: {history}
Target disease: {target_disease}
Horizon: {horizon_days} days
Anchor set: {anchors}
Budget: {budget}

Output JSON:
{{"steps": [{{"hypothesis": "...", "query": "...", "expected_evidence": "...", "budget_fraction": 0.5}}]}}
"""

EDGE_SELECTION_PROMPT = """Select which candidate edges to retain as evidence for this plan step.
Prioritize edges that directly support the hypothesis and satisfy the support and non-redundancy constraints.

Hypothesis: {hypothesis}
Expected evidence: {expected_evidence}
Candidates: {candidates}

Output JSON:
{{"selected_edge_ids": ["..."]}}
"""

VERIFICATION_PROMPT = """Verify whether the retrieved evidence supports the plan-step hypothesis.
A step passes if the edges are clinically consistent with the hypothesis and the average quality score is sufficient.
A step fails if evidence is absent, contradictory, or too weak.

Hypothesis: {hypothesis}
Expected evidence: {expected_evidence}
Retrieved evidence: {retrieved_evidence}

Output JSON:
{{"result": "PASS" or "FAIL", "reason": "..."}}
"""

PLAN_REVISION_PROMPT = """Regenerate the plan from the failed step onward using a different clinical pathway that avoids the failure mode.
Previously verified steps are frozen. The revised plan must fit within the remaining budget.

Failed step: {failed_step}
Failure reason: {failure_reason}
Verified context: {verified_context}
Remaining budget: {remaining_budget}

Output JSON:
{{"steps": [{{"hypothesis": "...", "query": "...", "expected_evidence": "...", "budget_fraction": 0.5}}]}}
"""

CITATION_JUDGE_PROMPT = """Determine whether a knowledge graph edge cited by a clinical reasoning system is relevant to the patient's risk for the target disease.
An edge is relevant if it describes a mechanism, risk factor, or pathway connecting the patient's conditions to the target.

Patient codes: {patient_codes}
Target disease: {target_disease}
Cited edge: {cited_edge}
Support score: {support_score}

Output: RELEVANT or NOT_RELEVANT with one-sentence reason.
"""
