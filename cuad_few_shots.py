"""CUAD-derived few-shot examples that ground the draft generation prompt.

The Contract Understanding Atticus Dataset (CUAD; Hendrycks et al., 2021) is a
publicly available collection of 510 commercial contracts annotated for 41
clause categories.  We keep a small, hand-picked set of paraphrased exemplars
per contract type so the LangChain draft chain has concrete, on-brand language
to mimic - without having to ship the full ~150MB dataset or call out to
HuggingFace at runtime.

Each entry maps a CUAD clause category to a single representative clause body
written in the same style as the dataset.  The exemplars are paraphrased rather
than copied verbatim so the project can include them under a permissive
license, while still preserving the structural cues (defined terms, numbering,
cross-references) that make CUAD clauses useful as priors.

Reference:
    Hendrycks, Burns, Chen, Ball.  CUAD: An Expert-Annotated NLP Dataset for
    Legal Contract Review.  NeurIPS 2021 Datasets & Benchmarks Track.
    https://www.atticusprojectai.org/cuad
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Curated CUAD-style exemplars.  Each list is ordered roughly by how useful the
# clause is as a structural prior for the draft chain (most useful first).
# ---------------------------------------------------------------------------

_SERVICE_AGREEMENT_EXAMPLES: List[Dict[str, str]] = [
    {
        "cuad_category": "Governing Law",
        "clause": (
            "This Agreement shall be governed by and construed in accordance "
            "with the laws of the State of [JURISDICTION], without regard to "
            "its conflict-of-laws principles.  The parties consent to the "
            "exclusive jurisdiction of the state and federal courts located "
            "in [JURISDICTION] for any action arising out of or relating to "
            "this Agreement."
        ),
    },
    {
        "cuad_category": "Termination for Convenience",
        "clause": (
            "Either party may terminate this Agreement, with or without cause, "
            "upon thirty (30) days' prior written notice to the other party.  "
            "Upon any such termination, Client shall pay Provider for all "
            "Services performed and expenses incurred through the effective "
            "date of termination, and each party shall promptly return or "
            "destroy any Confidential Information of the other party."
        ),
    },
    {
        "cuad_category": "Cap on Liability",
        "clause": (
            "Notwithstanding anything to the contrary, the aggregate liability "
            "of each party under this Agreement, whether in contract, tort, or "
            "otherwise, shall not exceed the fees paid or payable by Client to "
            "Provider during the twelve (12) months preceding the event giving "
            "rise to the claim."
        ),
    },
    {
        "cuad_category": "IP Ownership Assignment",
        "clause": (
            "All right, title, and interest in and to any deliverables created "
            "by Provider specifically for Client under this Agreement "
            "(\"Work Product\"), including all intellectual property rights "
            "therein, shall, upon full payment of all fees due hereunder, vest "
            "in and be assigned to Client.  Provider retains ownership of any "
            "pre-existing tools, methodologies, or know-how used in the "
            "performance of the Services."
        ),
    },
    {
        "cuad_category": "Confidentiality",
        "clause": (
            "Each party shall protect the Confidential Information of the "
            "other with the same degree of care it uses to protect its own "
            "confidential information of like importance, but in no event less "
            "than a reasonable degree of care.  The receiving party shall not "
            "disclose Confidential Information to any third party except to "
            "its employees, contractors, and advisors who have a need to know "
            "and are bound by confidentiality obligations no less protective "
            "than those set forth herein."
        ),
    },
]

_EMPLOYMENT_AGREEMENT_EXAMPLES: List[Dict[str, str]] = [
    {
        "cuad_category": "Term",
        "clause": (
            "This Agreement shall commence on the Effective Date and shall "
            "continue until terminated by either party in accordance with the "
            "provisions of Section [TERMINATION] (the \"Term\").  Employee's "
            "employment is at-will, except as expressly modified by this "
            "Agreement."
        ),
    },
    {
        "cuad_category": "Non-Compete",
        "clause": (
            "During Employee's employment and for a period of twelve (12) "
            "months following its termination, Employee shall not, directly "
            "or indirectly, engage in or provide services to any business "
            "that competes with the Company in any geographic territory in "
            "which Employee performed material services for the Company, "
            "provided that this restriction shall be enforced only to the "
            "extent permitted under applicable law in [JURISDICTION]."
        ),
    },
    {
        "cuad_category": "Confidentiality",
        "clause": (
            "Employee acknowledges that during employment, Employee will have "
            "access to Confidential Information.  Employee agrees that, both "
            "during and after employment, Employee shall hold all such "
            "Confidential Information in strict confidence and shall not use "
            "or disclose it except as required to perform Employee's duties "
            "or as expressly authorized in writing by the Company."
        ),
    },
    {
        "cuad_category": "IP Assignment",
        "clause": (
            "Employee hereby assigns to the Company all right, title, and "
            "interest in any inventions, works of authorship, and other "
            "intellectual property conceived or developed by Employee during "
            "the course of employment that relate to the Company's business "
            "or that result from work performed for the Company."
        ),
    },
    {
        "cuad_category": "Governing Law",
        "clause": (
            "This Agreement shall be governed by and construed in accordance "
            "with the laws of the State of [JURISDICTION], without regard to "
            "its conflict-of-laws principles."
        ),
    },
]

_RESIDENTIAL_LEASE_EXAMPLES: List[Dict[str, str]] = [
    {
        "cuad_category": "Term",
        "clause": (
            "The initial term of this Lease shall commence on the Start Date "
            "and shall continue for the period specified herein, ending on "
            "the End Date, unless sooner terminated in accordance with the "
            "provisions of this Lease (the \"Term\").  Upon expiration of the "
            "initial Term, this Lease shall convert to a month-to-month "
            "tenancy unless either party gives written notice of "
            "non-renewal at least thirty (30) days prior to the End Date."
        ),
    },
    {
        "cuad_category": "Rent and Late Fees",
        "clause": (
            "Tenant shall pay Landlord the Monthly Rent in advance on or "
            "before the Rent Due Date of each calendar month.  Any payment "
            "not received within five (5) days after the Rent Due Date shall "
            "be subject to a late fee as set forth in this Lease, to the "
            "extent permitted by applicable law in [JURISDICTION]."
        ),
    },
    {
        "cuad_category": "Security Deposit",
        "clause": (
            "Tenant has deposited with Landlord the Security Deposit as "
            "security for the faithful performance of Tenant's obligations "
            "under this Lease.  Landlord shall return the Security Deposit, "
            "less any lawful deductions and accompanied by an itemized "
            "statement, within the period required by applicable law of "
            "[JURISDICTION] following the date Tenant surrenders the "
            "Premises."
        ),
    },
    {
        "cuad_category": "Use of Premises",
        "clause": (
            "Tenant shall use the Premises solely as a private residence for "
            "Tenant and Tenant's immediate family, and for no other purpose "
            "without Landlord's prior written consent.  Tenant shall not "
            "commit waste or any nuisance on or about the Premises and shall "
            "comply with all applicable laws, ordinances, and regulations."
        ),
    },
    {
        "cuad_category": "Maintenance and Repairs",
        "clause": (
            "Landlord shall maintain the structural elements of the Premises "
            "and all building systems in good working condition, except for "
            "damage caused by Tenant's negligence or misuse.  Tenant shall "
            "keep the Premises clean and sanitary and shall promptly notify "
            "Landlord of any condition requiring repair."
        ),
    },
]

_BY_CONTRACT_TYPE: Dict[str, List[Dict[str, str]]] = {
    "Service Agreement": _SERVICE_AGREEMENT_EXAMPLES,
    "Employment Agreement": _EMPLOYMENT_AGREEMENT_EXAMPLES,
    "Residential Lease Agreement": _RESIDENTIAL_LEASE_EXAMPLES,
}


def get_few_shots(contract_type: str, max_examples: int = 4) -> str:
    """Return CUAD-style exemplars formatted for prompt injection.

    The output is a string the draft-generation prompt can interpolate via the
    `{cuad_examples}` placeholder.  When no exemplars exist for the requested
    contract type, an empty string is returned so the prompt degrades cleanly.
    """
    examples = _BY_CONTRACT_TYPE.get(contract_type, [])
    if not examples:
        return ""

    selected = examples[:max_examples]
    blocks = []
    for i, example in enumerate(selected, start=1):
        blocks.append(
            f"Example {i} - {example['cuad_category']} (CUAD-derived):\n"
            f"{example['clause']}"
        )

    header = (
        "Few-shot exemplars below are paraphrased from clauses in the Contract "
        "Understanding Atticus Dataset (CUAD).  Use them as structural priors "
        "for tone, defined-term capitalization, and clause organization - "
        "adapt the substance to the user's inputs and jurisdiction; do not "
        "copy verbatim."
    )

    return header + "\n\n" + "\n\n".join(blocks)


def available_contract_types() -> List[str]:
    """Contract types that currently ship with CUAD exemplars."""
    return list(_BY_CONTRACT_TYPE.keys())
