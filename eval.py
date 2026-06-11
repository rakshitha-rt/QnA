#!/usr/bin/env python3
"""
Evaluation script for the AI Knowledge Assistant.

Covers the three dimensions documented in README.md:
  1. Retrieval relevance  — are citations useful for answering the query?
  2. Groundedness         — is the answer supported by the returned citations?
  3. Clarification precision — does the system ask only when it should?

Usage:
    python eval.py [--base-url http://localhost:8000] [--project-id <existing-id>]

If --project-id is omitted the script creates a fresh project, uploads both
sample files, and tears it down at the end.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Test cases grounded in the two sample files
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    query: str
    query_type: str = "naive"           # naive | local | global
    expected_keywords: list[str] = field(default_factory=list)  # phrases that must appear in answer or citations
    should_abstain: bool = False         # system should say "no relevant info"
    should_clarify: bool = False         # system should return type=="clarification"
    label: str = ""                      # human-readable description


RETRIEVAL_AND_GROUNDEDNESS_CASES: list[TestCase] = [
    # ── Unstructured (SOP) ──────────────────────────────────────────────────
    TestCase(
        label="Shipment escalation process",
        query="What is the escalation process for delayed shipments?",
        query_type="local",
        expected_keywords=["level 1", "level 2", "level 3", "field operations", "regional operations manager", "vp of operations"],
    ),
    TestCase(
        label="Procurement approval workflow",
        query="What is the approval workflow for procurement requests?",
        query_type="local",
        expected_keywords=["tier a", "tier b", "tier c", "tier d", "department head", "finance manager", "cfo"],
    ),
    TestCase(
        label="Inventory aging KPI definition",
        query="Explain the inventory aging KPI.",
        query_type="local",
        expected_keywords=["aging days", "fresh stock", "dead stock", "45 days", "inventory review committee"],
    ),
    TestCase(
        label="Fill rate target",
        query="What is the fill rate target per branch?",
        query_type="naive",
        expected_keywords=["95%", "fill rate", "90%", "root cause"],
    ),
    TestCase(
        label="Emergency procurement limit",
        query="What is the maximum amount allowed for emergency procurement without prior approval?",
        query_type="local",
        expected_keywords=["25,000", "24 hours", "retrospective"],
    ),

    # ── Structured (CSV) ────────────────────────────────────────────────────
    TestCase(
        label="Highest Q1 sales branch",
        query="Which branch has the highest Q1 total sales?",
        query_type="naive",
        expected_keywords=["mumbai"],
    ),
    TestCase(
        label="Average inventory aging",
        query="What is the average inventory aging in days across all SKUs?",
        query_type="naive",
        expected_keywords=["57", "58"],          # ≈ 57.9 days
    ),
    TestCase(
        label="Top 5 SKUs by aging days",
        query="Show the top 5 SKUs by aging days.",
        query_type="naive",
        expected_keywords=["sku-1002", "sku-1009", "sku-1001", "sku-1018", "sku-1010"],
    ),
    TestCase(
        label="Lowest fill rate branch",
        query="Which branch has the lowest fill rate?",
        query_type="naive",
        expected_keywords=["pune", "93.2"],
    ),
    TestCase(
        label="L3 incident count",
        query="How many Level 3 escalation incidents occurred in Q1 2024?",
        query_type="naive",
        expected_keywords=["3"],
    ),
]

ABSTAIN_CASES: list[TestCase] = [
    TestCase(
        label="Out-of-scope: net profit",
        query="What is ACME Logistics' net profit for FY 2023?",
        should_abstain=True,
    ),
    TestCase(
        label="Out-of-scope: company founder",
        query="Who founded ACME Logistics and in what year?",
        should_abstain=True,
    ),
    TestCase(
        label="Out-of-scope: competitor pricing",
        query="What are the shipping rates charged by BlueDart and DHL in India?",
        should_abstain=True,
    ),
]

CLARIFICATION_CASES: list[TestCase] = [
    # Ambiguous — should trigger clarification
    TestCase(label="Ambiguous: bad report",       query="Why is the report bad?",          should_clarify=True),
    TestCase(label="Ambiguous: best manager",     query="Who is the best manager?",         should_clarify=True),
    # Clear — must NOT trigger clarification
    TestCase(label="Clear: escalation process",   query="What is the escalation process for delayed shipments?",  should_clarify=False),
    TestCase(label="Clear: highest sales branch", query="Which branch has the highest Q1 sales?",                 should_clarify=False),
    TestCase(label="Clear: fill rate target",     query="What is the fill rate target?",                          should_clarify=False),
    TestCase(label="Clear: inventory aging KPI",  query="Explain the inventory aging KPI.",                       should_clarify=False),
    TestCase(label="Clear: L3 incidents",         query="How many L3 escalation incidents happened in Q1 2024?",  should_clarify=False),
    TestCase(label="Clear: aging formula",        query="What is the formula for computing inventory aging days?", should_clarify=False),
]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def create_project(base_url: str, name: str = "eval-project") -> str:
    r = requests.post(f"{base_url}/projects", json={"name": name}, timeout=10)
    r.raise_for_status()
    return r.json()["id"]


def upload_file(base_url: str, project_id: str, path: Path) -> dict:
    with open(path, "rb") as fh:
        r = requests.post(
            f"{base_url}/projects/{project_id}/upload",
            files={"file": (path.name, fh)},
            timeout=300,  # ingestion can be slow
        )
    r.raise_for_status()
    return r.json()


def run_query(base_url: str, project_id: str, query: str, query_type: str = "naive") -> dict:
    payload = {
        "project_id": project_id,
        "query": query,
        "query_type": query_type,
        "clarification_history": [],
    }
    r = requests.post(f"{base_url}/query", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return text.lower()


def score_citation_relevance(citations: list[dict], expected_keywords: list[str]) -> float:
    """
    Per-citation relevance: 2 if directly answers, 1 if related, 0 if irrelevant.
    A citation scores 2 if it contains any expected keyword, else 1 if it has
    non-trivial content, else 0.
    Returns mean score across citations (0 if no citations).
    """
    if not citations:
        return 0.0
    scores = []
    for c in citations:
        text = _normalise(c.get("snippet", "") + " " + c.get("file", ""))
        if any(kw.lower() in text for kw in expected_keywords):
            scores.append(2)
        elif len(text.strip()) > 20:
            scores.append(1)
        else:
            scores.append(0)
    return sum(scores) / len(scores)


def score_groundedness_llm(answer: str, citations: list[dict], expected_keywords: list[str]) -> float:
    """
    LLM judge: fraction of expected_keywords that are supported by the answer
    or its citations.  Falls back to keyword overlap if model is unavailable.
    """
    if not expected_keywords:
        return 1.0

    combined_context = answer + "\n" + "\n".join(
        c.get("snippet", "") for c in citations
    )
    combined_lower = _normalise(combined_context)

    # Fast keyword-overlap fallback — always computed
    keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in combined_lower)
    keyword_score = keyword_hits / len(expected_keywords)

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        system = (
            "You are a grounding checker for a RAG system. "
            "Given an answer + supporting context, and a list of expected facts, "
            "return JSON: {\"grounded_count\": <int>, \"total\": <int>}. "
            "Count only facts explicitly present in the answer or context."
        )
        human = (
            f"Expected facts: {json.dumps(expected_keywords)}\n\n"
            f"Answer + Context:\n{combined_context[:4000]}"
        )
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        data = json.loads(resp.content)
        return data["grounded_count"] / max(data["total"], 1)
    except Exception:
        return keyword_score


def score_abstain(answer: str) -> bool:
    """Returns True if the system correctly abstained."""
    abstain_phrases = [
        "no relevant info",
        "no relevant information",
        "i don't have",
        "i do not have",
        "not available",
        "no information",
        "cannot answer",
    ]
    return any(p in answer.lower() for p in abstain_phrases)


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------

def run_retrieval_groundedness_eval(base_url: str, project_id: str) -> dict:
    results = []
    for tc in RETRIEVAL_AND_GROUNDEDNESS_CASES:
        print(f"  [{tc.label}] querying…", end=" ", flush=True)
        try:
            resp = run_query(base_url, project_id, tc.query, tc.query_type)
        except Exception as exc:
            print(f"ERROR: {exc}")
            results.append({"label": tc.label, "error": str(exc)})
            continue

        if resp.get("type") == "clarification":
            print("got clarification (unexpected)")
            results.append({
                "label": tc.label,
                "retrieval_score": 0.0,
                "groundedness_score": 0.0,
                "note": "unexpected clarification",
            })
            continue

        answer = resp.get("response", "")
        citations = resp.get("citations", [])
        ret_score = score_citation_relevance(citations, tc.expected_keywords)
        grnd_score = score_groundedness_llm(answer, citations, tc.expected_keywords)
        print(f"ret={ret_score:.2f} grnd={grnd_score:.2f}")
        results.append({
            "label": tc.label,
            "query": tc.query,
            "answer_snippet": answer[:200],
            "citations_count": len(citations),
            "retrieval_score": ret_score,
            "groundedness_score": grnd_score,
        })
    return {"cases": results}


def run_abstain_eval(base_url: str, project_id: str) -> dict:
    results = []
    for tc in ABSTAIN_CASES:
        print(f"  [{tc.label}] querying…", end=" ", flush=True)
        try:
            resp = run_query(base_url, project_id, tc.query)
        except Exception as exc:
            print(f"ERROR: {exc}")
            results.append({"label": tc.label, "error": str(exc)})
            continue

        answer = resp.get("response", "")
        abstained = score_abstain(answer)
        print("abstained=YES" if abstained else f"abstained=NO  (answer: {answer[:80]!r})")
        results.append({
            "label": tc.label,
            "abstained_correctly": abstained,
            "answer_snippet": answer[:200],
        })
    return {"cases": results}


def run_clarification_eval(base_url: str, project_id: str) -> dict:
    results = []
    for tc in CLARIFICATION_CASES:
        print(f"  [{tc.label}] querying…", end=" ", flush=True)
        try:
            resp = run_query(base_url, project_id, tc.query)
        except Exception as exc:
            print(f"ERROR: {exc}")
            results.append({"label": tc.label, "error": str(exc)})
            continue

        got_clarification = resp.get("type") == "clarification"
        if tc.should_clarify:
            correct = got_clarification
            status = "TP" if correct else "FN"
        else:
            correct = not got_clarification
            status = "TN" if correct else "FP"
        print(f"{status}")
        results.append({
            "label": tc.label,
            "should_clarify": tc.should_clarify,
            "got_clarification": got_clarification,
            "correct": correct,
            "status": status,
        })
    return {"cases": results}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(rg: dict, ab: dict, cl: dict) -> None:
    sep = "─" * 70

    print(f"\n{sep}")
    print("EVALUATION REPORT")
    print(sep)

    # ── Retrieval + Groundedness ─────────────────────────────────────────────
    print("\n1. RETRIEVAL RELEVANCE  (target: mean ≥ 1.5 / 2.0)")
    valid = [c for c in rg["cases"] if "retrieval_score" in c]
    if valid:
        scores = [c["retrieval_score"] for c in valid]
        mean = sum(scores) / len(scores)
        passed = mean >= 1.5
        print(f"   Mean score : {mean:.2f}  {'PASS ✓' if passed else 'FAIL ✗'}")
        for c in valid:
            bar = "█" * int(c["retrieval_score"] * 5)
            print(f"   {c['label'][:45]:<45} {c['retrieval_score']:.2f}  {bar}")
    else:
        print("   No valid results.")

    print(f"\n2. GROUNDEDNESS  (target: ≥ 0.85)")
    if valid:
        scores = [c["groundedness_score"] for c in valid]
        mean = sum(scores) / len(scores)
        passed = mean >= 0.85
        print(f"   Mean score : {mean:.2f}  {'PASS ✓' if passed else 'FAIL ✗'}")
        for c in valid:
            bar = "█" * int(c["groundedness_score"] * 10)
            print(f"   {c['label'][:45]:<45} {c['groundedness_score']:.2f}  {bar}")
    else:
        print("   No valid results.")

    # ── Abstain ──────────────────────────────────────────────────────────────
    print(f"\n3. ABSTAIN BEHAVIOUR  (should refuse out-of-scope questions)")
    ab_valid = [c for c in ab["cases"] if "abstained_correctly" in c]
    if ab_valid:
        correct = sum(1 for c in ab_valid if c["abstained_correctly"])
        total = len(ab_valid)
        passed = correct == total
        print(f"   Correct : {correct}/{total}  {'PASS ✓' if passed else 'FAIL ✗'}")
        for c in ab_valid:
            icon = "✓" if c["abstained_correctly"] else "✗"
            print(f"   {icon} {c['label']}")
    else:
        print("   No valid results.")

    # ── Clarification ────────────────────────────────────────────────────────
    print(f"\n4. CLARIFICATION PRECISION  (FP < 20%, FN < 10%)")
    cl_valid = [c for c in cl["cases"] if "status" in c]
    if cl_valid:
        ambiguous = [c for c in cl_valid if c["should_clarify"]]
        clear     = [c for c in cl_valid if not c["should_clarify"]]
        fn = sum(1 for c in ambiguous if c["status"] == "FN")
        fp = sum(1 for c in clear     if c["status"] == "FP")
        fn_rate = fn / len(ambiguous) if ambiguous else 0.0
        fp_rate = fp / len(clear)     if clear     else 0.0
        pass_fp = fp_rate < 0.20
        pass_fn = fn_rate < 0.10
        print(f"   False positive rate : {fp_rate:.0%}  {'PASS ✓' if pass_fp else 'FAIL ✗'}  (over-asked on clear queries)")
        print(f"   False negative rate : {fn_rate:.0%}  {'PASS ✓' if pass_fn else 'FAIL ✗'}  (missed ambiguous queries)")
        for c in cl_valid:
            icon = "✓" if c["correct"] else "✗"
            tag  = "ambiguous" if c["should_clarify"] else "clear    "
            print(f"   {icon} [{tag}] {c['label']}  ({c['status']})")
    else:
        print("   No valid results.")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------

SAMPLE_DIR = Path(__file__).parent / "temp"
SAMPLE_FILES = [
    SAMPLE_DIR / "operations_sop.txt",
    SAMPLE_DIR / "operations_inventory_report.csv",
]


def setup_project(base_url: str) -> str:
    print("Setting up evaluation project…")
    project_id = create_project(base_url)
    print(f"  Created project: {project_id}")
    for fp in SAMPLE_FILES:
        if not fp.exists():
            print(f"  WARNING: sample file not found: {fp}")
            continue
        print(f"  Uploading {fp.name}…", end=" ", flush=True)
        result = upload_file(base_url, project_id, fp)
        print(f"done ({result.get('type', '?')})")
    print("  Waiting 5s for ingestion to settle…")
    time.sleep(5)
    return project_id


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the AI Knowledge Assistant")
    parser.add_argument("--base-url",   default="http://localhost:8000")
    parser.add_argument("--project-id", default=None, help="Reuse an existing project (skips upload)")
    parser.add_argument("--output",     default=None, help="Write full JSON results to this file")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    # Health check
    try:
        requests.get(f"{base_url}/projects", timeout=5).raise_for_status()
    except Exception as exc:
        print(f"ERROR: API not reachable at {base_url}: {exc}")
        sys.exit(1)

    project_id = args.project_id or setup_project(base_url)

    print(f"\nRunning evaluation against project {project_id}\n")

    print("── Retrieval & Groundedness ─────────────────────────────────────────")
    rg = run_retrieval_groundedness_eval(base_url, project_id)

    print("\n── Abstain Behaviour ────────────────────────────────────────────────")
    ab = run_abstain_eval(base_url, project_id)

    print("\n── Clarification Precision ──────────────────────────────────────────")
    cl = run_clarification_eval(base_url, project_id)

    print_report(rg, ab, cl)

    if args.output:
        with open(args.output, "w") as fh:
            json.dump({"retrieval_groundedness": rg, "abstain": ab, "clarification": cl}, fh, indent=2)
        print(f"Full results written to {args.output}")


if __name__ == "__main__":
    main()
