# pip install anthropic python-dotenv
# For production RAGAS evaluation: pip install ragas
import os, json
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()

claude = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ══════════════════════════════════════════════════════
# RAG EVALUATION — The 4 metrics every FDE must know
#
# 1. Faithfulness    — Is the answer grounded in the context?
#                      (no hallucinations)
# 2. Answer Relevancy — Does the answer actually address the question?
# 3. Context Recall  — Did retrieval find ALL needed information?
# 4. Context Precision — Are retrieved chunks actually relevant?
#
# Think of it as a 2x2:
#   Retrieval quality:  Precision × Recall
#   Generation quality: Faithfulness × Relevancy
# ══════════════════════════════════════════════════════


# ── LLM-as-judge evaluation ───────────────────────────
# Use Claude to evaluate Claude — common production pattern

def evaluate_faithfulness(question: str, context: str, answer: str) -> dict:
    """
    Score: Is every claim in the answer supported by the context?
    1.0 = fully grounded  |  0.0 = completely hallucinated
    """
    prompt = f"""Evaluate if the answer is faithful to (supported by) the context.

<question>{question}</question>

<context>{context}</context>

<answer>{answer}</answer>

For each factual claim in the answer, check if it is explicitly supported by the context.
Return JSON only:
{{
  "score": 0.0 to 1.0,
  "unsupported_claims": ["list of claims NOT in context"],
  "reasoning": "brief explanation"
}}"""

    r = claude.messages.create(
        model="claude-sonnet-4-6", max_tokens=300, temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = r.content[0].text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except:
        return {"score": -1, "error": raw}


def evaluate_relevancy(question: str, answer: str) -> dict:
    """
    Score: Does the answer actually address the question?
    1.0 = perfectly answers  |  0.0 = completely off-topic
    """
    prompt = f"""Evaluate if this answer actually addresses the question.

<question>{question}</question>

<answer>{answer}</answer>

Return JSON only:
{{
  "score": 0.0 to 1.0,
  "reasoning": "brief explanation"
}}"""

    r = claude.messages.create(
        model="claude-sonnet-4-6", max_tokens=200, temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = r.content[0].text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except:
        return {"score": -1, "error": raw}


def evaluate_context_relevance(question: str, retrieved_chunks: list[str]) -> dict:
    """
    Score: What fraction of retrieved chunks are actually relevant?
    Measures retrieval precision.
    """
    chunks_str = "\n---\n".join(f"[Chunk {i+1}]: {c}" for i, c in enumerate(retrieved_chunks))
    prompt = f"""Evaluate which retrieved chunks are relevant to answering the question.

<question>{question}</question>

<retrieved_chunks>
{chunks_str}
</retrieved_chunks>

Return JSON only:
{{
  "relevant_chunks": [list of chunk numbers that are relevant, e.g. [1, 3]],
  "irrelevant_chunks": [list of irrelevant chunk numbers],
  "precision_score": fraction of relevant chunks (0.0 to 1.0)
}}"""

    r = claude.messages.create(
        model="claude-sonnet-4-6", max_tokens=200, temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = r.content[0].text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except:
        return {"precision_score": -1, "error": raw}


# ── Full evaluation run ────────────────────────────────
def run_evaluation_suite(test_cases: list[dict]) -> dict:
    """
    Run full evaluation on a set of RAG test cases.
    Each test case: {question, context, answer, retrieved_chunks}
    """
    results = []
    totals  = {"faithfulness": 0, "relevancy": 0, "precision": 0}

    print("=== RAG Evaluation Suite ===\n")

    for i, tc in enumerate(test_cases):
        print(f"Test {i+1}: {tc['question']}")

        faith  = evaluate_faithfulness(tc["question"], tc["context"], tc["answer"])
        rel    = evaluate_relevancy(tc["question"], tc["answer"])
        prec   = evaluate_context_relevance(tc["question"], tc["retrieved_chunks"])

        f_score = faith.get("score", 0)
        r_score = rel.get("score", 0)
        p_score = prec.get("precision_score", 0)

        totals["faithfulness"] += f_score
        totals["relevancy"]    += r_score
        totals["precision"]    += p_score

        print(f"  Faithfulness: {f_score:.2f} | Relevancy: {r_score:.2f} | Precision: {p_score:.2f}")
        if faith.get("unsupported_claims"):
            print(f"  Unsupported claims: {faith['unsupported_claims']}")
        print()

        results.append({
            "question":     tc["question"],
            "faithfulness": f_score,
            "relevancy":    r_score,
            "precision":    p_score,
        })

    n = len(test_cases)
    summary = {
        "avg_faithfulness": round(totals["faithfulness"] / n, 3),
        "avg_relevancy":    round(totals["relevancy"] / n, 3),
        "avg_precision":    round(totals["precision"] / n, 3),
        "overall_score":    round(sum(totals.values()) / (n * 3), 3),
        "total_cases":      n
    }

    print("=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return {"cases": results, "summary": summary}


# ── Run with sample test cases ────────────────────────
test_cases = [
    {
        "question": "How do I fix DS-401 error?",
        "context":  "DS-401: Token expired. Go to Settings > Connections > Refresh Token.",
        "answer":   "The DS-401 error means your authentication token has expired. To fix it, navigate to Settings > Connections and click Refresh Token next to the affected connector.",
        "retrieved_chunks": [
            "DS-401: Token expired. Go to Settings > Connections > Refresh Token.",
            "DS-500: Internal server error. Check status.datasync.io.",
        ]
    },
    {
        "question": "What databases are supported?",
        "context":  "Supported databases: PostgreSQL, MySQL, MariaDB, SQLite, MongoDB, DynamoDB. Cloud warehouses: Snowflake, BigQuery, Redshift.",
        "answer":   "DataSync Pro supports PostgreSQL, MySQL, MariaDB, SQLite, MongoDB, DynamoDB, Snowflake, BigQuery, and Redshift, among others.",
        "retrieved_chunks": [
            "Supported databases: PostgreSQL, MySQL, MariaDB, SQLite, MongoDB, DynamoDB.",
            "To add a connector: Connections > Add New > Select type > Enter credentials.",
        ]
    },
    {
        "question": "Who is the founder of DataSync?",
        "context":  "DataSync Pro is an enterprise data synchronization platform. It supports SAML, OAuth, and LDAP.",
        "answer":   "DataSync Pro was founded in 2015 by John Smith, a former Google engineer.",  # hallucinated!
        "retrieved_chunks": [
            "DataSync Pro is an enterprise data synchronization platform.",
            "Supported databases: PostgreSQL, MySQL, MongoDB.",
        ]
    },
]

run_evaluation_suite(test_cases)


# ── What to do with low scores ────────────────────────
print("""
=== Fixing Low Scores ===

Low Faithfulness  → Strengthen your system prompt:
                    "Answer ONLY from context. Never use outside knowledge."
                    Add: "If unsure, say 'not in documentation'"

Low Relevancy     → Improve your prompt clarity.
                    Add few-shot examples of good answers.

Low Precision     → Improve chunking (smaller, more focused chunks)
                    Tune n_results (try 3 instead of 5)
                    Add metadata filtering to narrow search space

Low Recall        → Increase n_results
                    Try hybrid search (keyword + semantic)
                    Check if document coverage is sufficient
""")
