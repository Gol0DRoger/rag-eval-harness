"""
RAG Evaluation Script

Usage:
  1. Ensure uvicorn backend is running: uvicorn main:app --reload
  2. Run evaluation:
     python eval.py                      # Fast: Keyword matching + metrics
     python eval.py --judge              # Detailed: Adds LLM-as-a-Judge grading
     python eval.py --top-k 5 --fail-under 0.80
"""
import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
NOT_FOUND_MARKER = "find that in the uploaded documents"
EMPTY_DB_MARKER = "no documents are loaded"
RESULTS_DIR = Path("eval_results")

JUDGE_PROMPT = """You are grading a retrieval-augmented QA system.

Question: {question}
Reference answer: {reference}
Retrieved context:
{context}
System answer: {answer}

Grade two things:
- faithful: every claim in the system answer is supported by the retrieved context.
- correct: the system answer agrees with the reference answer.

Reply with ONLY a JSON object, for example: {{"faithful": true, "correct": false}}"""


def contains(text: str, keyword: str) -> bool:
    return keyword.lower() in text.lower()


def mean(values: list) -> float | None:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def p95(values: list) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def check_backend() -> None:
    try:
        requests.get(f"{BACKEND_URL}/docs", timeout=5).raise_for_status()
    except Exception as e:
        sys.exit(f"Error: Backend not reachable at {BACKEND_URL}. Is uvicorn running?\n{e}")


def ask(question: str, top_k: int):
    start = time.perf_counter()
    r = requests.post(
        f"{BACKEND_URL}/api/query",
        json={"question": question, "top_k": top_k},
        timeout=180,
    )
    seconds = time.perf_counter() - start
    r.raise_for_status()
    data = r.json()
    answer = data.get("answer", "")
    if EMPTY_DB_MARKER in answer.lower():
        sys.exit("Error: Vector DB is empty. Ingest the Sapiens PDF first.")
    return answer, data.get("contexts"), seconds


def score_retrieval(contexts: list[str], keywords: list[str]):
    if not keywords:
        return 1.0, 1.0
    blob = " ".join(contexts)
    recall = sum(contains(blob, k) for k in keywords) / len(keywords)
    reciprocal_rank = 0.0
    for rank, chunk in enumerate(contexts, start=1):
        if any(contains(chunk, k) for k in keywords):
            reciprocal_rank = 1.0 / rank
            break
    return recall, reciprocal_rank


def judge(client: OpenAI, question: str, reference: str, contexts: list[str], answer: str):
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        context="\n---\n".join(contexts)[:8000],
        answer=answer,
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0,
    )
    text = response.choices[0].message.content or ""
    matches = re.findall(r"\{[^{}]*\}", text)
    try:
        data = json.loads(matches[-1])
        return bool(data["faithful"]), bool(data["correct"])
    except Exception:
        return None, None


def run_case(case: dict, top_k: int, min_keywords: float, judge_client) -> dict:
    question = case["question"]
    answer, contexts, seconds = ask(question, top_k)
    refused = NOT_FOUND_MARKER in answer.lower()

    row = {
        "id": case["id"],
        "question": question,
        "answerable": case["answerable"],
        "answer": answer,
        "refused": refused,
        "latency_s": round(seconds, 2),
    }

    if case["answerable"]:
        keywords = case.get("keywords", [])
        if contexts is not None:
            row["context_recall"], row["mrr"] = score_retrieval(contexts, keywords)
        row["keyword_recall"] = (
            sum(contains(answer, k) for k in keywords) / len(keywords) if keywords else 1.0
        )
        row["passed"] = (not refused) and row["keyword_recall"] >= min_keywords
        if judge_client and contexts is not None and not refused:
            row["faithful"], row["correct"] = judge(
                judge_client, question, case.get("reference", ""), contexts, answer
            )
    else:
        row["passed"] = refused
    return row


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]
    judged = [r for r in answerable if r.get("faithful") is not None]
    return {
        "cases": len(rows),
        "errors": len(rows) - len(ok),
        "overall_pass_rate": mean([r["passed"] for r in rows]),
        "context_recall": mean([r.get("context_recall") for r in answerable]),
        "mrr": mean([r.get("mrr") for r in answerable]),
        "answer_pass_rate": mean([r["passed"] for r in answerable]),
        "false_refusal_rate": mean([r["refused"] for r in answerable]),
        "correct_refusal_rate": mean([r["passed"] for r in unanswerable]),
        "faithfulness": mean([r.get("faithful") for r in judged]),
        "correctness": mean([r.get("correct") for r in judged]),
        "latency_avg_s": mean([r["latency_s"] for r in ok]),
        "latency_p95_s": p95([r["latency_s"] for r in ok]),
    }


def print_report(rows: list[dict], summary: dict, previous: dict | None) -> None:
    def fmt(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) else "-"

    print(f"\n{'RESULT':<8}{'CASE ID':<26}{'ANS':>6}{'CTX':>6}{'MRR':>6}{'SEC':>7}")
    print("-" * 60)
    for r in rows:
        status = "ERROR" if "error" in r else ("PASS" if r["passed"] else "FAIL")
        print(
            f"{status:<8}{r['id']:<26}{fmt(r.get('keyword_recall')):>6}"
            f"{fmt(r.get('context_recall')):>6}{fmt(r.get('mrr')):>6}{fmt(r.get('latency_s')):>7}"
        )
        if status != "PASS":
            detail = r.get("error") or r["answer"].replace("\n", " ")[:120]
            print(f"        -> {detail}")

    print("\nSUMMARY METRICS")
    print("-" * 60)
    for key, value in summary.items():
        if value is None:
            continue
        text = f"{value:.3f}" if isinstance(value, float) else str(value)
        line = f"  {key:<24}{text}"
        prev = (previous or {}).get(key)
        if isinstance(value, float) and isinstance(prev, float):
            diff = value - prev
            line += f"   ({diff:+.3f} vs last run)"
        print(line)


def save_results(results_dir: Path, rows: list[dict], summary: dict, config: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"config": config, "summary": summary, "results": rows}, indent=2)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (results_dir / f"{stamp}.json").write_text(payload, encoding="utf-8")
    (results_dir / "latest.json").write_text(payload, encoding="utf-8")


def load_previous_summary(results_dir: Path) -> dict | None:
    latest = results_dir / "latest.json"
    if latest.exists():
        try:
            return json.loads(latest.read_text(encoding="utf-8")).get("summary")
        except Exception:
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--dataset", default="eval_set_sapiens.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--judge", action="store_true", help="Also grade faithfulness/correctness with Groq LLM")
    parser.add_argument("--min-keywords", type=float, default=0.5)
    parser.add_argument("--fail-under", type=float, default=0.7)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        sys.exit(f"Error: Dataset file '{args.dataset}' not found.")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = dataset["cases"]
    print(f"Evaluating {len(cases)} cases against: {dataset.get('document', 'loaded document')}")

    check_backend()

    judge_client = None
    if args.judge:
        judge_client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

    results_dir = RESULTS_DIR / dataset_path.stem
    previous = load_previous_summary(results_dir)

    rows = []
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] Running test: {case['id']}...", flush=True)
        try:
            rows.append(run_case(case, args.top_k, args.min_keywords, judge_client))
        except Exception as e:
            rows.append({
                "id": case["id"],
                "question": case["question"],
                "answerable": case["answerable"],
                "passed": False,
                "error": str(e),
            })
        time.sleep(args.delay)

    summary = summarize(rows)
    print_report(rows, summary, previous)

    if any(r["answerable"] and "error" not in r and "context_recall" not in r for r in rows):
        print('\nNote: Retrieval scores (CTX/MRR) are blank because /api/query does not return "contexts".')
        print('Ensure main.py includes `"contexts": found["contexts"]` in the dict returned by /api/query.')

    config = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "top_k": args.top_k,
        "llm_model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        "judge": args.judge,
        "min_keywords": args.min_keywords,
        "dataset": args.dataset,
    }
    save_results(results_dir, rows, summary, config)

    rate = summary["overall_pass_rate"] or 0
    passed = rate >= args.fail_under
    print(f"\nSTATUS: {'PASSED' if passed else 'FAILED'} (Pass Rate: {rate:.3f} | Threshold: {args.fail_under})")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()