"""Agentic loop: Retrieve -> Generate -> Verify -> Retry."""
from src.generator_agent import generate_answer
from src.critic_agent import verify

def answer_with_verification(question, retriever, max_retries=3):
    log = []
    for attempt in range(1, max_retries + 1):
        retrieved = retriever.retrieve(question, k=5)
        draft = generate_answer(question, retrieved)
        report = verify(draft["answer"], retrieved)
        log.append({"attempt": attempt, "answer": draft["answer"], "cited": draft["cited_paper_ids"], "grounded": report["grounded"], "score": report["score"], "unsupported": len(report["unsupported_claims"])})
        if report["grounded"]:
            return {"final": draft["answer"], "cited": draft["cited_paper_ids"], "report": report, "retrieved": retrieved, "log": log, "status": "grounded"}
    return {"final": "INSUFFICIENT_EVIDENCE", "cited": [], "report": log[-1] if log else {}, "retrieved": [], "log": log, "status": "ungrounded_after_retries"}