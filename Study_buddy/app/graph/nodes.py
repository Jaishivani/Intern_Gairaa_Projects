"""Workflow steps for routing and executing study requests."""

import re
import time
from typing import Any

from app.core.config import get_settings
from app.core.guardrails import check_input, check_output
from app.core.logging import record_step
from app.graph.state import StudyState
from app.services.embedding_service import embed_texts
from app.services.cache_service import get_cached_response, store_response
from app.services.llm_service import (
    answer_from_document,
    answer_from_web,
    generate_quiz,
    judge_document_answer,
    polish_summary,
)
from app.services.retrieval_service import retrieve_chunks
from app.services.search_service import search_web
from app.services.summary_service import extractive_summary
from app.services.vector_store import get_document_chunks, query_document


_FOLLOW_UP_PATTERN = re.compile(
    r"\b(?:again|that|it|this|those|them|above|previous|last answer|"
    r"explain|elaborate|clarify|rephrase|simpler|simple words)\b",
    re.IGNORECASE,
)


def _is_contextual_follow_up(state: StudyState) -> bool:
    """Check whether the current question refers to an earlier exchange."""
    history = state.get("history", [])
    return bool(history and _FOLLOW_UP_PATTERN.search(state["message"]))


def _retrieval_query(state: StudyState) -> str:
    """Resolve follow-up references using the preceding user question and answer."""
    message = state["message"]
    if not _is_contextual_follow_up(state):
        return message

    previous_answer = next(
        (turn["content"] for turn in reversed(state["history"]) if turn["role"] == "assistant"),
        None,
    )
    previous_question = next(
        (turn["content"] for turn in reversed(state["history"]) if turn["role"] == "user"),
        None,
    )
    if not previous_question or not previous_answer:
        return message
    return f"{previous_question}\n{previous_answer[:1200]}\nFollow-up: {message}"


def guardrail_node(state: StudyState) -> dict[str, Any]:
    """Block overlong, unrelated, or prompt-injection requests."""
    reason = check_input(state["message"], get_settings().max_message_length)
    if reason is None:
        return {"blocked": False}
    return {
        "blocked": True,
        "answer": reason,
        "mode": "qa" if state["requested_mode"] == "auto" else state["requested_mode"],
        "source": "document",
    }


def route_node(state: StudyState) -> dict[str, str]:
    """Resolve auto mode with simple keyword rules before selecting a branch."""
    mode = state["requested_mode"]
    if mode == "auto":
        message = state["message"].lower()
        if re.search(r"\b(quiz|test me|practice questions|make questions)\b", message):
            mode = "quiz"
        elif re.search(r"\b(summarize|summary|overview|key points)\b", message):
            mode = "summary"
        else:
            mode = "qa"
    return {"mode": mode}


def cache_check_node(state: StudyState) -> dict[str, Any]:
    """Load a cached answer for the resolved document, question, and mode."""
    if _is_contextual_follow_up(state):
        return {"cache_hit": False}
    cached = get_cached_response(
        state["document_id"],
        state["message"],
        state["mode"],
        state["quiz_count"],
    )
    if cached is None:
        return {"cache_hit": False}
    return {
        "cache_hit": True,
        "cached_response": cached,
        **cached,
    }


def retrieve_node(state: StudyState) -> dict[str, Any]:
    """Retrieve supporting PDF passages for the current request."""
    document_id = state["document_id"]
    corpus = get_document_chunks(document_id)
    if not corpus:
        return {"chunks": []}
    settings = get_settings()
    query = _retrieval_query(state)
    query_embedding = embed_texts([query])[0]
    vector_results = query_document(document_id, query_embedding, min(len(corpus), settings.retrieval_top_k * 4))
    chunks = retrieve_chunks(query, corpus, vector_results, settings.retrieval_top_k)
    return {"chunks": chunks}


def answer_node(state: StudyState) -> dict[str, str]:
    """Generate an answer grounded in retrieved passages and recent history."""
    answer = answer_from_document(state["message"], state["chunks"], state.get("history", []))
    return {"answer": answer, "source": "document"}


def summary_node(state: StudyState) -> dict[str, Any]:
    """Summarize retrieved passages and polish the extractive result."""
    chunks = get_document_chunks(state["document_id"])
    summary, selected = extractive_summary(chunks)
    if not summary:
        return {
            "answer": "No extractable text was found in the document.",
            "citations": [],
            "chunks": chunks,
        }
    return {
        "answer": polish_summary(summary),
        "source": "document",
        "citations": selected,
        "chunks": chunks,
    }


def quiz_node(state: StudyState) -> dict[str, Any]:
    """Generate the requested validated quiz questions."""
    questions = generate_quiz(state["message"], state["chunks"], state["quiz_count"])
    answer = "\n\n".join(
        f"{index}. {question.question}\n"
        + "\n".join(f"{letter}. {option}" for letter, option in zip("ABCD", question.options))
        + f"\nAnswer: {question.answer}\nExplanation: {question.explanation} [Page {question.page}]"
        for index, question in enumerate(questions, start=1)
    )
    if not answer:
        answer = "No document passages were found for a quiz."
    return {"answer": answer, "quiz": questions, "source": "document"}


def judge_node(state: StudyState) -> dict[str, bool]:
    """Decide whether the document answer is supported by retrieved evidence."""
    supported = judge_document_answer(state["message"], state["answer"], state["chunks"])
    return {"supported": supported}


def web_search_node(state: StudyState) -> dict[str, Any]:
    """Search the web and answer with source URLs when PDF evidence is insufficient."""
    results = search_web(state["message"])
    answer = answer_from_web(state["message"], results)
    return {
        "answer": answer if answer.lower().startswith("from the web") else f"From the web: {answer}",
        "source": "web",
        "urls": [result["url"] for result in results],
        "citations": [],
    }


def finalize_node(state: StudyState) -> dict[str, Any]:
    """Attach document-page citations mentioned by the generated answer."""
    started_at = state.get("started_at", time.perf_counter())
    raw_answer = state.get("answer", "")
    answer = check_output(raw_answer)
    output_blocked = answer != raw_answer.strip()
    pages = set() if output_blocked else {
        int(page) for page in re.findall(r"\[Page\s+(\d+)\]", answer)
    }
    page_chunks = state.get("citations") or state.get("chunks", [])
    citations: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for chunk in page_chunks:
        page = int(chunk["page"])
        text = str(chunk["text"])
        if page in pages and (page, text) not in seen:
            citations.append({"page": page, "text": text})
            seen.add((page, text))

    quiz = [] if output_blocked else state.get("quiz", [])
    if output_blocked or state.get("source") == "web":
        citations = []
    elif quiz:
        pages = {question.page for question in quiz}
        page_citations: dict[int, dict[str, Any]] = {}
        for chunk in state.get("chunks", []):
            page = int(chunk["page"])
            if page in pages:
                page_citations.setdefault(page, {"page": page, "text": str(chunk["text"])})
        citations = list(page_citations.values())
    result = {
        "answer": answer,
        "citations": citations,
        "source": state.get("source", "document"),
        "urls": state.get("urls", []),
        "quiz": quiz,
        "mode": state.get("mode", "qa"),
        "chunks": state.get("chunks", []),
    }
    if not state.get("blocked", False):
        store_response(
            state["document_id"],
            state["message"],
            str(result["mode"]),
            result,
            state.get("quiz_count", 0),
        )
    record_step("finalize", (time.perf_counter() - started_at) * 1000, len(answer.split()))
    return result
