import json
from typing import Any

from openai import AuthenticationError, OpenAI, PermissionDeniedError

from app.config import get_nvidia_model, get_required_setting


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _client() -> OpenAI:
    return OpenAI(
        api_key=get_required_setting("NVIDIA_API_KEY"),
        base_url=NVIDIA_BASE_URL,
    )


def _complete(messages: list[dict[str, str]]) -> str:
    try:
        response = _client().chat.completions.create(
            model=get_nvidia_model(),
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )

    except (AuthenticationError, PermissionDeniedError) as error:
        raise RuntimeError(
            "NVIDIA API authentication failed. "
            "Check that NVIDIA_API_KEY is active and has inference access."
        ) from error

    except Exception as error:
        raise RuntimeError(
            f"NVIDIA request failed: {error}"
        ) from error

    if not response.choices:
        raise RuntimeError(
            "NVIDIA returned no response choices."
        )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "NVIDIA returned an empty response."
        )

    return content.strip()


def classify_question(question: str) -> dict[str, Any]:
    content = _complete(
        [
            {
                "role": "system",
                "content": (
                    "You are the routing step for a Kitchen Assistant. "
                    "Decide whether the user's question needs current or "
                    "external web information.\n\n"

                    "Use web search when the user asks for latest, "
                    "current, recent, today, online information, or "
                    "explicitly asks to search the web.\n\n"

                    "Do not use web search for normal cooking knowledge, "
                    "recipes, ingredients, cooking methods, or basic "
                    "kitchen questions.\n\n"

                    "Return ONLY valid JSON with exactly these keys:\n"
                    "{\n"
                    '  "search_required": true or false,\n'
                    '  "search_query": "string",\n'
                    '  "answer": "string"\n'
                    "}\n\n"

                    "If search_required is true, answer must be empty.\n"
                    "If search_required is false, answer must contain "
                    "the complete helpful answer."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]
    )

    # Debug: shows exactly what NVIDIA returned
    print("\nDEBUG NVIDIA RESPONSE:")
    print(content)
    print()

    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"NVIDIA returned an invalid routing response: {content}"
        ) from error

    if not isinstance(result, dict):
        raise RuntimeError(
            "NVIDIA returned an invalid routing response."
        )

    if not isinstance(
        result.get("search_required"),
        bool,
    ):
        raise RuntimeError(
            "NVIDIA returned an invalid search decision."
        )

    if result["search_required"]:
        search_query = str(
            result.get("search_query", "")
        ).strip()

        if not search_query:
            raise RuntimeError(
                "NVIDIA did not provide a search query."
            )

    return result


def answer_with_search_results(
    question: str,
    search_results: list[dict[str, str]],
) -> str:

    sources = "\n\n".join(
        (
            f"Title: {item.get('title', '')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Content: {item.get('content', '')}"
        )
        for item in search_results
    )

    return _complete(
        [
            {
                "role": "system",
                "content": (
                    "You are a careful Kitchen Assistant. "
                    "Answer the user's question using the supplied "
                    "web search results.\n\n"
                    "Do not invent facts.\n"
                    "Use current information from the results.\n"
                    "Give a clear and simple answer.\n"
                    "For recipes, include ingredients, steps, "
                    "cooking time, and practical tips.\n"
                    "Do not add a sources section; the application will "
                    "attach verified source links separately."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Web results:\n{sources}"
                ),
            },
        ]
    )