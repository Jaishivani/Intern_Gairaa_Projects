import re


SENSITIVE_PATTERNS = [
    r"\baccount\s*number\b",
    r"\baccount\s*no\b",
    r"\baccount\s*#\b",
    r"\bpassword\b",
    r"\bpin\b",
    r"\botp\b",
    r"\bcvv\b",
    r"\bapi\s*key\b",
    r"\bauthentication\s*token\b",
    r"\bauth\s*token\b",
    r"\baccess\s*token\b",
    r"\bsecret\s*key\b",
]


def check_input_guardrail(message: str) -> str | None:
    """
    Blocks requests asking for sensitive authentication
    or security information.
    """

    text = message.lower().strip()

    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, text):
            return (
                "For security reasons, I can't provide "
                "sensitive security information."
            )

    return None


def check_output_guardrail(answer: str) -> str:
    """
    Checks the LLM response before returning it to the user.
    """

    sensitive_patterns = [
        r"\bpassword\b",
        r"\bpin\b",
        r"\botp\b",
        r"\bcvv\b",
        r"\bapi\s*key\b",
        r"\bauthentication\s*token\b",
        r"\bauth\s*token\b",
        r"\baccess\s*token\b",
    ]

    text = answer.lower()

    for pattern in sensitive_patterns:
        if re.search(pattern, text):
            return (
                "For security reasons, I can't provide "
                "sensitive security information."
            )

    return answer