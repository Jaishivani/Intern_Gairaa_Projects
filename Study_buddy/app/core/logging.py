"""Application logger and compact workflow-step telemetry."""

import logging

logger = logging.getLogger("studybuddy")


def record_step(step_name: str, latency_ms: float, token_count: int) -> None:
    """Log the name, elapsed milliseconds, and estimated token count of a step."""
    logger.info(
        "workflow_step=%s latency_ms=%.2f estimated_tokens=%d",
        step_name,
        latency_ms,
        token_count,
    )
