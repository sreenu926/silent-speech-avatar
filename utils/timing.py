"""
utils/timing.py
High-resolution timestamp utility for end-to-end latency instrumentation.
"""

import time


def now_ms() -> float:
    """Returns current UNIX time in milliseconds."""
    return time.time() * 1000.0


def elapsed_ms(start_ms: float) -> float:
    """Returns elapsed time in milliseconds since start_ms."""
    return now_ms() - start_ms