"""Per-request deadline, cancellation and metrics (never global counters)."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Event
from queue import Queue
import time


class RequestTimeout(TimeoutError):
    pass


@dataclass
class RequestBudget:
    seconds: float
    started: float = field(default_factory=time.monotonic)
    cancelled: Event = field(default_factory=Event)
    events: Queue = field(default_factory=Queue)
    last_result: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=lambda: {"timings": {}, "llm_calls": 0,
                                                  "input_tokens": 0, "output_tokens": 0})

    def remaining(self):
        remaining = self.seconds - (time.monotonic() - self.started)
        if self.cancelled.is_set() or remaining <= 0:
            raise RequestTimeout("请求时间预算已耗尽")
        return remaining


_budget = ContextVar("rag_request_budget", default=None)


@contextmanager
def request_scope(budget):
    token = _budget.set(budget)
    try:
        budget.remaining()
        yield budget
    finally:
        _budget.reset(token)


def check_budget():
    budget = _budget.get()
    if budget:
        budget.remaining()


def bounded_timeout(seconds):
    budget = _budget.get()
    return min(seconds, budget.remaining()) if budget else seconds


def record_usage(response=None):
    budget = _budget.get()
    if budget:
        if response is None:
            budget.metrics["llm_calls"] += 1
        elif getattr(response, "usage", None):
            budget.metrics["input_tokens"] += response.usage.prompt_tokens or 0
            budget.metrics["output_tokens"] += response.usage.completion_tokens or 0


def emit_event(kind, value=""):
    budget = _budget.get()
    if budget:
        if kind == "token" and value and "first_token_seconds" not in budget.metrics:
            budget.metrics["first_token_seconds"] = time.monotonic() - budget.started
        budget.events.put((kind, value))


def checkpoint(state):
    budget = _budget.get()
    if budget and state.get('answer') and not state.get('generation_error'):
        budget.last_result = dict(state)


@contextmanager
def timed(name):
    check_budget()
    if name.startswith("node.") or name == "initialization":
        emit_event("stage", name)
    start = time.perf_counter()
    try:
        yield
    finally:
        budget = _budget.get()
        if budget:
            timings = budget.metrics["timings"]
            timings[name] = timings.get(name, 0) + time.perf_counter() - start
    check_budget()
