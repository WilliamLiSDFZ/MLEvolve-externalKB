"""Cooperative candidate execution; snapshots are artifacts, not search nodes."""

__all__ = ["CandidateSession"]


def __getattr__(name):
    if name == "CandidateSession":
        from .session import CandidateSession
        return CandidateSession
    raise AttributeError(name)
