from dataclasses import dataclass


@dataclass
class CandidateRuntimeConfig:
    enabled: bool = False
    # None preserves exec.timeout. These budgets INCLUDE validation and export.
    draft_budget_seconds: int | None = None
    candidate_budget_seconds: int | None = None
    first_validation_seconds: float = 900
    validation_interval_seconds: float = 1800
    export_interval_seconds: float = 3600
    smoke_steps: int = 5
    smoke_rows: int = 128
    finalization_reserve_seconds: float = 900
    finalization_safety_factor: float = 1.5
    validation_fraction: float = 0.05
    keep_snapshots: int = 2

    def validate(self):
        for name in ("first_validation_seconds", "validation_interval_seconds",
                     "export_interval_seconds", "smoke_steps", "smoke_rows",
                     "finalization_reserve_seconds", "keep_snapshots"):
            if getattr(self, name) <= 0:
                raise ValueError(f"candidate_runtime.{name} must be positive")
        for name in ("draft_budget_seconds", "candidate_budget_seconds"):
            if getattr(self, name) is not None and getattr(self, name) <= 0:
                raise ValueError(f"candidate_runtime.{name} must be positive or null")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.finalization_safety_factor < 1:
            raise ValueError("finalization_safety_factor must be at least 1")
