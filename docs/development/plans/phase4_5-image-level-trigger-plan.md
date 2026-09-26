# Phase 4.5 Implementation Plan

## Task 1: Image-only injection contract and tests

Acceptance: immutable validated case spec, deterministic image-copy transforms,
and paired source provenance are covered by focused tests.

## Task 2: Frozen window manifest and controlled M1/M2/M3 runner

Acceptance: 32 cases/controls are selected from the Phase 3 manifest; each
mode uses the same pre-commit evaluation and only M3 may write final output.

## Task 3: Audit tables and gate evaluator

Acceptance: all five CSVs have the specified fields and semantic distinctions;
gate output reports pass/fail reasons without changing production CSV output.

## Task 4: End-to-end validation and report

Acceptance: focused/full tests run, bounded case execution completes, all gates
are calculated, and the report retains `ALLOW_2000=NO` unless every gate passes.

## Checkpoints

- After Task 1: focused injector tests pass.
- After Task 2: runner/authority tests pass.
- After Task 3: output and gate tests pass.
- After Task 4: full `python -m pytest -q` passes; no 2000/5000/full run.
