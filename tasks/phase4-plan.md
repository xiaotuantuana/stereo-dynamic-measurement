# Implementation Plan: Phase 4 FULL_ENHANCED

## Overview

Implement the approved Phase 4 design in five gated slices.  The existing
`TemporalStereoPipeline` remains the sole final-result owner.  Every new
final-write path is experiment-scoped and disabled by default.

## Tasks

### Slice 1: Contracts and red tests

- Define immutable baseline, authority, I2/I3 outcomes, candidate safety, and
  arbitration decision contracts.
- Add failing tests for authority and atomic final-decision semantics.
- Verify that the current implementation cannot yet satisfy the new contract.

### Slice 2: I2 state isolation

- Add a standalone stateful processor with raw/corrected/trusted/prediction
  separation, episode tracking, hard-failure quarantine, and timestamped
  recovery confirmation.
- Add failing-then-passing correction-feedback, recovery, legitimate-motion,
  and complete-correction-or-abstention tests.

### Slice 3: Controlled pipeline integration

- Connect the processor and pure arbitrator at the existing pipeline finalizer.
- Add only the experiment authority path and the pipeline-owned atomic commit.
- Prove default and Shadow equivalence plus 199-column CSV compatibility.

### Slice 4: Structured I3 and end-to-end smoke

- Adapt I3 diagnosis to a typed recommendation boundary.
- Prove it cannot mutate results and that its blocking policy is respected.
- Run a small deterministic FULL_ENHANCED smoke only.

### Slice 5: Controlled validation

- Produce the three Phase 4 tables from six I2 scenarios and four system modes.
- Reuse and hash-check the existing 500-frame manifest only after all prior
  gates pass.
- Decide `ALLOW_2000`; do not launch a 2000-frame run.

## Checkpoints

- After slices 1-2: focused unit tests pass; no pipeline behavior changes.
- After slices 3-4: focused integration tests and full regression pass.
- After slice 5: controlled evidence and 500-frame comparison are complete;
  scale decision is documented.
