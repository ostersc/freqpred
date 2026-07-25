---
name: spec-and-issues
description: The SPEC.md + GitHub issue workflow for freqpred — task numbering (T{N} = issue #N), the required issue body format, how to decide what to work on next, and the ordered steps for completing a task. Use when asked "what's next", "what should we work on", when told to start or finish a tracked task, or when creating a new task/issue.
---

# GitHub issues and SPEC workflow

**SPEC.md is the source of truth for what will be built.** GitHub issues contain the implementation detail for each task. The two must stay in sync.

Note: the rule for *keeping SPEC.md current* on every code change lives in the root `CLAUDE.md` — it applies always, not just when this skill is loaded.

## Task numbering

Every planned task has a `T{N}` identifier in SPEC.md that matches its GitHub issue number. For example, T37 = issue #37. Never create a new task without creating a matching issue, and never create an issue without a matching SPEC entry. If a task is added mid-phase (e.g. discovered scope), pick the next available issue number and use that as the task ID.

## Issue format

Each issue must follow this structure (see issues #37–#48 as reference):
- **`## Context`** — what problem this solves and why it is needed; note any `**Depends on:**` tasks
- **`## Implementation scope`** — specific files to change, key function signatures, code sketches
- **`## New tests`** — named test cases with what each verifies
- **`## Acceptance criteria`** — checkbox list; these are the gates that must pass before closing the issue

## Determining what to work on next

When asked "what's next", "what should we work on", or told to start the next task:
1. Read the Phase 3 checkbox list in SPEC.md — identify all unchecked tasks
2. Run `gh issue list --state open` and compare against SPEC.md checkboxes — flag any discrepancy (e.g. SPEC says done but issue still open, or vice versa) and resolve it before proceeding
3. From the unchecked tasks, identify which are unblocked (all dependencies are checked in SPEC.md)
4. Recommend the highest-priority unblocked task, stating its T-number, title, and why it's next in dependency order
5. **Ask for confirmation before starting any implementation**

## Workflow

- When starting a task, read the GitHub issue before writing any code
- Check off acceptance criteria in the issue as they are met
- If scope changes during implementation, update both the issue body and SPEC.md before proceeding
- When adding a new Phase 3 task: add a checkbox entry to the Phase 3 list in SPEC.md (with issue link), create the issue, then implement

## Completing a task

When all acceptance criteria are met and tests pass, in this order:
1. **Summarize** what was implemented (files changed, key design decisions) and list the manual validation steps from the issue's acceptance criteria that require runtime verification (e.g. "run with --mode live and observe X"). Present this to the user and wait for them to confirm manual validation is done before proceeding.
2. Mark the task's checkbox in the Phase 3 list in SPEC.md as `[x]`
3. Commit and push the code
4. Close the GitHub issue (`gh issue close <N>`) — always **after** pushing, never before
