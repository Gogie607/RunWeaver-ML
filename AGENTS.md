# AGENTS.md
# RunWeaver-ML

Repository instructions for AI coding agents working on RunWeaver.

These instructions apply to analysis, implementation, refactoring,
testing, and documentation within this repository.

## 1. Required Architectural Context

Before analyzing, modifying, or recommending changes, read:

1. `ARCHITECTURE.md`
   - Describes RunWeaver's components, interfaces, responsibilities,
     and integration boundaries.

2. `docs/ARCHITECTURAL_DECISIONS.md`
   - Records developer-confirmed architectural intentions.
   - Explains historical compromises and intentional tradeoffs.
   - Defines principles for future architectural reviews.

These documents complement each other.

Source code establishes implemented behavior.
Architectural decisions establish design intent.

When implementation and documentation disagree, investigate the
discrepancy rather than assuming either is automatically correct.

Do not repeatedly challenge established architectural decisions
without new evidence or a concrete reason.

## 2. Design Philosophy

RunWeaver provides reusable mechanisms for experimental machine
learning workflows.

It is not an application-specific training framework.

Follow these principles:

- Prefer composition over inheritance.
- Keep components small and independently usable.
- Preserve experimental flexibility.
- Avoid unnecessary abstractions and dependencies.
- Do not impose application-specific policies.
- Respect established interfaces and behavior.
- Favor readability and maintainability over clever implementations.

A different design being possible does not mean the existing design
is defective.

## 3. Configuration and Phase Independence

Configuration dictionaries are intentional.

Consumers are responsible for interpreting and validating the
configuration values they require.

Framework-level validation may protect internal invariants and
provide useful diagnostics without imposing rigid application schemas.

PhaseManager provides convenient organization of sequential phase
configurations. It does not own application policy or execution.

Phases are independent execution configurations.

Do not assume continuity of execution infrastructure, optimizers,
schedulers, parameters, or global training steps between phases.

Do not introduce implicit cross-phase state dependencies.

Refer to `docs/ARCHITECTURAL_DECISIONS.md` for the authoritative
rationale and detailed contracts.

## 4. Architectural Reviews

When reviewing existing code, distinguish between:

**Intentional tradeoff**
A documented design choice with known advantages and limitations.

**Historical compromise**
Established behavior that evolved over time and may have
compatibility implications.

**Documentation ambiguity**
Behavior that is implemented but insufficiently explained.

**Implementation defect**
Behavior that violates an established contract or produces a
demonstrable failure.

For each proposed change:

1. Identify the actual behavior.
2. Explain the concrete problem.
3. Consider existing design intentions.
4. Propose the smallest appropriate correction.
5. Identify compatibility implications.

Do not report hypothetical misuse of deliberately flexible interfaces
as an implementation defect.

If the existing design adequately satisfies its intended purpose,
say so and move on.

## 5. Implementation Rules

Before writing code:

- Inspect relevant implementations and existing interfaces.
- Reuse existing components rather than duplicating functionality.
- Avoid inventing APIs or assuming methods exist.
- Keep changes within the requested scope.
- Preserve established behavior unless a change is intentional.
- Discuss substantial architectural changes before implementation.

Do not perform unrelated refactoring or cosmetic cleanup.

When a requested change conflicts with established architectural
decisions, explain the conflict before proceeding.

## 6. Development Environment

Use the existing project environment for Python execution.

Do not install, upgrade, downgrade, or remove dependencies without
explicit user authorization.

If a dependency is missing, report it rather than modifying the
environment automatically.

Do not assume that RunWeaver owns or manages the environment of
applications that consume it.

## 7. Verification

Verify completed code changes using appropriate existing tests.

Verification may include:

- Syntax and import checks.
- Relevant unit tests.
- Existing integration tests.
- Execution of affected functionality when practical.

Do not claim successful verification unless it was performed.

After implementation, review the changes for:

- Incorrect assumptions about existing interfaces.
- Unintended changes in behavior.
- Violations of documented architectural decisions.
- Unnecessary duplication.
- Unrelated modifications.

Correct identified problems and repeat relevant verification.

## 8. Documentation

Keep documentation consistent with implemented behavior.

Update relevant documentation when established interfaces or
architectural contracts change.

Use:

- `README.md` for introduction and usage.
- `ARCHITECTURE.md` for component responsibilities and structure.
- `docs/ARCHITECTURAL_DECISIONS.md` for design rationale and
  established architectural decisions.

Do not duplicate extensive architectural explanations in AGENTS.md.

## 9. Communication and Reporting

Be concise and technically precise.

Clearly distinguish:

- Verified implementation behavior.
- Developer-confirmed design intent.
- Inferences and assumptions.
- Unresolved questions.

Avoid repeating previously established findings unless new evidence
changes their interpretation.

For code changes, report:

1. What changed.
2. What was verified.
3. Any remaining concerns.

Do not initiate additional investigations or modifications outside
the requested scope.

When uncertain about architectural intent, ask rather than inventing
an explanation.

The objective is to help maintain and evolve RunWeaver while
preserving the flexibility and independence for which it was designed.
