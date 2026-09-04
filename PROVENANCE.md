# Provenance and chain of title

This records how llama-herd's code came to exist and who held the rights to license it, so the
project's licensing rests on a documented basis rather than assumption.

## This is a rewrite, not a port

**No code is copied into this repository from any other codebase.** llama-herd is written fresh,
including its binding to llama.cpp.

The author previously built a runtime of similar architecture as part of a private platform. That
work is not imported here — not its source, and not its history. The prior system carries
integrations specific to that platform (job-queue pull workers, storage and dispatch coupling)
that are deliberately absent from this project and will not appear in it.

What carries over is **architecture and knowledge, not code**: the weight-shared, single-decode-loop,
per-sequence-KV design described in the README, and the performance bar it demonstrated.

The two are not subsets of one another. llama-herd omits the platform-specific integrations of the
prior system, and it also carries **capabilities the private system does not have**. This is a
distinct project that diverges in both directions, not a stripped-down edition of something else.

That divergence supports treating llama-herd as its own work, but it does not by itself escape the
future-works clause discussed below, which is drafted broadly enough to reach anything that
"relates to" the assigned property. Sequencing, not divergence, is the answer there.

## Authorship

All code in this repository is authored by **Benjamin Goldman** and by contributors who sign off
under the [DCO](.github/DCO).

No employer, contractor, joint author, or third-party contributor holds an interest in the
originally authored code, and it was not created under a work-made-for-hire arrangement.

Because the author is reimplementing an architecture he designed and owns, no clean-room procedure
applies or is required: a clean-room separation exists to avoid copying **another party's**
protected expression, and there is no other party here.

## Rights at the time of release

As of the effective date below, copyright in the originally authored code is held **personally by
Benjamin Goldman**:

- The intellectual-property assignment to Liminal Intelligence is **drafted but not executed**.
- The related IP licence agreement is likewise **drafted but not executed**.
- That assignment's schedule of assigned property covers a **different repository** and does not
  name this one.

No assignment or exclusive licence having been executed, the copyright is free of any competing
claim and may be licensed on any terms without a counterparty's consent.

## Release

Benjamin Goldman, as author and copyright holder, licenses the code in this repository under the
**Apache License 2.0** (see [LICENSE](LICENSE)).

**Effective date:** 2026-09-04

**Signature:** Benjamin Goldman

## Sequencing: the future-works clause

The drafted assignment contains a **future-works clause** assigning, automatically upon creation,
every work that "relates to, is derived from, or improves upon" the assigned property. A runtime of
this architecture falls squarely within that description. Two consequences:

1. **Work written before that assignment is executed** is not a future work relative to it, and is
   not named in its schedule. It stays personally owned.
2. **Work written after it is executed** would assign automatically to the assignee. The Apache-2.0
   grant already made is irrevocable and survives the transfer — an assignee takes the copyright
   subject to it — but ownership of subsequent commits would sit with the assignee, and this
   document and [NOTICE](NOTICE) would need updating to say so.

If the assignment is executed, either **carve this project out of it explicitly**, or record the
Apache-2.0 licence as an existing encumbrance in its schedule. Left unaddressed, the schedule
becomes inaccurate on a document intended to support a federal copyright recordation.

## Contributions

Contributions are licensed to the project under Apache-2.0 by operation of Apache-2.0 §5, and are
certified by a [DCO](.github/DCO) sign-off on every commit.

There is no CLA and no copyright assignment: **contributors keep the copyright in their own work.**
The project is a collective work — each contributor holds rights in their own contribution only,
and contributing conveys no ownership of, or right to control, the codebase as a whole. The grant
is irrevocable; a merged contribution cannot later be withdrawn.

## Third-party components

Third-party code carries its own licence and is inventoried in [NOTICE](NOTICE). llama-herd links
llama.cpp, which is MIT-licensed. Model weights published alongside this project carry the licence
of their upstream base model, which is not Apache-2.0 and may impose additional restrictions.

## Status

This document is a **factual record prepared by the project, not legal advice**, and has not been
reviewed by counsel. It was signed and dated on 2026-09-04. The facts stated about the
unexecuted assignment and licence were verified on 2026-08-21 and re-affirmed by the author at
signing.
