# Security Policy

## Reporting a vulnerability

**Do not open a public issue for a security problem.**

Report privately through GitHub's [private vulnerability reporting][pvr] on this repository, or by
email to `benjamin.goldman@gmail.com`.

Please include: what you found, the affected version or commit, how to reproduce it, and what an
attacker gains. We aim to acknowledge within 3 business days and to ship a fix or a documented
mitigation within 90 days, crediting you unless you prefer otherwise.

[pvr]: https://github.com/sideblank/llama-herd/security/advisories/new

## Scope

This project is an inference runtime: it loads model files and serves inference over HTTP. Things
we consider vulnerabilities:

- Memory-safety or parsing bugs reachable from a **model file** (a malicious `.gguf` or `.onnx`
  causing out-of-bounds access, overflow, or arbitrary code execution on load).
- **Cross-stream leakage** — one request's KV cache, tokens, or logits becoming visible to another
  concurrent stream. Per-sequence isolation is a security property here, not just a correctness one.
- Unauthenticated paths that read or write outside the configured model directory.
- Remote crash or unbounded resource consumption triggerable by a well-formed request.

Out of scope:

- Model *outputs* — hallucination, jailbreaks, or prompt injection. These are model behaviour, not
  runtime vulnerabilities.
- Resource exhaustion from a deliberately oversized context you configured yourself.
- Anything requiring an attacker to already have local access to the host or the weights.

## Operational note

The runtime performs **no authentication or authorization of its own**. Do not expose it directly
to an untrusted network — put it behind a gateway you control. Treat model files like executables:
only load weights you obtained from a source you trust.
