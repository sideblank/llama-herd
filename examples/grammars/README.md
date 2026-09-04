# Grammars

GBNF grammars for constraining a stream's output to a shape that code can merge.

Pass one as `"grammar"` on a chat completion. It is compiled against the model's vocabulary and
masks tokens that cannot appear next, so the response is **structurally valid by construction** —
not valid-if-the-model-cooperates.

```bash
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d "$(python3 - <<'PY'
import json
print(json.dumps({
  "model": "chat",
  "messages": [{"role": "user", "content": "System A uses port 8080.\n\nExtract as JSON."}],
  "grammar": open("examples/grammars/chunk-digest.gbnf").read(),
  "temperature": 0, "max_tokens": 60,
}))
PY
)"
```

Measured on a 0.5B, two chunks, no retries and no repair:

```
{"system": "System A", "port": 8080}
{"system": "System B", "port": 9090}
```

## Why this matters for merging many streams

A herd runs dozens of streams at once. Merging their output means either code parses it or a model
reconciles it — and asking a model costs another generation and can disagree with itself.

A grammar makes the first option safe. Forty-eight schema-valid objects combine deterministically,
in under a millisecond, with no parse that can fail on the forty-seventh reply.

## Writing them

**Load from a file, do not interpolate through a shell.** A GBNF is full of quotes and backslashes,
and passing it through shell then JSON quoting corrupts it — which shows up as
`grammar failed to parse`, an error about the grammar rather than about the quoting that broke it.

**Constrain tightly.** The point is not merely valid JSON; it is a shape the merge already knows.
`[0-9]+` for a port beats a general `number` that could return a float.

**Keep it small.** Sixteen to thirty tokens per stream is the target. The grammar is what makes that
safe: a short response cannot trail off mid-object.

**A malformed grammar is refused at chain construction**, not ignored. Silently dropping it would
let generation proceed and produce plausible output whose shape is wrong — discovered downstream,
far from the mistake.
