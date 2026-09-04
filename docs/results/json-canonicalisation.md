# Does flattening JSON save prefill tokens?

**Measured 2026-08-23.** Short answer: **no — the flattening itself saves nothing.** What saves
tokens is compacting the whitespace (free, lossless, one stdlib call) and abbreviating repeated key
names (real, smaller than it looks, and not free).

## Method

4,000 synthetic symbol records — the shape a code-analysis payload actually has (`file_path`,
`symbol_type`, `symbol_name`, `is_exported`, `line_number`, `package_name`), 236,126 tokens
pretty-printed, so within a few percent of the 256k case under discussion.

Tokenised with the real **qwen3.5 vocabulary** (`ggml-vocab-qwen35.gguf`), vocab-only load, via
`llama-herd canon --json`. Counts are exact, not estimated.

## Result

Against a **pretty-printed** payload:

| form | chars | tokens | char cut | token cut |
|---|---|---|---|---|
| original JSON | 735,766 | 236,126 | — | — |
| `json.Compact` | 567,765 | 152,127 | 22.8% | **35.6%** |
| flattened | 466,565 | 152,123 | 36.6% | **35.6%** |
| flattened + abbreviated keys | 234,565 | 120,123 | 68.1% | **49.1%** |

Against a **compact** payload — the control that matters, because a real API or database dump is
rarely pretty-printed:

| form | chars | tokens | char cut | token cut |
|---|---|---|---|---|
| original (compact) JSON | 567,765 | 152,127 | — | — |
| flattened | 466,565 | 152,123 | 17.8% | **0.0%** |
| flattened + abbreviated keys | 234,565 | 120,123 | 58.7% | **21.0%** |

## What this means

**Flattening JSON syntax saves 0.0% of tokens.** 152,123 against 152,127 — four tokens in 152,000 —
while cutting 17.8% of the characters. Replacing `{`, `}`, `"`, `:`, `,` with `|` and `:` is a wash,
because a BPE vocabulary already carries merged tokens for the common JSON punctuation sequences.
`","` and `":"` are single tokens; so are `|` and `:`.

**The 35.6% that flattening appears to win is the pretty-printer's whitespace**, and `json.Compact`
takes all of it — losslessly, reversibly, with no format change, no escaping rules, no lost nesting,
and one line of standard library.

**Abbreviating keys is the only part that works: 21.0%.** And even there the character saving
(58.7%) is **2.8× the token saving**, for the same reason — `file_path` is a handful of tokens and
`f` is one, so shortening it does not save proportionally.

Abbreviation is also not free. The schema header that decodes it is prepended to every stream, so it
is multiplied by the stream count. At 48 streams a ~200-token header costs ~9,600 tokens against
~32,000 saved — still positive, and roughly a third of the benefit gone.

## The general lesson

**Character count is not a proxy for token count, and reasoning about token savings from character
savings will be wrong in the direction that flatters the change.** Every row above where the two
columns diverge is a case where arithmetic said one thing and the tokenizer said another.

The design being tested estimated 20–30% and "~23 seconds off a 78 s prefill" from replacing JSON
syntax. Against a compact payload the true figure for that specific change is 0%. The estimate was
not careless — it was the natural inference from character counts, and it happens to be wrong.

This is the same trap that `collapse-inline-space` fell into in the text canonicalisation work,
where a pass that plainly removes characters measured zero effect on real text.

## Recommendation

1. **`json.Compact` always.** Free, lossless, reversible, and on a pretty payload it is the entire
   win.
2. **Abbreviate keys only if measured on your payload**, and cost the schema header at ×streams.
3. **Do not flatten for token savings.** It buys nothing and costs delimiter-escaping bugs, refused
   nesting, and a lossy form that cannot round-trip.

Sub-tree partitioning at element boundaries is unaffected by any of this and remains correct — that
is about not slicing a record in half, not about token count.

## Reproduce

```
llama-herd canon --json --model /path/ggml-vocab-qwen35.gguf --doc payload.json
```
