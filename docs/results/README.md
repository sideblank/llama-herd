# Measured results

Each file records what was measured, on what hardware, with which model and settings, and how.
A figure without those is a claim, not a result. See [../BENCHMARKING.md](../BENCHMARKING.md)
for what each measurement means and how to reproduce it.

| File | What it records |
|---|---|
| [3090.md](3090.md) | Aggregate decode on one RTX 3090 with `Qwen3.6-35B-A3B-UD-IQ3_S`: 728.71 tok/s at 48 streams, the stream curve, the node-to-node variance, and what killed the process |
| [json-canonicalisation.md](json-canonicalisation.md) | Whether flattening a JSON payload saves prefill tokens (it does not; compacting whitespace does) |

The manifest that produced the 3090 figure is
[../../examples/3090-throughput.json](../../examples/3090-throughput.json).

Every table in a results file comes from **one boot on one node**. One configuration has
measured 42 and 118 tok/s on two rented 3090s, and the library's own `tg128` on the same model
ranged from 129 to 145 across the nodes used. A figure taken on one machine and a comparison
figure taken on another cannot support a conclusion about either.
