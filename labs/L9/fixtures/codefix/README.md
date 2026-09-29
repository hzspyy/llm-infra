# codefix fixture (L9.1)

A tiny, self-contained Python package used by the L9.1 agent trace collection as the
"read a small repo and fix it" task class.

- `textstat/` — three modules with 14 pure functions.
- `tests/test_textstat.py` — a `unittest` suite that pins the documented behaviour.

Run the suite from the repo root:

```bash
python -m unittest discover -s tests -v
```

The agent harness copies this tree into a scratch directory, applies exactly one
mutation (a single operator, constant or boundary change), and asks the model to find
and repair it with file-read/write tools. The suite is read-only for the agent: the
harness rejects any write under `tests/`.
