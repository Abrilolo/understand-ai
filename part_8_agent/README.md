# Part VIII — A local model with tools

Start Ollama with `qwen2.5-coder:3b` installed (`ollama pull qwen2.5-coder:3b` if
needed). This machine already had it. Run `python -m part_8_agent.agent`
from the repository root. Model weights are external to this repository.

The model chooses a file read, arithmetic calculation or final answer through
an explicit JSON action protocol. Python validates, executes and returns results.
The successful `results/transcript.json` reads fictional `data/order.txt`, computes
`17 * 23.75 = 403.75`, then `403.75 * 0.88 = 355.3`, and answers **$355.30**. Without
the reader the model cannot access this local order. The final sentence omitted
the requested working; the arithmetic remains visible in the tool trace.

Both installed coder models initially produced ordinary JSON-looking text instead
of native API tool calls. `native_3b_failure.json` and `native_7b_failure.json`
preserve these failures. Their legacy `answered` status only means a final message
was returned: empty `events` proves no tools ran. Structured JSON actions resolved
the interoperability problem in this demo; the model still chose each tool.

The reader confines resolved paths to the data folder and caps file size. The
calculator accepts arithmetic AST nodes, never `eval`, names or function calls.
The loop has a round limit and returns errors. These controls constrain this demo;
they do not establish reliability across tasks or solve prompt injection generally.
Arithmetic uses Python numeric types, including ordinary floating-point rounding.

Reference: [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling).
