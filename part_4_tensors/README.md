# Part IV — Plain Python versus tensors

Run `python -m part_4_tensors.benchmark` from the repository root.
The benchmark copies Part I's weights into float64 CPU tensors and compares the
same 32–64–16 sigmoid forward pass. It checks numerical equality, warms up five
times, alternates timing order and records 15 repetitions. Tensor construction is
outside the timed region; PyTorch uses one CPU thread.

`results/benchmark.json` records approximately **6.46×** speedup for one row and
**229.87×** for 256 rows. Numerical error was below `1e-12`. These local timings
depend on hardware, load, batch size and allocation costs. Other work was running
on the machine during measurement. This does not benchmark training or conversion.

Python loops operate on individual objects. PyTorch batches arithmetic in native
kernels over numeric storage, amortizing dispatch overhead. Smaller operations,
especially including conversion, need not show the same advantage.

A tensor is an array with axes: a scalar has zero axes, a vector one, a matrix
two. Inputs here have shape `[batch, 32]`. A token is a discrete unit with an ID.
An embedding is a learned vector for an ID; the integer ID itself has no semantic
distance. See the [PyTorch tensor documentation](https://docs.pytorch.org/docs/stable/tensors.html).
