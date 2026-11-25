# OpenMP matrix multiplication study

How much faster does n × n matrix multiplication get with OpenMP threads, and what stops it scaling? I measured three loop orders (`ijk`, `ikj` and cache-blocked tiles) on a 16-core, 32-thread AMD Ryzen 9 8945HX, fitted Amdahl's law and the Karp–Flatt metric to the results, and wrote a small `/proc` tool that draws a process's memory map while the benchmark runs. This was my project for the Operating Systems course at Kasetsart University, first semester of 2026.

## Results

Over 363 runs (1 to 32 threads, n from 100 to 6400, plus tile sizes, OpenMP schedules, gcc flags, CPU affinity, power profiles and a competing job):

- Memory access order mattered more than thread count. At n = 3200 a single thread took 57.4 s with `ijk`, 11.4 s with `ikj` and 5.6 s with blocking.
- The best speedup was 14.0× (`ikj`, n = 3200, 24 threads). Blocking reached 13.5× at n = 6400 on all 32 threads.
- The fitted serial fraction ranges from 0.029 (blocked, n = 6400) to 0.065 (`ikj`, n = 3200), which caps the speedup near 35× and 15×. Some of that is parallel overhead, not serial code.
- SMT adds little: for blocked n = 6400, going from 16 to 32 threads raised the speedup from 11.5× to 13.5× while the Karp–Flatt fraction rose from 0.026 to 0.044.
- Small matrices don't pay for threads. With `ijk` at n = 100, 16 threads gave 3.55×, and 32 threads ran about 11 times slower than one.
- Tile size and compiler flags matter: at n = 3200 on 16 threads, 192 × 192 tiles gave 200.4 GFLOP/s against 40.3 for 8 × 8, and at n = 1600 `ikj` went from 14.5 GFLOP/s at `-O0` to 203.1 at `-O3 -march=native`.

The full write-up with every figure is [report/report.pdf](report/report.pdf).

## Layout

| Folder | Contents |
|---|---|
| `parallel/` | the OpenMP program and its Makefile |
| `mmap/` | `mmap.py`, the memory map viewer (standard library only, Linux) |
| `bench/` | `run_bench.py` runs the experiments, `analyze.py` draws `figures/` and writes `data/summary.md` |
| `data/` | measured results (CSV), the run log and the machine description |
| `report/` | the report source (Markdown, built with pandoc and XeLaTeX), `numbers.md` and the PDF |

## Running it

```bash
make -C parallel
./parallel/parallel -n 800 -t 8 -a ikj
./parallel/parallel --help

python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python bench/run_bench.py --exp all --dry-run   # count the runs
.venv/bin/python bench/run_bench.py --exp scaling         # rows already in data/ are skipped
.venv/bin/python bench/analyze.py

python3 mmap/mmap.py --self
python3 mmap/mmap.py --run --html run.html -- ./parallel/parallel -n 3000 -t 8 --hold 3
```

The full benchmark takes about an hour; delete an experiment's CSV to measure it again. Rebuilding the report needs XeLaTeX, latexmk, biber and the TeX Gyre Pagella font: `.venv/bin/python report/make_numbers.py && .venv/bin/python report/build_report.py`.

`parallel` refuses to start if the matrices would need more than half the free memory or more threads than there are logical CPUs. `mmap.py` only reads `/proc`.
