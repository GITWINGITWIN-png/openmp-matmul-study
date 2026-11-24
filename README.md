# OpenMP matrix multiplication study

How fast does n × n matrix multiplication get with OpenMP, and what stops it from scaling? This repo has a C program with three loop orders, a harness that sweeps threads, matrix size, program settings and OS settings, an Amdahl and Karp–Flatt analysis of the measurements, and a small `/proc` tool that draws a process's memory map while the benchmark runs.

Written for the Operating Systems course at Kasetsart University, first semester of the 2026 academic year. **For educational purposes only: this is not production software.**

## What was measured

- **Machine:** AMD Ryzen 9 8945HX (16 cores, 32 hardware threads), Ubuntu 26.04 (Linux 7.0), gcc 15.2 with `-O3 -march=native -fopenmp`.
- **Program:** `parallel/parallel.c` computes C = A × B with the `ijk` or `ikj` loop order or with cache-blocked tiles. It times each phase, reports GFLOP/s and memory use, and checks random entries against a serial dot product.
- **Runs:** 363, over 1 to 32 threads, n from 100 to 6400, tile sizes, OpenMP schedules, gcc optimisation levels, thread affinity, CPU sets, power profiles and a competing job. The raw results are in `data/`.
- **Method:** speedup S(p) = T(1)/T(p) on whole-program time, a least-squares Amdahl fit of the serial fraction over 2 to 16 threads, and the Karp–Flatt serial fraction at each thread count.

## Key results

- **Memory access order mattered more than threads.** At n = 3200 one thread took 57.4 s with `ijk`, 11.4 s with `ikj` and 5.6 s with blocking, about 10 times apart.
- **Best speedup: 14.0×** (`ikj`, n = 3200, 24 threads). Blocking reached 13.5× at n = 6400 with all 32 threads.
- **Amdahl:** the fitted serial fraction is between 0.029 (blocked, n = 6400) and 0.065 (`ikj`, n = 3200), a ceiling of about 35× and 15×. Part of it is parallel overhead rather than truly serial code.
- **Past 16 cores:** for blocked n = 6400, doubling from 16 to 32 threads raised the speedup only from 11.5× to 13.5×, and the Karp–Flatt fraction rose from 0.026 to 0.044.
- **Small matrices do not pay for threads:** with `ijk` at n = 100, 16 threads gave 3.55×, but 32 threads ran about 11 times slower than one.
- **Settings matter:** at n = 3200 with 16 threads, 192 × 192 tiles gave 200.4 GFLOP/s against 40.3 with 8 × 8 tiles. At n = 1600 with 16 threads, `ikj` went from 14.5 GFLOP/s at `-O0` to 203.1 at `-O3 -march=native`.

The full analysis with every figure is in [report/report.pdf](report/report.pdf) (in Thai).

## Layout

| Folder | What |
|---|---|
| `parallel/` | the OpenMP program and its Makefile |
| `mmap/` | `mmap.py`, the memory-map viewer (Python standard library, Linux only) |
| `bench/` | `run_bench.py` runs the experiments; `analyze.py` draws the figures and writes `data/summary.md` |
| `data/` | measured results (CSV), the run log and the machine description |
| `figures/` | the plots |
| `report/` | the report source (Markdown built with pandoc and XeLaTeX), its numbers file and the PDF |

## Run it

```bash
make -C parallel                          # builds parallel/parallel
./parallel/parallel -n 800 -t 8 -a ikj    # one run, human-readable
./parallel/parallel --help

python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python bench/run_bench.py --exp all --dry-run   # count the runs
.venv/bin/python bench/run_bench.py --exp scaling         # one experiment; rows already in data/ are skipped
.venv/bin/python bench/analyze.py                         # figures/ and data/summary.md

python3 mmap/mmap.py --self               # memory map of the tool itself
python3 mmap/mmap.py --run --html run.html -- ./parallel/parallel -n 3000 -t 8 --hold 3
```

The whole benchmark takes about an hour; to measure an experiment again from scratch, delete its CSV first. Rebuilding the report needs XeLaTeX, latexmk, biber and the Sarabun font: `.venv/bin/python report/make_numbers.py && .venv/bin/python report/build_report.py`.

`parallel` refuses to start if the matrices would need more than half of the available memory, or if asked for more threads than logical CPUs. `mmap.py` only reads `/proc`.

Built with help from AI coding assistants.
