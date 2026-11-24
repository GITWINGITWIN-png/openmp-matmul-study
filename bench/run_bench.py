#!/usr/bin/env python3
"""รัน benchmark ของ parallel/parallel.c แล้วเขียนผลลง data/<exp>.csv

การทดลอง (ตรงกับข้อ 2a–2e ของโจทย์):
  scaling   a+b  threads × n × algo — ข้อมูลหลักสำหรับ speedup / Amdahl / หน่วยความจำ (2e)
  block     c    ขนาด tile ของ blocked
  schedule  c    OpenMP loop schedule (static / dynamic / guided + chunk)
  compiler  c    ระดับ optimisation ของ gcc (-O0 … -O3 -march=native)
  os        d    affinity (OMP_PROC_BIND/OMP_PLACES), taskset (SMT, CCD), power profile,
                 OMP_STACKSIZE, การแย่ง CPU กับโปรเซสอื่น

วิธีใช้ (จากโฟลเดอร์บนสุดของ repo):
  .venv/bin/python bench/run_bench.py --exp all            # รันทั้งหมด (~1 ชม.)
  .venv/bin/python bench/run_bench.py --exp scaling,block   # เลือกบางชุด
  .venv/bin/python bench/run_bench.py --exp all --dry-run   # แค่นับจำนวนรัน

รันซ้ำได้: แถวที่มีใน CSV แล้วจะถูกข้าม (ถ้าอยากรันใหม่ ลบ CSV ของชุดนั้นทิ้ง)
DO NO HARM: ทุกรันมี timeout · power profile ถูกคืนค่าเดิมใน finally · ไม่ใช้ sudo
"""
import argparse
import csv
import datetime as dt
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "parallel/parallel.c"
BUILD_DIR = ROOT / "bench/build"
DATA_DIR = ROOT / "data"

THREADS = [1, 2, 4, 8, 12, 16, 24, 32]
SIZES = [100, 200, 400, 800, 1600, 3200]
COOLDOWN_S = 0.5
RUN_TIMEOUT_S = 900

BUILDS = {
    "O0": "-O0", "O1": "-O1", "O2": "-O2", "O3": "-O3",
    "O3native": "-O3 -march=native",
}


# ----------------------------------------------------------------- system helpers

def find_cpu_temp_file():
    for h in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (h / "name").read_text().strip() in ("k10temp", "coretemp"):
                return h / "temp1_input"
        except OSError:
            pass
    return None


TEMP_FILE = find_cpu_temp_file()


def cpu_temp():
    try:
        return int(TEMP_FILE.read_text()) / 1000 if TEMP_FILE else ""
    except OSError:
        return ""


def power_profile_get():
    if not shutil.which("powerprofilesctl"):
        return ""
    r = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True)
    return r.stdout.strip()


def power_profile_set(name):
    r = subprocess.run(["powerprofilesctl", "set", name], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"powerprofilesctl set {name} failed: {r.stderr.strip()}")
    time.sleep(2)


def build(tag):
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    exe = BUILD_DIR / f"parallel_{tag}"
    if not exe.exists() or exe.stat().st_mtime < SRC.stat().st_mtime:
        cmd = ["gcc", *BUILDS[tag].split(), "-fopenmp", "-std=c11", "-o", str(exe), str(SRC), "-lm"]
        print("  build:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    return exe


def write_sysinfo():
    out = DATA_DIR / "sysinfo.txt"
    parts = [f"date: {dt.datetime.now().isoformat(timespec='seconds')}",
             f"kernel: {platform.release()}", f"python: {platform.python_version()}"]
    for title, cmd in [("gcc", ["gcc", "--version"]), ("lscpu", ["lscpu"]),
                       ("topology", ["lscpu", "-e=CPU,CORE,CACHE"]),
                       ("memory", ["free", "-h"]), ("power profile", ["powerprofilesctl", "get"]),
                       ("stack limit (ulimit -s, KB)", ["bash", "-c", "ulimit -s"])]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            parts.append(f"\n## {title}\n{r.stdout.strip()}")
        except FileNotFoundError:
            pass
    for f in ["/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
              "/sys/devices/system/cpu/cpufreq/boost"]:
        try:
            parts.append(f"{f}: {Path(f).read_text().strip()}")
        except OSError:
            pass
    out.write_text("\n".join(parts) + "\n")


# ----------------------------------------------------------------- one run

class Run:

    def __init__(self, exp, label, n, threads, algo="ikj", block=64, reps=5, build="O3native",
                 schedule=None, bind=None, places=None, stacksize=None, cpus=None,
                 power=None, background=None):
        self.exp, self.label = exp, label
        self.n, self.threads, self.algo, self.block, self.reps = n, threads, algo, block, reps
        self.build, self.schedule, self.bind, self.places = build, schedule, bind, places
        self.stacksize, self.cpus, self.power, self.background = stacksize, cpus, power, background

    def key(self):
        return "|".join(str(x) for x in (self.label, self.n, self.threads, self.algo,
                                         self.block if self.algo == "blocked" else 0, self.build))

    def command(self):
        cmd = [str(build(self.build)), "-n", str(self.n), "-t", str(self.threads),
               "-a", self.algo, "-b", str(self.block), "-r", str(self.reps), "--csv"]
        if self.cpus:
            cmd = ["taskset", "-c", self.cpus] + cmd
        return cmd

    def env(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OMP_", "GOMP_"))}
        for var, val in [("OMP_SCHEDULE", self.schedule), ("OMP_PROC_BIND", self.bind),
                         ("OMP_PLACES", self.places), ("OMP_STACKSIZE", self.stacksize)]:
            if val:
                env[var] = val
        return env

    def execute(self):
        bg = None
        if self.background:
            bg = subprocess.Popen(self.background, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, env=self.env())
            time.sleep(1)
        t_before = cpu_temp()
        try:
            r = subprocess.run(self.command(), capture_output=True, text=True,
                               env=self.env(), timeout=RUN_TIMEOUT_S)
        finally:
            if bg:
                bg.terminate()
                bg.wait(timeout=10)
        if r.returncode != 0:
            raise RuntimeError(f"run failed ({r.returncode}): {r.stderr.strip()}")
        header = subprocess.run([str(build(self.build)), "--header"], capture_output=True,
                                text=True).stdout.strip().split(",")
        row = dict(zip(header, r.stdout.strip().split(",")))
        if row["ok"] != "1":
            raise RuntimeError(f"WRONG RESULT: {row}")
        row.update(exp=self.exp, label=self.label, build=self.build,
                   omp_proc_bind=self.bind or "", omp_places=self.places or "",
                   omp_stacksize=self.stacksize or "", cpus=self.cpus or "all",
                   power_profile=self.power or power_profile_get(), temp_before=t_before,
                   temp_after=cpu_temp(), timestamp=dt.datetime.now().isoformat(timespec="seconds"))
        return row


# ----------------------------------------------------------------- experiments

def exp_scaling():
    runs = []
    for algo in ["ijk", "ikj", "blocked"]:
        for n in SIZES:
            for t in THREADS:
                reps = 3 if (algo == "ijk" and n >= 3200 and t <= 2) else 5
                runs.append(Run("scaling", "default", n, t, algo, reps=reps))
    for t in THREADS:
        runs.append(Run("scaling", "default", 6400, t, "blocked", reps=3))
    return runs


def exp_block():
    return [Run("block", f"b{b}", n, t, "blocked", block=b, reps=3)
            for n in [1600, 3200] for t in [1, 16]
            for b in [8, 16, 32, 48, 64, 96, 128, 192, 256, 512]]


def exp_schedule():
    schedules = ["static", "static,1", "static,16", "dynamic,1", "dynamic,16", "guided", "guided,16"]
    return [Run("schedule", s, n, t, algo, schedule=s)
            for algo in ["ikj", "blocked"] for n in [400, 1600, 3200] for t in [16, 32]
            for s in schedules]


def exp_compiler():
    return [Run("compiler", tag, 1600, t, algo, build=tag, reps=3)
            for tag in BUILDS for algo in ["ijk", "ikj", "blocked"] for t in [1, 16]]


def exp_os():
    runs = []
    # (1) affinity
    affinity = [("default", None, None), ("close-cores", "close", "cores"),
                ("spread-cores", "spread", "cores"), ("close-threads", "close", "threads"),
                ("spread-threads", "spread", "threads")]
    for label, bind, places in affinity:
        for algo in ["ikj", "blocked"]:
            for t in [8, 16, 32]:
                runs.append(Run("os", f"bind:{label}", 3200, t, algo, bind=bind, places=places))
    # (2) taskset
    cpusets = [("16cores-noSMT", "0-15", 16), ("8cores-SMT", "0-7,16-23", 16),
               ("1CCD-8cores", "0-7", 8), ("2CCD-4+4cores", "0-3,8-11", 8)]
    for label, cpus, t in cpusets:
        for algo in ["ikj", "blocked"]:
            runs.append(Run("os", f"taskset:{label}", 3200, t, algo, cpus=cpus))
    # (3) OMP_STACKSIZE
    for ss in ["1M", "8M", "64M"]:
        runs.append(Run("os", f"stack:{ss}", 1600, 32, "blocked", stacksize=ss))
    # (4) แย่ง CPU
    bg = [str(build("O3native")), "-n", "3200", "-t", "16", "-a", "ijk", "-r", "20"]
    for t in [16, 32]:
        runs.append(Run("os", "contention:none", 3200, t, "blocked"))
        runs.append(Run("os", "contention:bg16", 3200, t, "blocked", background=bg))
    # (5) power profile
    for prof in ["performance", "balanced", "power-saver"]:
        for algo in ["ikj", "blocked"]:
            for t in [1, 16]:
                runs.append(Run("os", f"power:{prof}", 3200, t, algo, power=prof, reps=3))
    return runs


EXPERIMENTS = {"scaling": exp_scaling, "block": exp_block, "schedule": exp_schedule,
               "compiler": exp_compiler, "os": exp_os}


# ----------------------------------------------------------------- driver

def run_experiment(name, dry_run):
    runs = EXPERIMENTS[name]()
    out = DATA_DIR / f"{name}.csv"
    done = set()
    if out.exists():
        with out.open() as f:
            for row in csv.DictReader(f):
                done.add("|".join([row["label"], row["n"], row["threads"], row["algo"],
                                   row["block"], row["build"]]))
    todo = [r for r in runs if r.key() not in done]
    print(f"\n== {name}: {len(runs)} runs, {len(runs) - len(todo)} already done, {len(todo)} to go")
    if dry_run or not todo:
        return

    original_profile = power_profile_get()
    try:
        for i, run in enumerate(todo, 1):
            if run.power and run.power != power_profile_get():
                power_profile_set(run.power)
            try:
                row = run.execute()
            except Exception as e:
                print(f"  [{i}/{len(todo)}] {run.key()}  FAILED: {e}", file=sys.stderr)
                continue
            new_file = not out.exists()
            with out.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new_file:
                    w.writeheader()
                w.writerow(row)
            print(f"  [{i}/{len(todo)}] {run.key():<40} mult={float(row['t_mult_med']):9.4f}s "
                  f"{float(row['gflops']):7.1f} GF/s  {row['temp_before']}→{row['temp_after']}°C",
                  flush=True)
            time.sleep(COOLDOWN_S)
    finally:
        if original_profile and power_profile_get() != original_profile:
            power_profile_set(original_profile)
            print(f"  power profile restored to '{original_profile}'")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="all", help="comma list: " + ",".join(EXPERIMENTS) + " or all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    names = list(EXPERIMENTS) if args.exp == "all" else args.exp.split(",")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        write_sysinfo()
    for name in names:
        run_experiment(name, args.dry_run)


if __name__ == "__main__":
    main()
