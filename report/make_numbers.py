#!/usr/bin/env python3
"""สร้าง report/numbers.md — คลังตัวเลขทั้งหมดสำหรับเขียนรายงาน (สร้างจาก CSV ผลจริง)"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "report/numbers.md"

D = {k: pd.read_csv(DATA / f"{k}.csv") for k in ["scaling", "block", "schedule", "compiler", "os"]}
S = D["scaling"]
ALGOS = ["ijk", "ikj", "blocked"]
lines = []


def h(text):
    lines.extend(["", text, ""])


def p(text=""):
    lines.append(text)


def table(df, floatfmt=".2f", index=True):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(str)
    lines.append(df.to_markdown(floatfmt=floatfmt, index=index))
    lines.append("")


def source(text):
    lines.append(f"<sub>ที่มา: {text}</sub>")
    lines.append("")


def pct(new, old):
    return f"{100 * (new - old) / old:+.0f}%"


def amdahl_fit(p_, s_):
    p_, s_ = np.asarray(p_, float), np.asarray(s_, float)
    m = p_ > 1
    x, y = 1 - 1 / p_[m], 1 / s_[m] - 1 / p_[m]
    f = float((x * y).sum() / (x * x).sum())
    pred = 1 / (f + (1 - f) / p_[m])
    ss_tot = ((s_[m] - s_[m].mean()) ** 2).sum()
    r2 = 1 - ((s_[m] - pred) ** 2).sum() / ss_tot if ss_tot > 0 else float("nan")
    return f, r2


S = S.copy()
t1 = S[S.threads == 1].set_index(["algo", "n"])
S["S"] = [t1.t_total[(a, n)] / t for a, n, t in zip(S.algo, S.n, S.t_total)]
S["S_mult"] = [t1.t_mult_med[(a, n)] / t for a, n, t in zip(S.algo, S.n, S.t_mult_med)]
S["E"] = S.S / S.threads
S["KF"] = np.where(S.threads > 1, (1 / S.S - 1 / S.threads) / (1 - 1 / S.threads), np.nan)
S["serial_share"] = (S.t_init + S.t_verify) / S.t_total

# =====================================================================================
p("# คลังตัวเลขสำหรับเขียนรายงาน OS Report 2026")
p()
p("> สร้างอัตโนมัติโดย `report/make_numbers.py` จาก `data/*.csv` — **อย่าแก้ด้วยมือ** (รันสคริปต์ใหม่แทน)")
p("> หัวข้อเลขตรงกับ `report_outline.md` / `report.md` · ไฟล์นี้มีแต่ข้อมูล: เลือกตัวเลขที่ต้องการแล้วเขียนอธิบายเอง")
p("> ปัดเศษในตาราง: ใช้ตามที่แสดงได้เลย · ค่าละเอียดเต็มอยู่ใน CSV")
p()
p("**สารบัญ:** [นิยาม](#นิยามและหน่วย) · [2 สภาพแวดล้อม](#2-สภาพแวดล้อมการทดลอง) · "
  "[3 ความถูกต้อง](#3-ความถูกต้องและการทดสอบ) · [4 การทดลอง](#4-ภาพรวมการทดลอง) · "
  "[5.1 threads](#51-จำนวน-thread) · [5.2 n](#52-ขนาดเมทริกซ์-n) · [5.3 ตั้งค่าโปรแกรม](#53-การตั้งค่าโปรแกรม) · "
  "[5.4 ตั้งค่า OS](#54-การตั้งค่าระบบปฏิบัติการ) · [5.5 หน่วยความจำ](#55-หน่วยความจำ) · "
  "[5.6 Amdahl](#56-amdahl--karpflatt) · [5.7 ข้อจำกัด](#57-ข้อจำกัด--ความแกว่งของผล) · "
  "[6 mmap.py](#6-mmappy) · [7 ขนาดโค้ด](#7-ขนาดโค้ด)")

h("## นิยามและหน่วย")
table(pd.DataFrame([
    ["T(p)", "t_init + t_mult_med + t_verify", "เวลาทั้งโปรแกรม 1 รอบ (วินาที) ที่ p threads"],
    ["t_mult_med", "median ของการคูณ r รอบ", "เฉพาะส่วนขนาน"],
    ["S(p)", "T(1) / T(p)", "speedup ของทั้งโปรแกรม"],
    ["S_mult(p)", "t_mult(1) / t_mult(p)", "speedup เฉพาะส่วนคูณ"],
    ["E(p)", "S(p) / p", "efficiency (1.0 = ใช้ core คุ้มเต็มที่)"],
    ["GFLOP/s", "2n³ / t_mult_med / 10⁹", "n³ ครั้งของ คูณ+บวก = 2n³ floating-point operations"],
    ["f_fit", "least squares ของ 1/S − 1/p = f(1 − 1/p)", "serial fraction ตาม Amdahl (จากข้อมูล)"],
    ["f_meas", "(t_init + t_verify) / T(1)", "serial fraction ที่วัดตรงจากเวลาเฟส"],
    ["KF(p)", "(1/S − 1/p) / (1 − 1/p)", "Karp–Flatt: serial fraction ที่ประมาณได้ ณ p นั้น"],
    ["serial share", "(t_init + t_verify) / T(p)", "สัดส่วนเวลาของส่วน serial ณ p นั้น"],
], columns=["สัญลักษณ์", "สูตร", "ความหมาย"]), index=False)

# =====================================================================================
h("## 2 สภาพแวดล้อมการทดลอง")
table(pd.DataFrame([
    ["CPU", "AMD Ryzen 9 8945HX · 16 core / 32 thread · 2 CCD · max 5.46 GHz"],
    ["L1d / L2 / L3", "32 KB ต่อ core · 1 MB ต่อ core · 32 MB ต่อ CCD (รวม 64 MB)"],
    ["RAM", "30 GB · swap 8 GB"],
    ["OS / kernel", "Ubuntu 26.04.1 LTS · Linux 7.0.0-38-generic"],
    ["compiler", "gcc 15.2.0 · OpenMP runtime: libgomp"],
    ["glibc / Python", "2.43 / 3.14.4"],
    ["governor / boost / power profile", "performance / on / performance"],
    ["ไฟเลี้ยง", "AC (เสียบสาย)"],
    ["stack limit (ulimit -s)", "8192 KB = stack ของแต่ละ thread"],
    ["ผัง CPU", "CPU 0–15 = core จริง 0–15 · CPU 16–31 = SMT · core 0–7 = CCD0 · core 8–15 = CCD1"],
    ["Tjmax", "100 °C (อุณหภูมิสูงสุดที่ CPU ออกแบบให้ทำงาน)"],
], columns=["รายการ", "ค่า"]), index=False)
p("**n ที่ 3 เมทริกซ์ (3·n²·8 B) เท่ากับขนาด cache:**")
p()
table(pd.DataFrame([[name, size, f"{(b / 24) ** 0.5:,.0f}"] for name, size, b in [
    ("L1d (ต่อ core)", "32 KB", 32 << 10), ("L2 (ต่อ core)", "1 MB", 1 << 20),
    ("L3 (ต่อ CCD)", "32 MB", 32 << 20), ("L3 (รวม 2 CCD)", "64 MB", 64 << 20)]],
    columns=["ชั้น", "ขนาด", "n สูงสุดที่ยังพอดี"]), index=False)
source("`data/sysinfo.txt`, `lscpu`, `ldd --version`")

# =====================================================================================
h("## 3 ความถูกต้องและการทดสอบ")
allruns = sum(len(df) for df in D.values())
okruns = sum(int(df.ok.sum()) for df in D.values())
p(f"- รันที่ผ่านการตรวจ (ok = 1): **{okruns} / {allruns}**")
cs = S.groupby("n").checksum.nunique()
p(f"- checksum ต่างกันกี่ค่าต่อ n (ข้าม 3 algo × 8 threads): {', '.join(f'n={n}: {v}' for n, v in cs.items())}")
p("- เทียบกับ numpy (n = 7, 64, 129 · ทั้ง 3 algo): relative error 2.2e-12 – 3.2e-12 (มาจากการพิมพ์ checksum ทศนิยม 10 หลัก)")
p("- ค่า tolerance ที่ใช้ตรวจแต่ละจุด: relative 1e-9 · จุดที่สุ่มตรวจ: k = 64 (ค่าเริ่มต้น)")
p("- กรณีขอบที่ทดสอบ: n = 1 · n = 333 กับ b = 50 (tile ไม่ลงตัว) · n = 300 กับ t = 3, 32")
p("- guard ที่ทดสอบ: `-n 0` · `-t 999` · `-n 20000 -t 1` (ต้องใช้ 3 × 3200 MB, RAM ว่างตอนนั้น 18,686 MB, ขีด 50%) → ถูกปฏิเสธทั้งหมด")
p("- ค่าเริ่มต้นของ `schedule(runtime)` ใน libgomp เมื่อไม่ตั้ง `OMP_SCHEDULE` = dynamic,1 → โปรแกรมตั้งเป็น static เอง")
source("`data/*.csv` คอลัมน์ ok, checksum · การทดสอบด้วยมือ 2026-10-24")

# =====================================================================================
h("## 4 ภาพรวมการทดลอง")
rows = []
for k, df in D.items():
    ts = pd.to_datetime(df.timestamp)
    rows.append([k, len(df), f"{ts.min():%H:%M}–{ts.max():%H:%M}",
                 f"{(ts.max() - ts.min()).total_seconds() / 60:.0f}",
                 ", ".join(str(r) for r in sorted(df.reps.unique())),
                 f"{df.temp_before.min():.0f}–{df.temp_after.max():.0f}"])
table(pd.DataFrame(rows, columns=["ชุด", "รัน", "ช่วงเวลา (24 ต.ค.)", "นาที", "reps ต่อรัน", "Tctl °C"]), index=False)
p(f"- รวม {allruns} รัน · วัดซ้ำในโปรแกรม 3–5 รอบต่อรัน ใช้ median · พักระหว่างรัน 0.5 วินาที · timeout ต่อรัน 900 วินาที")
p("- threads ที่ใช้: 1, 2, 4, 8, 12, 16, 24, 32 · n: 100, 200, 400, 800, 1600, 3200 (+6400 เฉพาะ blocked) · tile เริ่มต้น b = 64")
source("`data/*.csv` คอลัมน์ timestamp, reps, temp_before, temp_after")

# =====================================================================================
h("## 5.1 จำนวน thread")
p("**Speedup ของทั้งโปรแกรม S(p)** (แถว = n, คอลัมน์ = p)")
for a in ALGOS:
    p()
    p(f"*{a}*")
    p()
    table(S[S.algo == a].pivot(index="n", columns="threads", values="S"))
p("**Efficiency E(p)**")
for a in ALGOS:
    p()
    p(f"*{a}*")
    p()
    table(S[S.algo == a].pivot(index="n", columns="threads", values="E"))
p("**เวลาส่วนคูณ t_mult_med (วินาที)**")
for a in ALGOS:
    p()
    p(f"*{a}*")
    p()
    table(S[S.algo == a].pivot(index="n", columns="threads", values="t_mult_med"), floatfmt=".4f")
p("**จุดสำคัญ**")
p()
best = S.loc[S.groupby(["algo", "n"]).S.idxmax()][["algo", "n", "threads", "S"]]
table(best.rename(columns={"threads": "p ที่ S สูงสุด", "S": "S สูงสุด"}), index=False)
slow = S[(S.threads > 1) & (S.S < 1)][["algo", "n", "threads", "S"]]
p("รันที่ **ช้ากว่า 1 thread** (S < 1):")
p()
table(slow, index=False)
g16 = S[S.threads.isin([16, 32])].pivot_table(index=["algo", "n"], columns="threads", values="S")
g16["เปลี่ยน 16→32"] = [pct(r[32], r[16]) for _, r in g16.iterrows()]
p("ผลของ SMT (16 → 32 threads):")
p()
table(g16)
source("`data/scaling.csv` (152 แถว)")

# =====================================================================================
h("## 5.2 ขนาดเมทริกซ์ n")
for t in (1, 16, 32):
    p(f"**GFLOP/s ที่ {t} thread{'s' if t > 1 else ''}**")
    p()
    table(S[S.threads == t].pivot(index="n", columns="algo", values="gflops")[ALGOS], floatfmt=".1f")
g1 = S[S.threads == 1].pivot(index="n", columns="algo", values="gflops")
ratio = pd.DataFrame({"ikj ÷ ijk": g1.ikj / g1.ijk, "blocked ÷ ijk": g1.blocked / g1.ijk,
                      "blocked ÷ ikj": g1.blocked / g1.ikj})
p("**อัตราส่วนความเร็วระหว่างอัลกอริทึม (1 thread)**")
p()
table(ratio)
p("**เปลี่ยนแปลงเมื่อ n โต (1 thread)**")
for a in ALGOS:
    v = g1[a].dropna()
    p(f"- {a}: n={v.index[0]} → n={v.index[-1]}: {v.iloc[0]:.1f} → {v.iloc[-1]:.1f} GFLOP/s ({pct(v.iloc[-1], v.iloc[0])})")
p(f"- ikj: n=1600 → 3200: {g1.ikj[1600]:.1f} → {g1.ikj[3200]:.1f} ({pct(g1.ikj[3200], g1.ikj[1600])}) · "
  f"ijk: n=1600 → 3200: {g1.ijk[1600]:.1f} → {g1.ijk[3200]:.1f} ({pct(g1.ijk[3200], g1.ijk[1600])})")
p()
p("**เวลาส่วนคูณที่ 1 thread (วินาที)**")
p()
table(S[S.threads == 1].pivot(index="n", columns="algo", values="t_mult_med")[ALGOS], floatfmt=".4f")
source("`data/scaling.csv`")

# =====================================================================================
h("## 5.3 การตั้งค่าโปรแกรม")
B = D["block"]
p("**ขนาด tile b (blocked) — GFLOP/s**")
p()
table(B.pivot_table(index="block", columns=["n", "threads"], values="gflops"), floatfmt=".1f")
for (n, t), g in B.groupby(["n", "threads"]):
    bb, ww = g.loc[g.gflops.idxmax()], g.loc[g.gflops.idxmin()]
    p(f"- n={n}, {t} thr: ดีสุด b={int(bb.block)} ({bb.gflops:.1f}) · แย่สุด b={int(ww.block)} ({ww.gflops:.1f}) · "
      f"ต่างกัน {bb.gflops / ww.gflops:.1f}× · b=64 (ค่าเริ่มต้น) = {g[g.block == 64].gflops.iloc[0]:.1f}")
p()
p("tile ที่ใช้ cache: 3 × b² × 8 B → " + " · ".join(f"b={b}: {3 * b * b * 8 / 1024:.0f} KB"
                                                  for b in (8, 32, 64, 128, 192, 256, 512)))
p()
source("`data/block.csv` (40 แถว)")

C = D["schedule"]
p("**OpenMP schedule — GFLOP/s** (คอลัมน์ = algo · threads)")
for n in sorted(C.n.unique()):
    g = C[C.n == n].copy()
    g["col"] = g.algo + " · " + g.threads.astype(str)
    piv = g.pivot_table(index="label", columns="col", values="gflops", sort=False)
    p()
    p(f"*n = {n}*")
    p()
    table(piv, floatfmt=".1f")
    rel = (piv / piv.loc["static"] * 100).round(0)
    p(f"*n = {n} เทียบกับ static (%)*")
    p()
    table(rel, floatfmt=".0f")
p("**จำนวนงาน (ชิ้น) ที่แบ่งได้ — ใช้คิดว่า thread ทำงานกี่ตัว**")
p()
rows = []
for n in (400, 1600, 3200):
    tiles = int(np.ceil(n / 64)) ** 2
    rows.append([n, n, tiles, int(np.ceil(tiles / 16)), int(np.ceil(n / 16))])
table(pd.DataFrame(rows, columns=["n", "แถว (ikj)", "tile 64×64 (blocked, collapse 2)",
                                  "ก้อน static,16 ของ blocked", "ก้อน static,16 ของ ikj"]), index=False)
source("`data/schedule.csv` (84 แถว)")

K = D["compiler"]
K["col"] = K.algo + " · " + K.threads.astype(str)
piv = K.pivot_table(index="build", columns="col", values="gflops", sort=False)
p("**ระดับ optimisation ของ gcc (n = 1600) — GFLOP/s**")
p()
table(piv, floatfmt=".1f")
p("เร็วขึ้นกี่เท่าเทียบกับ -O0:")
p()
table(piv / piv.loc["O0"], floatfmt=".1f")
source("`data/compiler.csv` (30 แถว) · O3native = `-O3 -march=native`")

# =====================================================================================
h("## 5.4 การตั้งค่าระบบปฏิบัติการ")
O = D["os"].copy()
O["kind"] = O.label.str.split(":").str[0]
O["setting"] = O.label.str.split(":").str[1]
O["col"] = O.algo + " · " + O.threads.astype(str)
for kind, title in [("bind", "affinity (OMP_PROC_BIND / OMP_PLACES), n=3200"),
                    ("taskset", "taskset — SMT / CCD, n=3200"),
                    ("contention", "มีโปรเซสอื่น (parallel 16 thr ijk) รันพร้อมกัน, n=3200"),
                    ("power", "power profile, n=3200")]:
    g = O[O.kind == kind]
    p(f"**{title} — GFLOP/s**")
    p()
    table(g.pivot_table(index="setting", columns="col", values="gflops", sort=False), floatfmt=".1f")
tk = O[O.kind == "taskset"].set_index(["setting", "algo"]).gflops
p("**การเปรียบเทียบที่คำนวณไว้**")
p(f"- 16 thr: 8 core × SMT vs 16 core จริง → blocked {pct(tk[('8cores-SMT', 'blocked')], tk[('16cores-noSMT', 'blocked')])} · "
  f"ikj {pct(tk[('8cores-SMT', 'ikj')], tk[('16cores-noSMT', 'ikj')])}")
p(f"- 8 thr: 2 CCD (4+4) vs 1 CCD → blocked {pct(tk[('2CCD-4+4cores', 'blocked')], tk[('1CCD-8cores', 'blocked')])} · "
  f"ikj {pct(tk[('2CCD-4+4cores', 'ikj')], tk[('1CCD-8cores', 'ikj')])}")
bd = O[O.kind == "bind"].set_index(["setting", "algo", "threads"]).gflops
p(f"- blocked 16 thr: close-threads {bd[('close-threads', 'blocked', 16)]:.1f} vs default {bd[('default', 'blocked', 16)]:.1f} "
  f"({pct(bd[('close-threads', 'blocked', 16)], bd[('default', 'blocked', 16)])}) · เทียบ taskset 8 core × SMT = {tk[('8cores-SMT', 'blocked')]:.1f}")
ct = O[O.kind == "contention"].set_index(["setting", "threads"]).gflops
for t in (16, 32):
    p(f"- contention {t} thr: {ct[('none', t)]:.1f} → {ct[('bg16', t)]:.1f} ({pct(ct[('bg16', t)], ct[('none', t)])})")
pw = O[O.kind == "power"].set_index(["setting", "algo", "threads"]).gflops
for a in ("ikj", "blocked"):
    for t in (1, 16):
        p(f"- power-saver vs performance, {a} {t} thr: {pct(pw[('power-saver', a, t)], pw[('performance', a, t)])} · "
          f"balanced: {pct(pw[('balanced', a, t)], pw[('performance', a, t)])}")
p()
source("`data/os.csv` (57 แถว)")

# =====================================================================================
h("## 5.5 หน่วยความจำ")
m = S[(S.threads == 1) & (S.algo == "blocked")].sort_values("n")
mem = pd.DataFrame({"n": m.n, "ทฤษฎี 3n²·8 (MiB)": 3 * m.n ** 2 * 8 / 2 ** 20,
                    "peak RSS วัดได้ (MiB)": m.maxrss_kb / 1024, "VmPeak (MiB)": m.vmpeak_kb / 1024})
mem["RSS − ทฤษฎี (MiB)"] = mem.iloc[:, 2] - mem.iloc[:, 1]
p("**peak RSS เทียบทฤษฎี (blocked, 1 thread)**")
p()
table(mem, floatfmt=".1f", index=False)
v = S[(S.n == 1600) & (S.algo == "blocked")].sort_values("threads")
slope = np.polyfit(v.threads, v.vmpeak_kb / 1024, 1)[0]
p("**VmPeak กับ RSS ตามจำนวน thread (blocked, n = 1600)**")
p()
table(pd.DataFrame({"threads": v.threads, "VmPeak (MiB)": v.vmpeak_kb / 1024,
                    "peak RSS (MiB)": v.maxrss_kb / 1024}), floatfmt=".0f", index=False)
bad = mem[mem["peak RSS วัดได้ (MiB)"] > mem["VmPeak (MiB)"]]
p(f"- n ที่ peak RSS > VmPeak (RSS มากกว่าพื้นที่ที่จองไว้ทั้งหมด → ไม่ใช่ค่าของโปรเซสนี้เอง): "
  f"{', '.join(str(n) for n in bad.n)}")
p()
p(f"- VmPeak เพิ่มเฉลี่ย **{slope:.2f} MiB ต่อ thread** (linear fit) ≈ stack 8 MiB + guard")
st = O[O.kind == "stack"]
p()
p("**OMP_STACKSIZE (blocked, n = 1600, 32 threads)**")
p()
table(pd.DataFrame({"OMP_STACKSIZE": st.setting, "VmPeak (MiB)": st.vmpeak_kb / 1024,
                    "peak RSS (MiB)": st.maxrss_kb / 1024, "GFLOP/s": st.gflops}), floatfmt=".1f", index=False)
p("**ru_maxrss ถูกส่งต่อข้าม execve()** (ทดสอบ 2026-10-25, `./parallel -n 100 -t 1`)")
p()
table(pd.DataFrame([["bash", "2,380 KB (2.3 MiB)"], ["Python ที่ถือ RSS ~200 MB แล้วเรียก", "216,308 KB (211 MiB)"],
                    ["run_bench.py (Python) — ค่าในข้อมูลจริงที่ n ≤ 800", f"{m[m.n <= 800].maxrss_kb.min() / 1024:.1f}–{m[m.n <= 800].maxrss_kb.max() / 1024:.1f} MiB"]],
                   columns=["โปรเซสแม่ที่เรียก parallel", "peak RSS ที่ parallel รายงาน"]), index=False)
p("- `man getrusage`: \"Resource usage metrics are preserved across an execve(2).\"")
p("- glibc malloc ใช้ mmap เมื่อขอ ≥ M_MMAP_THRESHOLD = 128 KB (ค่าเริ่มต้น) · เมทริกซ์ n=100 = 78 KB ต่อตัว (ต่ำกว่า), n=200 = 313 KB (สูงกว่า)")
p()
source("`data/scaling.csv`, `data/os.csv` · การทดสอบด้วยมือ")

# =====================================================================================
h("## 5.6 Amdahl / Karp–Flatt")
rows = []
for (a, n), g in S.groupby(["algo", "n"]):
    f_all, r2 = amdahl_fit(g.threads, g.S)
    g16 = g[g.threads <= 16]
    f16, r216 = amdahl_fit(g16.threads, g16.S)
    r1 = g[g.threads == 1].iloc[0]
    best = g.loc[g.S.idxmax()]
    rows.append([a, n, r1.t_total, f_all, r2, f16, r216, r1.serial_share, f16 / r1.serial_share,
                 best.S, int(best.threads), 1 / f16 if f16 > 0 else np.inf])
am = pd.DataFrame(rows, columns=["algo", "n", "T(1) s", "f_fit ทุก p", "R² ทุก p", "f_fit p≤16", "R² p≤16",
                                 "f_meas", "f_fit≤16 ÷ f_meas", "S_max", "p ที่ S_max", "1/f_fit≤16"])
table(am, floatfmt=".4f", index=False)
p("**Karp–Flatt e(p)** (แถว = algo · n)")
p()
kf = S[S.threads > 1].pivot_table(index=["algo", "n"], columns="threads", values="KF")
table(kf, floatfmt=".4f")
p("**สัดส่วนเวลาของส่วน serial ณ p ต่างๆ (blocked)** — (t_init + t_verify) / T(p)")
p()
table((S[S.algo == "blocked"].pivot(index="n", columns="threads", values="serial_share") * 100).round(1),
      floatfmt=".1f")
p("**เวลาส่วน serial (วินาที) ที่ 1 thread** — ใช้คำนวณ f_meas")
p()
table(S[S.threads == 1].pivot_table(index="n", columns="algo", values=["t_init", "t_verify"]), floatfmt=".4f")
source("`data/scaling.csv` · วิธี fit: least squares ผ่านจุดกำเนิด f = Σxy / Σx² (x = 1 − 1/p, y = 1/S − 1/p)")

# =====================================================================================
h("## 5.7 ข้อจำกัด / ความแกว่งของผล")
S["spread"] = S.t_mult_max / S.t_mult_min
sp = S.groupby("n").spread.agg(["median", "max"])
sp.columns = ["median ของ max÷min", "max ของ max÷min"]
p("**ความแกว่งของเวลาส่วนคูณระหว่างรอบซ้ำในรันเดียว** (t_mult_max ÷ t_mult_min)")
p()
table(sp, floatfmt=".2f")
allt = pd.concat(D.values())
p(f"- อุณหภูมิ Tctl ก่อนรัน: {allt.temp_before.min():.1f}–{allt.temp_before.max():.1f} °C · "
  f"หลังรัน: {allt.temp_after.min():.1f}–{allt.temp_after.max():.1f} °C · median หลังรัน {allt.temp_after.median():.1f} °C")
p(f"- รันที่อุณหภูมิหลังรัน ≥ 99 °C: {(allt.temp_after >= 99).sum()} / {len(allt)}")
p(f"- เวลาส่วนคูณสั้นที่สุดที่วัด: {S.t_mult_med.min() * 1e6:.0f} µs · ยาวที่สุด: {S.t_mult_med.max():.1f} s")
p("- ไม่ได้วัดโดยตรง: จำนวน cache miss, memory bandwidth, ความถี่ CPU ระหว่างรัน")
p()
source("`data/*.csv` คอลัมน์ t_mult_min, t_mult_max, temp_before, temp_after")

# =====================================================================================
h("## 6 mmap.py")
p("**จากการรันจริง** (`parallel -n 2000 -t 8 -a ijk -r 1 --hold 3`, 2026-10-24)")
p()
table(pd.DataFrame([
    ["จำนวน VMA ทั้งหมด", "32"], ["virtual / resident (VmSize / VmRSS)", "154,288 kB / 96,252 kB"],
    ["anon mmap (เมทริกซ์ A, B, C)", "1 VMA · 91.6 MB · RSS 91.6 MB (malloc 3 ครั้งถูก kernel รวมเป็น VMA เดียว)"],
    ["เมทริกซ์ 1 ตัวที่ n = 2000", "2000² × 8 B = 30.5 MiB"],
    ["thread stack", "1 VMA · 56.0 MB · RSS 56 KB → ≈ 7 stack × 8 MB (8 threads − main)"],
    ["[stack] ของ main thread", "136 KB · RSS 20 KB"], ["[heap]", "132 KB · RSS 12 KB"],
    ["libgomp / libc (code)", "264 KB / 1.6 MB (r-x)"],
    ["ช่องว่างใน address space", "15.2 TB และ 18.4 TB (ระหว่างกลุ่ม library/stack/โปรแกรม)"],
], columns=["รายการ", "ค่า"]), index=False)
p("**thread stack บน glibc 2.43 + Linux 7.0** (ทดสอบด้วย Python 4 threads ที่ sleep)")
p()
table(pd.DataFrame([
    ["stack ของ 4 thread", "1 VMA ขนาด 32,784 kB = 4 × (8,192 + 4) kB · VmFlags: rd wr mr mw me **gu** ac sd **nh**"],
    ["malloc arena (ต่อ thread)", "rw-p 132 kB + ---p 65,404 kB (สำรองรวม 64 MiB = HEAP_MAX_SIZE)"],
    ["วิธีเดิม (นับ guard VMA)", "พบ 2 stack (ผิด — ไปจับ malloc arena) จากที่ควรเป็น 4"],
    ["วิธีใหม่ (VmFlags + ขนาด)", "32 MB ÷ (8 MB + 4 KB) ≈ 4 ✓ ตรงกับ Threads: 5 − 1"],
    ["gu", "มี guard region ที่ติดตั้งด้วย MADV_GUARD_INSTALL (Linux ≥ 6.13)"],
    ["nh", "no huge page — ตั้งให้เมื่อ mmap ด้วย MAP_STACK"],
], columns=["รายการ", "ค่า"]), index=False)
p("**ค่าที่ออกแบบไว้ในโปรแกรม**")
p("- หมวดของ VMA: 13 หมวด · รีเฟรชถี่สุด 0.5 วินาที · guard เล็ก ≤ 1 MiB (เกินนี้ถือเป็นพื้นที่สำรอง)")
p("- แหล่งข้อมูลที่อ่าน: `/proc/<pid>/smaps` (หรือ maps) · status · limits · exe · cmdline")
p("- โหมด: snapshot · `--all` · `--watch` · `--html` · `--run -- CMD` · `--self` · `--list`")
p("- ภาพในเอกสาร: แอนิเมชัน 39 เฟรม (snapshot ทุก 0.1 วินาที) · รายงาน HTML 112 snapshot (ทุก 0.02 วินาที)")
p()
source("ผลรันจริงของ `mmap.py` (ภาพ `report/assets/mmap_terminal.png`, `mmap_html_report.png`) · `/proc/<pid>/smaps`")

# =====================================================================================
h("## 7 ขนาดโค้ด")
files = [("parallel/parallel.c", "โปรแกรม"),
         ("parallel/Makefile", "โปรแกรม"),
         ("mmap/mmap.py", "โปรแกรม"),
         ("bench/run_bench.py", "สคริปต์ทดลอง"), ("bench/analyze.py", "สคริปต์ทดลอง")]
rows = []
for f, status in files:
    path = ROOT / f if (ROOT / f).exists() else None
    if path:
        rows.append([path.name, status, sum(1 for _ in path.open())])
table(pd.DataFrame(rows, columns=["ไฟล์", "", "บรรทัด"]), index=False)

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUT.relative_to(ROOT)} ({len(lines)} lines)")
