#!/usr/bin/env python3
"""mmap.py — visualise the memory map of a Linux process from /proc/<pid>/smaps.

OS Report 2026, task 2. For educational purposes only — not production software.
Python 3.8+ standard library only (no pip install needed). Linux only.

Usage
  python3 mmap.py PID                    snapshot: summary + address-space layout
  python3 mmap.py PID --all              also list every region (no merging of segments)
  python3 mmap.py PID --watch            real-time view, refresh every --interval seconds
  python3 mmap.py PID --html out.html    write a self-contained HTML/SVG report
  python3 mmap.py --run -- CMD [ARGS...] start CMD, watch it until it exits, then report
  python3 mmap.py --self                 inspect this Python process itself (demo)
  python3 mmap.py --list                 list your own processes

Example (watch the task-1 program allocate its matrices and start its threads):
  python3 mmap.py --run --html run.html -- ../parallel/parallel -n 3000 -t 8 --hold 3

How it works
  /proc/<pid>/maps lists every virtual memory area (VMA): address range, permissions, file.
  /proc/<pid>/smaps adds, per VMA, how much is really in RAM (Rss), shared (Pss), swapped...
  We classify every VMA (program code, libraries, heap, stacks, anonymous mmap, ...) and
  draw them from high to low address, collapsing the huge unmapped gaps.

Safety: read-only. Only reads /proc files that the kernel already lets you read (your own
processes); no ptrace, no root. Refresh is limited to >= 0.5 s because reading smaps briefly
takes the target's memory-map lock.
"""
import argparse
import html
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

MIN_INTERVAL = 0.5
PAGE = os.sysconf("SC_PAGE_SIZE")

CATEGORIES = {
    "program code": (33, "#2563eb", "executable instructions of the program (r-x)"),
    "program data": (39, "#60a5fa", "read-only constants, .data and .bss of the program"),
    "library code": (35, "#9333ea", "instructions of shared libraries (.so, r-x)"),
    "library data": (141, "#c084fc", "data segments of shared libraries"),
    "heap":         (208, "#ea580c", "[heap] grown with brk() — small malloc() blocks"),
    "anon mmap":    (220, "#ca8a04", "anonymous mmap: big malloc() blocks (>= 128 KB) etc."),
    "malloc arena": (178, "#a16207", "per-thread malloc arena (used part; rest reserved ---p)"),
    "thread stack": (41, "#16a34a", "stacks of non-main threads (MAP_STACK, guard page)"),
    "stack":        (34, "#15803d", "[stack] of the main thread"),
    "guard":        (244, "#9ca3af", "---p: reserved/guard pages, no access, never in RAM"),
    "mapped file":  (51, "#0891b2", "other memory-mapped files (locale, fonts, caches...)"),
    "shared mem":   (199, "#db2777", "shared memory (/dev/shm, memfd, SysV)"),
    "kernel":       (250, "#6b7280", "[vdso]/[vvar]: kernel pages mapped for fast syscalls"),
}


@dataclass
class Region:
    start: int
    end: int
    perms: str
    offset: int
    path: str
    stats: dict = field(default_factory=dict)
    category: str = ""
    flags: set = field(default_factory=set)

    @property
    def size(self):
        return self.end - self.start

    @property
    def rss(self):
        return self.stats.get("Rss", 0) * 1024

    @property
    def anonymous(self):
        return self.path == "" or self.path.startswith("[anon")


@dataclass
class Snapshot:
    pid: int
    time: float
    regions: list
    status: dict
    cmdline: str
    has_rss: bool
    stack_limit: int


# ------------------------------------------------------------------ reading /proc

HEADER_RE = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) (\S{4}) ([0-9a-f]+) \S+ \d+\s*(.*)$")


def parse_maps_text(text):
    regions = []
    for line in text.splitlines():
        m = HEADER_RE.match(line)
        if m:
            regions.append(Region(int(m.group(1), 16), int(m.group(2), 16), m.group(3),
                                  int(m.group(4), 16), m.group(5).strip()))
        elif regions and line.endswith(" kB"):
            key, _, value = line.partition(":")
            regions[-1].stats[key] = int(value.split()[0])
        elif regions and line.startswith("VmFlags:"):
            regions[-1].flags = set(line.split()[1:])
    return regions


def read_stack_limit(pid):
    try:
        with open(f"/proc/{pid}/limits") as f:
            for line in f:
                if line.startswith("Max stack size"):
                    soft = line.split()[3]
                    return int(soft) if soft.isdigit() else 8 << 20
    except OSError:
        pass
    return 8 << 20


def read_status(pid):
    wanted = ("Name", "State", "Threads", "VmPeak", "VmSize", "VmRSS", "VmHWM", "RssAnon", "RssFile",
              "RssShmem", "VmSwap", "VmStk", "VmData", "VmExe", "VmLib")
    status = {}
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            key, _, value = line.partition(":")
            if key in wanted:
                status[key] = value.strip()
    return status


def take_snapshot(pid):
    try:
        status = read_status(pid)
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
        try:
            with open(f"/proc/{pid}/smaps") as f:
                regions, has_rss = parse_maps_text(f.read()), True
        except PermissionError:
            with open(f"/proc/{pid}/maps") as f:
                regions, has_rss = parse_maps_text(f.read()), False
    except FileNotFoundError:
        raise ProcessLookupError(pid) from None
    try:
        exe = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        exe = next((r.path for r in regions if r.path.startswith("/")), "")
    classify(regions, exe)
    return Snapshot(pid, time.time(), regions, status, cmdline or status.get("Name", "?"),
                    has_rss, read_stack_limit(pid))


# ------------------------------------------------------------------ classification

GUARD_MAX = 1 << 20


def classify(regions, exe):
    for i, r in enumerate(regions):
        p = r.path
        if p in ("[vdso]", "[vvar]", "[vsyscall]", "[vvar_vclock]"):
            r.category = "kernel"
        elif p == "[heap]":
            r.category = "heap"
        elif p.startswith("[stack"):
            r.category = "stack"
        elif p.startswith(("/dev/shm", "/memfd:", "/SYSV")):
            r.category = "shared mem"
        elif r.anonymous:
            prev = regions[i - 1] if i else None
            nxt = regions[i + 1] if i + 1 < len(regions) else None
            if r.perms.startswith("---"):
                r.category = "guard"
            elif "gu" in r.flags or "nh" in r.flags:
                r.category = "thread stack"
            elif (prev and prev.end == r.start and prev.anonymous and prev.perms.startswith("---")
                  and prev.size <= GUARD_MAX and "w" in r.perms):
                r.category = "thread stack"
            elif (nxt and nxt.start == r.end and nxt.anonymous and nxt.perms.startswith("---")
                  and nxt.size > GUARD_MAX):
                r.category = "malloc arena"
            else:
                r.category = "anon mmap"
        elif p == exe:
            r.category = "program code" if "x" in r.perms else "program data"
        elif ".so" in os.path.basename(p):
            r.category = "library code" if "x" in r.perms else "library data"
        else:
            r.category = "mapped file"


def summarise(snap):
    out = {c: [0, 0, 0] for c in CATEGORIES}
    for r in snap.regions:
        s = out[r.category]
        s[0] += 1; s[1] += r.size; s[2] += r.rss
    return out


def merge_segments(regions):
    merged = []
    for r in regions:
        last = merged[-1] if merged else None
        if (last and last.path and last.path == r.path and last.category == r.category
                and last.end == r.start):
            last.end = r.end
            for k, v in r.stats.items():
                last.stats[k] = last.stats.get(k, 0) + v
            if r.perms not in last.perms.split("/"):
                last.perms += "/" + r.perms
        else:
            merged.append(Region(r.start, r.end, r.perms, r.offset, r.path, dict(r.stats),
                                 r.category, set(r.flags)))
    return merged


# ------------------------------------------------------------------ formatting helpers

def human(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024 or unit == "TB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024


class Painter:
    def __init__(self, enabled):
        self.enabled = enabled

    def __call__(self, text, category=None, code=None):
        if not self.enabled:
            return text
        code = code if code is not None else CATEGORIES[category][0]
        return f"\x1b[38;5;{code}m{text}\x1b[0m"


def bar(fraction, width, paint, category):
    fraction = max(0.0, min(1.0, fraction))
    full = round(fraction * width)
    return paint("█" * full, category) + paint("░" * (width - full), code=240 if paint.enabled else None)


def display_name(r):
    if r.path:
        return r.path if len(r.path) < 60 else "…" + r.path[-58:]
    return {"thread stack": "(thread stack)", "guard": "(guard / reserved)"}.get(
        r.category, "(anonymous)")


# ------------------------------------------------------------------ terminal views

def render_header(snap, paint):
    st = snap.status
    lines = [paint(f"PID {snap.pid}", code=15) + f"  {snap.cmdline[:90]}",
             f"threads {st.get('Threads', '?')}  ·  virtual {st.get('VmSize', '?')}  ·  "
             f"resident {st.get('VmRSS', '?')} (anon {st.get('RssAnon', '?')}, file "
             f"{st.get('RssFile', '?')})  ·  peak RSS {st.get('VmHWM', '?')}  ·  "
             f"{len(snap.regions)} VMAs"]
    if not snap.has_rss:
        lines.append(paint("! smaps not readable — showing address ranges only (no RSS)", code=196))
    return lines


def render_summary(snap, paint, prev=None):
    summ, prev_summ = summarise(snap), summarise(prev) if prev else None
    total_rss = sum(v[2] for v in summ.values()) or 1
    lines = [f"{'category':<14}{'VMAs':>6}{'virtual':>12}{'resident':>12}  share of RSS"]
    for cat, (count, virt, rss) in summ.items():
        if not count:
            continue
        delta = ""
        if prev_summ:
            d = rss - prev_summ[cat][2]
            if abs(d) >= PAGE:
                delta = paint(f" {'+' if d > 0 else '−'}{human(abs(d))}", code=46 if d > 0 else 203)
        lines.append(f"{paint(f'{cat:<14}', cat)}{count:>6}{human(virt):>12}{human(rss):>12}  "
                     f"{bar(rss / total_rss, 24, paint, cat)} {100 * rss / total_rss:5.1f}%{delta}")
    stack_vmas, stack_bytes = summ["thread stack"][0], summ["thread stack"][1]
    est = round(stack_bytes / (snap.stack_limit + PAGE))
    threads = int(snap.status.get("Threads", "1"))
    lines.append(paint(f"thread-stack check: {stack_vmas} stack VMA(s), {human(stack_bytes)} "
                       f"≈ {est} stack(s) of {human(snap.stack_limit)} · process has {threads} "
                       f"thread(s) -> expected {threads - 1}", code=245))
    return lines


def render_layout(snap, paint, show_all=False):
    regions = snap.regions if show_all else merge_segments(snap.regions)
    ordered = sorted(regions, key=lambda r: r.start, reverse=True)
    w = max(len(f"{r.end:x}") for r in ordered) if ordered else 12
    lines = [f"{'address range':<{2 * w + 1}}  {'perm':<9}{'size':>10}{'RSS':>10}  "
             f"{'resident':<16} name"]
    for i, r in enumerate(ordered):
        if i:
            gap = ordered[i - 1].start - r.end
            if ordered[i - 1].start >= 1 << 63:
                lines.append(paint(f"{'⋮':>{w}}   non-canonical hole + kernel space", code=240))
            elif gap >= 1 << 30:
                lines.append(paint(f"{'⋮':>{w}}   {human(gap)} unmapped", code=240))
        frac = r.rss / r.size if r.size else 0
        lines.append(f"{r.start:0{w}x}-{r.end:0{w}x}  {r.perms[:9]:<9}{human(r.size):>10}"
                     f"{human(r.rss):>10}  {bar(frac, 16, paint, r.category)} "
                     f"{paint(display_name(r), r.category)}")
    return lines


def legend(paint):
    return ["legend: " + "  ".join(paint("■ " + c, c) for c in CATEGORIES),
            "bars: " + paint("█", code=15) + " resident in RAM  " + paint("░", code=240) +
            " mapped but not resident (demand paging: not touched yet / swapped / guard)"]


def sparkline(values, width=50):
    blocks = "▁▂▃▄▅▆▇█"
    values = values[-width:]
    hi = max(values) or 1
    return "".join(blocks[min(7, int(v / hi * 7.999))] for v in values)


# ------------------------------------------------------------------ HTML / SVG report

def html_report(snaps, path):
    snap = max(snaps, key=lambda s: sum(r.rss for r in s.regions))
    regions = sorted(merge_segments(snap.regions), key=lambda r: r.start, reverse=True)

    y, rows = 10, []
    for i, r in enumerate(regions):
        if i and regions[i - 1].start - r.end >= 1 << 30:
            text = ("non-canonical hole + kernel space" if regions[i - 1].start >= 1 << 63
                    else f"{human(regions[i - 1].start - r.end)} unmapped")
            rows.append(f'<text x="20" y="{y + 14}" class="gap">⋮ {text}</text>')
            y += 22
        h = max(14, (math.log2(max(r.size, PAGE)) - 11) * 3)
        frac = r.rss / r.size if r.size else 0
        colour = CATEGORIES[r.category][1]
        tip = html.escape(f"{r.start:#x}-{r.end:#x} {r.perms}\n{r.category}: {display_name(r)}\n"
                          f"size {human(r.size)}, resident {human(r.rss)} ({100 * frac:.0f}%)")
        rows.append(f'<g><title>{tip}</title>'
                    f'<rect x="150" y="{y}" width="260" height="{h:.0f}" fill="{colour}" opacity="0.25"/>'
                    f'<rect x="150" y="{y}" width="{260 * frac:.0f}" height="{h:.0f}" fill="{colour}"/>'
                    f'<text x="144" y="{y + h / 2 + 4:.0f}" class="addr">{r.start:012x}</text>'
                    f'<text x="418" y="{y + h / 2 + 4:.0f}" class="lbl">{html.escape(display_name(r))}'
                    f' — {human(r.size)} / RSS {human(r.rss)}</text></g>')
        y += h + 2
    layout_svg = f'<svg width="1100" height="{y + 10:.0f}" class="layout">{"".join(rows)}</svg>'

    summ = summarise(snap)
    table = "".join(
        f'<tr><td><span class="sw" style="background:{CATEGORIES[c][1]}"></span>{c}</td>'
        f'<td>{n}</td><td>{human(v)}</td><td>{human(rs)}</td><td>{CATEGORIES[c][2]}</td></tr>'
        for c, (n, v, rs) in summ.items() if n)

    timeline = ""
    if len(snaps) > 1:
        t0, t1 = snaps[0].time, snaps[-1].time
        series = [summarise(s) for s in snaps]
        peak = max(sum(v[2] for v in s.values()) for s in series) or 1
        W, H = 820, 260
        xs = [60 + (s.time - t0) / max(t1 - t0, 1e-9) * W for s in snaps]
        base = [0.0] * len(snaps)
        polys = []
        for cat in CATEGORIES:
            vals = [s[cat][2] for s in series]
            if not any(vals):
                continue
            top = [b + v for b, v in zip(base, vals)]
            pts = [f"{x:.1f},{H + 20 - t / peak * H:.1f}" for x, t in zip(xs, top)]
            pts += [f"{x:.1f},{H + 20 - b / peak * H:.1f}" for x, b in reversed(list(zip(xs, base)))]
            polys.append(f'<polygon points="{" ".join(pts)}" fill="{CATEGORIES[cat][1]}">'
                         f'<title>{cat}</title></polygon>')
            base = top
        timeline = (f'<h2>Resident memory over time</h2><svg width="900" height="{H + 50}">'
                    f'{"".join(polys)}<line x1="60" y1="{H + 20}" x2="{60 + W}" y2="{H + 20}" stroke="currentColor"/>'
                    f'<text x="56" y="24" class="addr">{human(peak)}</text>'
                    f'<text x="60" y="{H + 40}" class="addr">0 s</text>'
                    f'<text x="{60 + W - 40}" y="{H + 40}" class="addr">{t1 - t0:.1f} s</text></svg>'
                    f'<p>{len(snaps)} samples · hover a band to see its category</p>')

    st = snap.status
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memory map PID {snap.pid}</title><style>
:root {{ --bg:#ffffff; --fg:#1f2937; --muted:#6b7280; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#111827; --fg:#e5e7eb; --muted:#9ca3af; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px system-ui, sans-serif; margin:16px; }}
svg {{ max-width:100%; color:var(--fg); }} .addr {{ font:11px monospace; fill:var(--muted); text-anchor:end; }}
.lbl {{ font:12px monospace; fill:var(--fg); }} .gap {{ font:12px monospace; fill:var(--muted); }}
table {{ border-collapse:collapse; }} td,th {{ padding:4px 10px; border-bottom:1px solid #8884; text-align:left; }}
.sw {{ display:inline-block; width:12px; height:12px; margin-right:6px; vertical-align:middle; }}
</style></head><body>
<h1>Memory map — PID {snap.pid}</h1>
<p><code>{html.escape(snap.cmdline)}</code><br>
threads {st.get('Threads', '?')} · virtual {st.get('VmSize', '?')} · resident {st.get('VmRSS', '?')} ·
peak RSS {st.get('VmHWM', '?')} · {len(snap.regions)} VMAs · captured {time.ctime(snap.time)} (sample with the highest RSS)</p>
<h2>By category</h2><table><tr><th>category</th><th>VMAs</th><th>virtual</th><th>resident</th><th>meaning</th></tr>{table}</table>
{timeline}
<h2>Address space (high → low)</h2>
<p>Box height ∝ log₂(size). Solid part = resident in RAM, faded part = mapped but not resident.</p>
{layout_svg}
<p style="color:var(--muted)">Generated by mmap.py (OS Report 2026) — educational tool.</p>
</body></html>"""
    with open(path, "w") as f:
        f.write(page)


# ------------------------------------------------------------------ modes

def watch(pid, interval, paint, record):
    snaps, prev = [], None
    try:
        while True:
            try:
                snap = take_snapshot(pid)
            except ProcessLookupError:
                snap = None
            if snap is None or snap.status.get("State", "").startswith("Z"):
                print(f"\nprocess {pid} exited")
                break
            if record:
                snaps.append(snap)
            history = [sum(v[2] for v in summarise(s).values()) for s in snaps] if record else []
            out = ["\x1b[H\x1b[2J" if sys.stdout.isatty() else "", *render_header(snap, paint), ""]
            out += render_summary(snap, paint, prev)
            if history:
                out += ["", f"RSS history  {sparkline(history)}  {human(history[-1])}"]
            if prev:
                old = {(r.start, r.end) for r in prev.regions}
                new = [r for r in snap.regions if (r.start, r.end) not in old]
                gone = len(prev.regions) - (len(snap.regions) - len(new))
                out += ["", f"since last refresh: {len(new)} new/changed VMA(s), {gone} removed"]
                out += [f"  + {r.start:012x} {human(r.size):>9}  {paint(r.category, r.category)}"
                        f"  {display_name(r)}" for r in new[:8]]
            out += ["", f"refresh {interval}s · Ctrl-C to stop"]
            print("\n".join(out), flush=True)
            prev = snap
            time.sleep(interval)
    except KeyboardInterrupt:
        print()
    return snaps


def list_processes():
    uid = os.getuid()
    print(f"{'PID':>8} {'RSS':>10}  command")
    for entry in sorted(os.listdir("/proc"), key=lambda s: int(s) if s.isdigit() else 0):
        if not entry.isdigit():
            continue
        try:
            if os.stat(f"/proc/{entry}").st_uid != uid:
                continue
            st = read_status(int(entry))
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
            print(f"{entry:>8} {st.get('VmRSS', '-'):>10}  {(cmd or st.get('Name', ''))[:90]}")
        except OSError:
            continue


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pid", nargs="?", type=int, help="process ID to inspect")
    ap.add_argument("--self", action="store_true", help="inspect this Python process")
    ap.add_argument("--list", action="store_true", help="list your own processes")
    ap.add_argument("--run", action="store_true", help="start the command after -- and watch it")
    ap.add_argument("--watch", action="store_true", help="real-time view")
    ap.add_argument("--interval", type=float, default=1.0, help="refresh seconds (min 0.5)")
    ap.add_argument("--all", action="store_true", help="list every VMA, do not merge segments")
    ap.add_argument("--html", metavar="FILE", help="also write an HTML/SVG report")
    ap.add_argument("--no-color", action="store_true")
    argv = sys.argv[1:]
    command = []
    if "--" in argv:
        cut = argv.index("--")
        argv, command = argv[:cut], argv[cut + 1:]
    args = ap.parse_args(argv)

    paint = Painter(sys.stdout.isatty() and not args.no_color and "NO_COLOR" not in os.environ)
    interval = max(MIN_INTERVAL, args.interval)
    if args.list:
        return list_processes()

    child = None
    if args.run:
        if not command:
            ap.error("--run needs a command after --, e.g. --run -- ./parallel -n 2000 --hold 3")
        child = subprocess.Popen(command)
        pid = child.pid
        time.sleep(0.05)
    elif args.self:
        pid = os.getpid()
    elif args.pid:
        pid = args.pid
    else:
        ap.error("give a PID, --self, --run or --list")

    try:
        if args.watch or args.run:
            snaps = watch(pid, interval, paint, record=True)
        else:
            snaps = [take_snapshot(pid)]
            snap = snaps[0]
            print("\n".join(render_header(snap, paint) + [""] + render_summary(snap, paint) + [""]
                            + render_layout(snap, paint, args.all) + [""] + legend(paint)))
    except ProcessLookupError:
        sys.exit(f"no such process: {pid}")
    except PermissionError:
        sys.exit(f"permission denied reading /proc/{pid} — you can only inspect your own processes")
    finally:
        if child:
            child.wait()

    if args.html:
        if snaps:
            html_report(snaps, args.html)
            print(f"wrote {args.html}")
        else:
            print("no snapshot captured (process ended too quickly?)")


if __name__ == "__main__":
    main()
