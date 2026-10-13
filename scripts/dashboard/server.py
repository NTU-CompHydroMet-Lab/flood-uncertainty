#!/usr/bin/env python3
"""Local project dashboard: experiment records, training progress, project status.

Standard library only, so it runs with any Python >= 3.8; the per-image sample view (samples.py) additionally
needs numpy / rasterio / Pillow, so prefer the project venv:

    .venv/bin/python scripts/dashboard/server.py       # http://127.0.0.1:8765
    python scripts/dashboard/server.py --port 9000 --root /path/to/flood-uncertainty

Everything is read live from disk on each request:
  - experiments: artifacts/models/<experiment>/lightning_logs/version_*/{metrics.csv,hparams.yaml}
  - evaluation:  artifacts/results/**/metrics_*.csv (per-image TP/FP/FN -> pooled IoU)
  - training:    running scripts/train/*.py processes (/proc), their stdout log, nvidia-smi
  - project:     git, GitHub PRs/issues via `gh` (if logged in), pytest run on demand (CPU only)

Binds to 127.0.0.1 by default. From another machine use an SSH tunnel, e.g.
    ssh -L 8765:localhost:8765 <host>
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # replaced in main() by --root / auto-detection

# Display names for evaluation tags (artifacts/results/.../<subset>/<tag>/). First matching prefix wins;
# "{ep}" is replaced by the _ep<N> suffix. Unknown tags are shown as-is. See docs/edl_vs_terramind_comparison.md.
TAG_NAMES = [
    ("EDL-TERRAMIND_nofreeze_wf2dec", "tm_l2a6_wf2dec"),
    ("EDL-TERRAMIND_nofreeze", "tm_l2a6"),
    ("EDL-TERRAMIND_freeze_wf2dec", "tm_frozen_l2a6_wf2dec"),
    ("EDL-TERRAMIND_freeze", "tm_frozen_l2a6"),
    ("EDL-TERRAMIND_s2l1c", "tm_l1c13"),
    ("EDL-TERRAMIND_tim", "tm_l1c13_tim"),
    ("EDL-TERRAMIND_inskip", "tm_l1c13_inskip"),
    ("EDL_unet_bs64", "unet_bs64"),
    ("EDL_unet_bs16", "unet_bs16"),
]
EXACT_TAG_NAMES = {"EDL": "unet_baseline@ep8"}
# Experiment directories without a config in configurations/ that names them.
EXPERIMENT_NAMES = {
    "edl_EpochKL_20260819": "unet_baseline",
    "edl_terramind_v1_base_bgriswirs_freeze_wf2dec": "tm_frozen_l2a6_wf2dec",
    "edl_terramind_v1_base_bgriswirs_nofreeze_wf2dec": "tm_l2a6_wf2dec",
    "edl_unet_bgriswirs_bs16": "unet_bs16",
    "edl_unet_bgriswirs_bs64": "unet_bs64",
}

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")
PROGRESS = re.compile(
    r"Epoch (\d+)/(\d+)\s.*?(\d+)/(\d+)\s+([\d:]+)\s*•\s*([-\d:]+)\s+([\d.]+)\s*it/s"
)
TRAIN_SCRIPT = re.compile(r"scripts/(train|inference|eval|analysis)/[\w./-]+\.py")


# --------------------------------------------------------------------------- helpers


def run(cmd, cwd=None, timeout=20):
    try:
        p = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, "", str(e)


_cache = {}
_cache_lock = threading.Lock()


def cached(key, ttl, fn):
    """Return fn() memoised for ttl seconds. key may include file mtimes so edits invalidate it."""
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _cache_lock:
        _cache[key] = (now, val)
    return val


def tail_text(path, nbytes=200_000):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            data = f.read()
    except OSError:
        return ""
    return ANSI.sub("", data.decode("utf-8", "replace"))


def log_lines(text, n=40):
    """Last n display lines; carriage-return progress redraws keep only their final state."""
    out = []
    for raw in text.split("\n"):
        seg = raw.split("\r")
        line = next((s for s in reversed(seg) if s.strip()), "")
        if line.strip():
            out.append(line.rstrip())
    return out[-n:]


def fnum(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None  # drop NaN


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------- experiments


def config_names():
    """experiment_name -> config stem, from configurations/**/*.json (sub-folders win over top level)."""

    def build():
        names = {}
        for cfg in sorted((ROOT / "configurations").rglob("*.json"), key=lambda p: (len(p.parts), str(p))):
            try:
                c = json.loads(cfg.read_text())
            except (OSError, ValueError):
                continue
            exp = (c.get("shared") or c).get("experiment_name")
            if exp and (exp not in names or cfg.parent.name != "configurations"):
                names[exp] = cfg.stem
        return names

    return cached("config_names", 60, build)


def experiment_display(exp_dir_name):
    return EXPERIMENT_NAMES.get(exp_dir_name) or config_names().get(exp_dir_name) or exp_dir_name


def parse_hparams(path):
    info = {}
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return info
    for key in ("max_epochs", "batch_size", "backbone", "backbone_freeze", "channel_configuration", "lr",
                "num_channels", "experiment_name"):
        m = re.search(rf"^\s*{key}:\s*(\S+)\s*$", text, re.M)
        if m:
            info[key] = m.group(1).strip("'\"")
    m = re.search(r"^\s*backbone_modalities:\s*\n((?:\s*-\s*\S+\s*\n)+)", text, re.M)
    if m:
        info["backbone_modalities"] = re.findall(r"-\s*(\S+)", m.group(1))
    m = re.search(r"^\s*backbone_tim_modalities:\s*\n((?:\s*-\s*\S+\s*\n)+)", text, re.M)
    if m:
        info["tim"] = re.findall(r"-\s*(\S+)", m.group(1))
    return info


def parse_metrics(path):
    """metrics.csv -> per-epoch validation metrics + downsampled train loss. Cached on mtime."""

    def build():
        epochs, loss = {}, []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            val_cols = [c for c in cols if c.startswith("val_")]
            for row in reader:
                ep = fnum(row.get("epoch"))
                step = fnum(row.get("step"))
                lv = fnum(row.get("loss"))
                if lv is not None and step is not None:
                    loss.append((step, lv))
                vals = {c: fnum(row[c]) for c in val_cols if fnum(row.get(c)) is not None}
                tle = fnum(row.get("train_loss_epoch"))
                if tle is not None:
                    vals["train_loss_epoch"] = tle
                if vals and ep is not None:
                    epochs.setdefault(int(ep), {}).update(vals)
        if len(loss) > 600:
            k = len(loss) / 600
            loss = [loss[int(i * k)] for i in range(600)] + [loss[-1]]
        ep_list = [dict(epoch=e, **epochs[e]) for e in sorted(epochs)]
        metric_cols = sorted({k for e in ep_list for k in e if k != "epoch"})
        return {"epochs": ep_list, "loss": loss, "metrics": metric_cols}

    return cached(("metrics", str(path), path.stat().st_mtime), 3600, build)


def pick(cols, *candidates):
    return next((c for c in candidates if c in cols), None)


def scan_runs(running_exps):
    models = ROOT / "artifacts" / "models"
    runs = []
    if not models.is_dir():
        return runs
    for exp in sorted(models.iterdir()):
        if not exp.is_dir() or exp.name == "smoke":
            continue
        versions = [v for v in exp.glob("lightning_logs/version_*") if (v / "metrics.csv").exists()]
        if not versions:
            continue
        v = max(versions, key=lambda p: (p / "metrics.csv").stat().st_mtime)
        mpath = v / "metrics.csv"
        try:
            m = parse_metrics(mpath)
        except (OSError, csv.Error) as e:
            runs.append({"dir": exp.name, "name": experiment_display(exp.name), "error": str(e)})
            continue
        hp = parse_hparams(v / "hparams.yaml")
        cols = set(m["metrics"])
        iou_key = pick(cols, "val_Global_iou_land_water water", "val_iou_land_water water")
        bce_key = pick(cols, "val_bce_loss", "val_bce_land_water")
        eps = m["epochs"]
        best = min((e for e in eps if e.get(bce_key) is not None), key=lambda e: e[bce_key], default=None) \
            if bce_key else None
        ckpts = sorted(p.name for p in (exp / "checkpoint").glob("*.ckpt")) if (exp / "checkpoint").is_dir() else []
        mtime = mpath.stat().st_mtime
        runs.append({
            "dir": exp.name,
            "name": experiment_display(exp.name),
            "version": v.name,
            "hparams": hp,
            "epochs_done": (eps[-1]["epoch"] + 1) if eps else 0,
            "max_epochs": int(hp["max_epochs"]) if hp.get("max_epochs", "").isdigit() else None,
            "last_step": m["loss"][-1][0] if m["loss"] else None,
            "iou_key": iou_key,
            "bce_key": bce_key,
            "best_epoch": best["epoch"] if best else None,
            "best_bce": best.get(bce_key) if best else None,
            "best_iou": best.get(iou_key) if best and iou_key else None,
            "max_iou": max((e[iou_key] for e in eps if e.get(iou_key) is not None), default=None) if iou_key else None,
            "checkpoints": ckpts,
            "updated": mtime,
            "running": exp.name in running_exps,
        })
    return runs


def run_curves(exp_dir_name):
    models = ROOT / "artifacts" / "models"
    # name-based check (experiment dirs may be symlinks, so resolve() is not usable here)
    if not exp_dir_name or "/" in exp_dir_name or exp_dir_name.startswith(".") or not (models / exp_dir_name).is_dir():
        raise FileNotFoundError(exp_dir_name)
    exp = models / exp_dir_name
    versions = [v for v in exp.glob("lightning_logs/version_*") if (v / "metrics.csv").exists()]
    if not versions:
        raise FileNotFoundError(exp_dir_name)
    v = max(versions, key=lambda p: (p / "metrics.csv").stat().st_mtime)
    return dict(parse_metrics(v / "metrics.csv"), dir=exp_dir_name, name=experiment_display(exp_dir_name))


# --------------------------------------------------------------------------- evaluation results


def tag_display(tag):
    if tag in EXACT_TAG_NAMES:
        return EXACT_TAG_NAMES[tag]
    m = re.search(r"_ep(\d+)$", tag)
    base = tag[: m.start()] if m else tag
    for prefix, name in TAG_NAMES:
        if base == prefix:
            return f"{name}@ep{m.group(1)}" if m else name
    return tag


def scan_results():
    res_root = ROOT / "artifacts" / "results"

    def build():
        groups = {}
        for f in res_root.rglob("metrics_*.csv"):
            parts = f.relative_to(res_root).parts[:-1]
            if any("smoke" in p for p in parts):
                continue
            if "val_test_inference" in parts:
                i = parts.index("val_test_inference")
                dataset = "/".join(parts[:i]) or "val_test_inference（預設輸出）"
                rest = parts[i + 1:]
            else:
                dataset, rest = "/".join(parts[:-2]) or "results", parts[-2:]
            subset = rest[0] if len(rest) >= 2 else "?"
            tag = rest[-1] if rest else f.stem
            tp = fp = fn = 0.0
            ious, n = [], 0
            try:
                with open(f, newline="") as fh:
                    for row in csv.DictReader(fh):
                        t, p_, n_ = fnum(row.get("TP")), fnum(row.get("FP")), fnum(row.get("FN"))
                        if t is None or p_ is None or n_ is None:
                            continue
                        tp, fp, fn, n = tp + t, fp + p_, fn + n_, n + 1
                        iou = fnum(row.get("iou"))
                        if iou is not None:
                            ious.append(iou)
            except (OSError, csv.Error):
                continue
            if not n:
                continue
            groups.setdefault((dataset, subset), []).append({
                "tag": tag,
                "name": tag_display(tag),
                "n": n,
                "iou_overall": tp / (tp + fp + fn) if tp + fp + fn else None,
                "iou_average": sum(ious) / len(ious) if ious else None,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "path": rel(f),
                "dir": str(f.parent.relative_to(res_root)),
                "updated": f.stat().st_mtime,
            })
        out = []
        for (dataset, subset), rows in sorted(groups.items()):
            rows.sort(key=lambda r: -(r["iou_overall"] or 0))
            out.append({"dataset": dataset, "subset": subset, "rows": rows})
        return out

    if not res_root.is_dir():
        return []
    return cached(("results", res_root.stat().st_mtime, int(time.time() // 30)), 30, build)


def _result_dir(rel_dir):
    """artifacts/results/<rel_dir> holding a metrics_*.csv; rejects absolute paths and '..'."""
    res_root = ROOT / "artifacts" / "results"
    p = Path(rel_dir or "")
    if not rel_dir or p.is_absolute() or ".." in p.parts:
        raise FileNotFoundError(rel_dir)
    d = res_root / p
    metrics = sorted(d.glob("metrics_*.csv")) if d.is_dir() else []
    if not metrics:
        raise FileNotFoundError(rel_dir)
    return d, metrics[0]


def _per_image(metrics_csv):
    items = []
    with open(metrics_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("filename")
            if not name or name.startswith(("OVERALL", "AVERAGE")):
                continue
            tp, tn, fp, fn = (fnum(row.get(k)) for k in ("TP", "TN", "FP", "FN"))
            if None in (tp, tn, fp, fn):
                continue
            items.append({"file": name, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                          "iou": fnum(row.get("iou")), "precision": fnum(row.get("precision")),
                          "recall": fnum(row.get("recall"))})
    return items


def api_samples(q):
    rel_dir = q.get("dir", [""])[0]
    d, metrics = _result_dir(rel_dir)
    items = _per_image(metrics)
    try:
        import samples  # noqa: F401  (numpy / rasterio / Pillow)
        render_ok, render_error = True, None
    except ImportError as e:
        render_ok, render_error = False, f"{e}（請用 .venv/bin/python 啟動 server 才能畫圖）"
    return {"dir": rel_dir, "subset": d.parent.name, "tag": d.name, "name": tag_display(d.name),
            "items": items, "render": render_ok, "render_error": render_error}


def api_image(q):
    rel_dir, image, kind = q.get("dir", [""])[0], q.get("file", [""])[0], q.get("kind", ["cm"])[0]
    try:
        max_px = max(0, min(4096, int(q.get("max", ["900"])[0])))
    except ValueError:
        max_px = 900
    d, metrics = _result_dir(rel_dir)
    if image not in {it["file"] for it in _per_image(metrics)} or kind not in ("rgb", "gt", "cm"):
        raise FileNotFoundError(image)
    import samples

    crop = q.get("crop", ["1"])[0] != "0"
    key = ("img", str(d), image, kind, max_px, crop, metrics.stat().st_mtime)
    return cached(key, 3600, lambda: samples.render(d, image, kind, max_px or 4096, crop))


# --------------------------------------------------------------------------- training processes


def _proc_args(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")[:-1]
    except OSError:
        return []


def _ppid(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return int(stat.rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return None


def _elapsed(pid):
    try:
        start_ticks = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        return uptime - start_ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError):
        return None


def _stdout_file(pid):
    try:
        target = os.readlink(f"/proc/{pid}/fd/1")
    except OSError:
        return None
    return target if os.path.isfile(target) else None


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def scan_processes():
    """Top-level project jobs: python scripts/{train,inference,eval,analysis}/*.py and artifacts/runs/*.sh queues."""
    procs = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        args = _proc_args(int(d.name))
        if args:
            procs[int(d.name)] = args
    jobs, queues = [], []
    for pid, args in procs.items():
        line = " ".join(args)
        parent = procs.get(_ppid(pid))
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            cwd = ""
        if TRAIN_SCRIPT.search(line) and "python" in os.path.basename(args[0]):
            if parent and " ".join(parent) == line:
                continue  # DataLoader worker of the same job
            job = {"pid": pid, "cmd": line, "cwd": cwd, "elapsed": _elapsed(pid), "kind": TRAIN_SCRIPT.search(line).group(1)}
            if "--config" in args:
                cfg_path = Path(args[args.index("--config") + 1])
                cfg_path = cfg_path if cfg_path.is_absolute() else Path(cwd) / cfg_path
                job["config"] = str(cfg_path)
                try:
                    cfg = json.loads(cfg_path.read_text())
                    job["experiment"] = _find_key(cfg, "experiment_name")
                    job["max_epochs"] = _find_key(cfg, "max_epochs")
                except (OSError, ValueError):
                    pass
            log = _stdout_file(pid)
            if log:
                text = tail_text(log)
                job["log"] = log
                job["log_tail"] = log_lines(text, 30)
                prog = None
                for m in PROGRESS.finditer(text):
                    prog = m
                if prog:
                    ep, ep_max, b, b_tot, el, eta, its = prog.groups()
                    job["progress"] = {"epoch": int(ep), "epoch_max": int(ep_max), "batch": int(b),
                                       "batches": int(b_tot), "epoch_elapsed": el, "epoch_eta": eta, "it_s": float(its)}
            if job.get("experiment"):
                job["name"] = experiment_display(job["experiment"])
            jobs.append(job)
        elif os.path.basename(args[0]) in ("bash", "sh") and len(args) > 1 and re.search(r"artifacts/runs/.*\.sh$", args[1]):
            q = {"pid": pid, "cmd": line, "cwd": cwd, "elapsed": _elapsed(pid)}
            log = _stdout_file(pid)
            if log:
                q["log"] = log
                q["log_tail"] = log_lines(tail_text(log, 20_000), 12)
            queues.append(q)
    jobs.sort(key=lambda j: j["pid"])
    return jobs, queues


def gpu_status():
    code, out, err = run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                          "--format=csv,noheader,nounits"], timeout=10)
    if code != 0:
        return {"error": (err or "nvidia-smi unavailable").strip()}
    gpus = []
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 7:
            gpus.append({"index": f[0], "name": f[1], "util": fnum(f[2]), "mem_used": fnum(f[3]),
                         "mem_total": fnum(f[4]), "temp": fnum(f[5]), "power": fnum(f[6])})
    return {"gpus": gpus}


def recent_logs(n=6):
    runs_dir = ROOT / "artifacts" / "runs"
    if not runs_dir.is_dir():
        return []
    logs = sorted(runs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:n]
    return [{"path": rel(p), "updated": p.stat().st_mtime, "tail": log_lines(tail_text(p, 30_000), 8)} for p in logs]


# --------------------------------------------------------------------------- project status


def git_status():
    def g(*a):
        return run(["git", *a], timeout=15)[1].strip()

    common = Path(g("rev-parse", "--path-format=absolute", "--git-common-dir") or ROOT / ".git")
    fetch_head = common / "FETCH_HEAD"
    branches = []
    for line in g("for-each-ref", "refs/heads", "--sort=-committerdate",
                  "--format=%(refname:short)\t%(upstream:short)\t%(upstream:track)\t%(committerdate:relative)\t%(subject)").splitlines():
        b = line.split("\t")
        if len(b) == 5:
            branches.append(dict(zip(("name", "upstream", "track", "date", "subject"), b)))
    default = "origin/dev" if g("rev-parse", "--verify", "-q", "origin/dev") else "HEAD"
    commits = []
    for line in g("log", default, "-15", "--format=%h\t%an\t%ar\t%s").splitlines():
        c = line.split("\t")
        if len(c) == 4:
            commits.append(dict(zip(("sha", "author", "date", "subject"), c)))
    worktrees, cur = [], {}
    for line in g("worktree", "list", "--porcelain").splitlines() + [""]:
        if not line:
            if cur:
                worktrees.append(cur)
            cur = {}
        elif line.startswith("worktree "):
            cur["path"] = line[9:]
        elif line.startswith("branch "):
            cur["branch"] = line[7:].replace("refs/heads/", "")
        elif line.startswith("HEAD "):
            cur["head"] = line[5:12]
    return {
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "head": g("log", "-1", "--format=%h %s"),
        "status": g("status", "--porcelain=v1", "-b").splitlines(),
        "branches": branches,
        "log_ref": default,
        "commits": commits,
        "worktrees": worktrees,
        "stashes": g("stash", "list", "--format=%gd\t%cr\t%gs").splitlines(),
        "last_fetch": fetch_head.stat().st_mtime if fetch_head.exists() else None,
    }


def gh_bin():
    return shutil.which("gh") or next((p for p in (Path.home() / ".local/bin/gh",) if p.exists()), None)


def github_status():
    def build():
        gh = gh_bin()
        if not gh:
            return {"error": "找不到 gh CLI（安裝並執行 gh auth login 後即可顯示 PR / issue）"}
        out = {}
        code, prs, err = run([str(gh), "pr", "list", "--state", "all", "--limit", "15", "--json",
                              "number,title,state,isDraft,author,headRefName,baseRefName,updatedAt,url,reviewDecision"], timeout=30)
        if code != 0:
            return {"error": (err or "gh pr list failed").strip().splitlines()[-1]}
        out["prs"] = json.loads(prs or "[]")
        code, issues, _ = run([str(gh), "issue", "list", "--state", "open", "--limit", "15", "--json",
                               "number,title,author,labels,updatedAt,url"], timeout=30)
        out["issues"] = json.loads(issues or "[]") if code == 0 else []
        out["fetched"] = time.time()
        return out

    return cached("github", 120, build)


class TestRunner:
    """Runs pytest (CPU only, HF offline) in the background; results persist in artifacts/dashboard/."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.started = None

    @property
    def out_dir(self):
        return ROOT / "artifacts" / "dashboard"

    def start(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return False
            self.out_dir.mkdir(parents=True, exist_ok=True)
            py = ROOT / ".venv" / "bin" / "python"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
            log = open(self.out_dir / "pytest.log", "w")
            self.proc = subprocess.Popen(
                [str(py if py.exists() else sys.executable), "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider",
                 "--color=no", f"--junitxml={self.out_dir / 'pytest.xml'}"],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            self.started = time.time()
            return True

    def status(self):
        running = bool(self.proc and self.proc.poll() is None)
        xml = self.out_dir / "pytest.xml"
        res = {"running": running, "started": self.started}
        if xml.exists() and not running:
            try:
                root = ET.parse(xml).getroot()
                suite = root if root.tag == "testsuite" else root.find("testsuite")
                cases = []
                for tc in suite.iter("testcase"):
                    state, msg = "passed", ""
                    for tag in ("failure", "error", "skipped"):
                        el = tc.find(tag)
                        if el is not None:
                            state = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
                            msg = (el.get("message") or "").splitlines()[0][:300] if el.get("message") else ""
                    cases.append({"name": f"{tc.get('classname')}::{tc.get('name')}", "state": state, "message": msg,
                                  "time": fnum(tc.get("time"))})
                res.update(finished=xml.stat().st_mtime, cases=cases,
                           totals={s: sum(c["state"] == s for c in cases) for s in ("passed", "failed", "error", "skipped")},
                           duration=fnum(suite.get("time")))
            except (ET.ParseError, AttributeError) as e:
                res["error"] = f"cannot parse pytest.xml: {e}"
        if running:
            res["log_tail"] = log_lines(tail_text(self.out_dir / "pytest.log", 5000), 3)
        return res


TESTS = TestRunner()


def docs_list():
    out = []
    for p in sorted((ROOT / "docs").glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        title = next((l[2:].strip() for l in p.read_text(errors="replace").splitlines() if l.startswith("# ")), p.stem)
        out.append({"path": rel(p), "title": title, "updated": p.stat().st_mtime})
    return out


# --------------------------------------------------------------------------- HTTP


def api_training():
    jobs, queues = scan_processes()
    for j in jobs:
        exp = j.get("experiment")
        mdir = ROOT / "artifacts" / "models" / exp if exp else None
        if mdir and mdir.is_dir():
            try:
                c = run_curves(exp)
                j["curves"] = {"epochs": c["epochs"], "loss": c["loss"], "metrics": c["metrics"]}
            except (OSError, ValueError, csv.Error):
                pass
    return {"jobs": jobs, "queues": queues, "gpu": gpu_status(), "logs": recent_logs(), "now": time.time()}


def api_experiments():
    jobs, _ = scan_processes()
    running = {j.get("experiment") for j in jobs if j.get("experiment")}
    return {"runs": scan_runs(running), "now": time.time()}


def api_project():
    return {"git": git_status(), "github": github_status(), "tests": TESTS.status(), "docs": docs_list(),
            "root": str(ROOT), "now": time.time()}


ROUTES = {
    "/api/training": lambda q: api_training(),
    "/api/experiments": lambda q: api_experiments(),
    "/api/run": lambda q: run_curves(q.get("dir", [""])[0]),
    "/api/results": lambda q: {"groups": scan_results(), "now": time.time()},
    "/api/project": lambda q: api_project(),
    "/api/samples": api_samples,
}


class Handler(BaseHTTPRequestHandler):
    server_version = "flood-dashboard/1"

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/image":
            try:
                return self._send(200, api_image(parse_qs(u.query)), "image/png")
            except FileNotFoundError as e:
                return self._send(404, {"error": f"not found: {e}"})
            except ImportError as e:
                return self._send(501, {"error": f"{e}（請用 .venv/bin/python 啟動 server）"})
            except Exception as e:
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        fn = ROUTES.get(u.path)
        if not fn:
            return self._send(404, {"error": "not found"})
        try:
            self._send(200, fn(parse_qs(u.query)))
        except FileNotFoundError as e:
            self._send(404, {"error": f"not found: {e}"})
        except Exception as e:  # keep the dashboard up; report the failure to the page
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        if urlparse(self.path).path == "/api/tests/run":
            # simple CSRF guard: only same-origin page requests carry this header
            if self.headers.get("X-Dashboard") != "1":
                return self._send(403, {"error": "forbidden"})
            return self._send(200, {"started": TESTS.start()})
        if urlparse(self.path).path == "/api/github/refresh":
            with _cache_lock:
                _cache.pop("github", None)
            return self._send(200, {"ok": True})
        self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        pass


def resolve_root(arg):
    if arg:
        return Path(arg).resolve()
    root = HERE.parents[1]
    if (root / "artifacts").is_dir():
        return root
    # Running from a git worktree: artifacts/ (gitignored) live in the main checkout.
    code, out, _ = run(["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"])
    if code == 0 and out.strip():
        main = Path(out.strip()).parent
        if (main / "artifacts").is_dir():
            return main
    return root


def main():
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", default=None, help="project checkout that holds artifacts/ (default: auto)")
    args = ap.parse_args()
    ROOT = resolve_root(args.root)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dashboard: http://{args.host}:{args.port}  (root: {ROOT})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
