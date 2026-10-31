"""Run the smoke gate, then continue the same frozen manifest in bulk."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON containing argv, output, and log_dir; no secrets")
    a = ap.parse_args()
    config = json.loads(Path(a.config).read_text(encoding="utf-8"))
    argv, out, logs = config["argv"], Path(config["output"]), Path(config["log_dir"])
    logs.mkdir(parents=True, exist_ok=True)

    def state(phase, **extra):
        value = dict(phase=phase, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra)
        tmp = logs / "supervisor_status.tmp"
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(logs / "supervisor_status.json")
        print(json.dumps(value, ensure_ascii=False), flush=True)

    state("smoke_running")
    with open(logs / "final_smoke.log", "a", encoding="utf-8") as f:
        result = subprocess.run(argv + ["--limit", "12"], stdout=f, stderr=subprocess.STDOUT)
    status = json.loads((out / "status.json").read_text(encoding="utf-8")) if (out / "status.json").exists() else {}
    passed = status.get("results", {}).get("auto_accepted", 0)
    if result.returncode or status.get("state") != "smoke_finished" or not passed:
        state("smoke_failed", exit_code=result.returncode, accepted=passed)
        raise SystemExit(1)
    state("bulk_running", smoke_accepted=passed)
    with open(logs / "bulk.log", "a", encoding="utf-8") as f:
        result = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT)
    state("bulk_finished" if result.returncode == 0 else "bulk_failed", exit_code=result.returncode)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
