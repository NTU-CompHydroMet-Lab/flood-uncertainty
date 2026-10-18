#!/usr/bin/env bash
# One entry point for the NCHC Nano5 workflow, run on the lab machine.
#
#   bash scripts/nchc/nchc.sh connect          # 2FA once: login (22) + transfer (2222) connections, kept 4 h
#   bash scripts/nchc/nchc.sh up [--wf]        # upload code, weights, Sen1 data (+ WorldFloods with --wf),
#                                              #   nchc.env; build the venv on first use
#   bash scripts/nchc/nchc.sh run sen1 [opts]  # dry-run, then submit on Nano5 (you type the project code)
#   bash scripts/nchc/nchc.sh run smoke | run train [opts] CONFIG...
#   bash scripts/nchc/nchc.sh status           # jobs, today's history, tail of the newest logs
#   bash scripts/nchc/nchc.sh fetch            # pull Sen1 results + logs into artifacts/nchc, wandb sync
#   bash scripts/nchc/nchc.sh shell            # interactive shell on the login node, in the repo
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/env.sh"

[[ -f "$SCRIPT_DIR/nchc.env" && "$NCHC_USER" != CHANGE_ME ]] \
  || { echo "fill NCHC_USER / NCHC_ACCOUNT in scripts/nchc/nchc.env first" >&2; exit 1; }

REMOTE="$NCHC_USER@$NCHC_HOST"
read -r -a LOGIN_OPTS <<<"$(nchc_ssh_opts "$NCHC_LOGIN_PORT")"
read -r -a XFER_OPTS <<<"$(nchc_ssh_opts "$NCHC_XFER_PORT")"
SEN1_RUNS=(tm_l1c13_wf_s1f11 tm_s1grd_kuro_s1f11 tm_l1c13_s1f11 tm_s1grd_s1f11 tm_l1c13_s1grd_s1f11)

connected() { ssh "$@" -O check "$REMOTE" 2>/dev/null; }

need_login() {
  connected "${LOGIN_OPTS[@]}" || { echo "no login connection; run: nchc.sh connect" >&2; exit 1; }
}

# Run a command in the repo on the login node (with a tty when one is available).
remote() {
  need_login
  local tty=(-T)
  [[ -t 0 && -t 1 ]] && tty=(-t)
  ssh "${LOGIN_OPTS[@]}" "${tty[@]}" "$REMOTE" "cd $(printf %q "$NCHC_REPO") && $*"
}

cmd="${1:-}"
[[ $# -gt 0 ]] && shift
case "$cmd" in
  connect)
    for opts in LOGIN_OPTS XFER_OPTS; do
      declare -n o="$opts"
      if connected "${o[@]}"; then
        echo "already connected (${o[1]})"
      else
        echo "connecting to $REMOTE port ${o[1]} (2FA prompt follows)"
        ssh "${o[@]}" -fN "$REMOTE"
      fi
    done
    ;;
  up)
    bash "$SCRIPT_DIR/push.sh" code
    bash "$SCRIPT_DIR/push.sh" hf
    bash "$SCRIPT_DIR/push.sh" sen1
    [[ "${1:-}" == --wf ]] && bash "$SCRIPT_DIR/push.sh" data
    # nchc.env is gitignored, so push.sh code skips it; the scripts on Nano5 need it.
    sftp -P "$NCHC_XFER_PORT" "${XFER_OPTS[@]:2}" -b - "$REMOTE" \
      <<<"put $SCRIPT_DIR/nchc.env $NCHC_REPO/scripts/nchc/nchc.env"
    remote "test -x .venv/bin/python || bash scripts/nchc/setup_env.sh"
    echo "Nano5 is ready. Next: nchc.sh run sen1"
    ;;
  run)
    what="${1:-}"
    [[ $# -gt 0 ]] && shift
    case "$what" in
      sen1)  script="scripts/nchc/sen1_finetune.sh"; extra=() ;;
      train) script="scripts/nchc/submit.sh"; extra=() ;;
      smoke) script="scripts/nchc/submit.sh"; extra=(--smoke) ;;
      *) echo "usage: nchc.sh run sen1|smoke|train [options]" >&2; exit 2 ;;
    esac
    args=""
    for a in "${extra[@]}" "$@"; do args+=" $(printf %q "$a")"; done
    echo "== dry-run on Nano5"
    remote "bash $script$args --dry-run"
    # The real submission asks for the project code on this terminal; it cannot run unattended.
    [[ -t 0 ]] || { echo "submission needs your terminal; run this yourself" >&2; exit 1; }
    read -r -p "Submit these jobs? [y/N] " ok
    [[ "$ok" == y || "$ok" == Y ]] || { echo "not submitted"; exit 0; }
    remote "bash $script$args"
    ;;
  status)
    remote 'squeue -u $USER -o "%.10i %.24j %.9P %.8T %.10M %.10l %R";
      echo; sacct -u $USER -S today -X --format=JobID,JobName%24,State,Elapsed,ExitCode;
      echo; for f in $(ls -t artifacts/nchc_logs/*.out 2>/dev/null | head -3); do echo "== $f"; tail -n 5 "$f"; done'
    ;;
  fetch)
    bash "$SCRIPT_DIR/pull.sh" --results sen1_fusion_compare \
      --logs-only "${SEN1_RUNS[@]/#/terramind_sen1/}"
    shopt -s nullglob
    runs=("$LAB_REPO"/artifacts/nchc/models/terramind_sen1/*/wandb/offline-run-*)
    if [[ ${#runs[@]} -gt 0 ]]; then
      "$LAB_REPO/.venv/bin/wandb" sync "${runs[@]}"
    fi
    echo "results: $LAB_REPO/artifacts/nchc/results/sen1_fusion_compare"
    ;;
  shell)
    need_login
    ssh "${LOGIN_OPTS[@]}" -t "$REMOTE" "cd $(printf %q "$NCHC_REPO") && exec \$SHELL -l"
    ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
