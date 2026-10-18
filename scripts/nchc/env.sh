# Shared settings for running this repo on NCHC Nano5 (晶創25).
# Sourced by every script in scripts/nchc/, both on the lab machine and on Nano5.
#
# Per-user values go in scripts/nchc/nchc.env (gitignored); copy nchc.env.example.

NCHC_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$NCHC_SCRIPT_DIR/nchc.env" ]]; then
  # shellcheck disable=SC1091
  source "$NCHC_SCRIPT_DIR/nchc.env"
fi

# --- Nano5 side -------------------------------------------------------------
: "${NCHC_USER:=$USER}"                       # iService 主機帳號
: "${NCHC_ACCOUNT:=}"                          # iService 計畫代號 (Slurm --account)
: "${NCHC_HOST:=nano5.nchc.org.tw}"
: "${NCHC_LOGIN_PORT:=22}"
: "${NCHC_XFER_PORT:=2222}"                    # 資料傳輸節點 (sftp)
: "${NCHC_WORK:=/work/$NCHC_USER}"             # 大容量工作區; 確認實際路徑後在 nchc.env 覆寫
: "${NCHC_REPO:=$NCHC_WORK/flood-uncertainty}"
: "${NCHC_DATA:=$NCHC_WORK/data}"
: "${NCHC_HF_HOME:=$NCHC_WORK/hf_cache}"

# --- lab side (source of the transfer and of the hard-coded config paths) ---
: "${LAB_REPO:=/home/NAS/house/ycchen-10014/repo/flood-uncertainty}"
: "${LAB_DATA:=/home/NAS/house/ycchen-10014/data}"
: "${LAB_HF_HOME:=$HOME/.cache/huggingface}"
# Sen1Floods11 v1.1 root used by configurations/terramind_sen1 and *_sen1.json (teammate's copy).
: "${LAB_SEN1:=/home/NAS/house/chlunchen-10030/datasets/sen1floods11_v1.1}"
: "${NCHC_SEN1:=$NCHC_DATA/sen1floods11_v1.1}"

# Runtime environment for jobs on Nano5. Call after sourcing this file.
nchc_runtime_env() {
  export PATH="$HOME/.local/bin:$PATH"
  export UV_PROJECT_ENVIRONMENT="$NCHC_REPO/.venv"
  export UV_CACHE_DIR="$NCHC_WORK/.cache/uv"
  export UV_PYTHON_INSTALL_DIR="$NCHC_WORK/.local/uv-python"
  export HF_HOME="$NCHC_HF_HOME"
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export PYTHONUNBUFFERED=1
  # Rewrites the /home/NAS/... paths baked into configurations/*.json
  # (see flood_uncertainty/utils/config_loader.py).
  # Extra "old=new;..." pairs (e.g. other members' dirs) go in NCHC_EXTRA_REMAP.
  export FLOOD_PATH_REMAP="$LAB_REPO=$NCHC_REPO;$LAB_DATA=$NCHC_DATA;$LAB_SEN1=$NCHC_SEN1${NCHC_EXTRA_REMAP:+;$NCHC_EXTRA_REMAP}"
}

# Compute nodes may have no internet: never let uv or huggingface_hub reach out.
nchc_offline_env() {
  export UV_NO_SYNC=1
  export UV_OFFLINE=1
  export HF_HUB_OFFLINE=1
}

# Billing guard: before any sbatch, the user must type the project code on a real
# terminal. There is no environment variable or flag that skips this, so scripts and
# agents cannot submit jobs on the project without the person at the keyboard.
nchc_confirm_account() {
  local summary="$1" typed
  if ! { : </dev/tty; } 2>/dev/null; then
    echo "refusing to submit: no terminal to confirm use of project $NCHC_ACCOUNT" >&2
    return 1
  fi
  {
    echo "About to submit on NCHC project $NCHC_ACCOUNT (billed):"
    echo "  $summary"
    printf 'Type the project code to confirm: '
  } >/dev/tty
  read -r typed </dev/tty
  if [[ "$typed" != "$NCHC_ACCOUNT" ]]; then
    echo "not confirmed; nothing submitted" >&2
    return 1
  fi
}

# ssh/sftp options sharing one authenticated connection (2FA once per session).
nchc_ssh_opts() {
  local port="$1"
  mkdir -p "$HOME/.ssh/cm"
  # Keepalives: the transfer node (2222) drops idle connections well before ControlPersist.
  echo "-p $port -o ControlMaster=auto -o ControlPersist=4h -o ControlPath=$HOME/.ssh/cm/%r@%h:%p -o ServerAliveInterval=60 -o ServerAliveCountMax=3"
}
