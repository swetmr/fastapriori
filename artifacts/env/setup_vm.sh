#!/usr/bin/env bash
# One-time setup of a Linux machine (tested target: Ubuntu 24.04, x86_64) for this bundle.
#
#   bash env/setup_vm.sh                 # fastapriori built from lib/fastapriori-0.2.0 (needs Rust)
#   bash env/setup_vm.sh --lib pypi      # fastapriori==0.2.0 wheel from PyPI (same code), once that version is on PyPI;
#                                        # skips Rust, which notebook 19 still needs
#   bash env/setup_vm.sh --venv DIR      # virtualenv location (default ~/.venvs/fastapriori)
#   bash env/setup_vm.sh --no-apt        # skip apt-get (no sudo); needs gcc, make, /usr/bin/time, python3-venv
#
# Builds the Borgelt Apriori/Eclat/FP-Growth binaries and LCM 5.3 from the sources in tools/,
# installs the pinned Python environment and verifies every shipped file against SHA256SUMS.
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB=source
VENV="$HOME/.venvs/fastapriori"
APT=1
RUST_TOOLCHAIN=1.95.0          # rustc recorded in the May 2026 raw results
while [ $# -gt 0 ]; do
  case "$1" in
    --lib) LIB="$2"; shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    --no-apt) APT=0; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done

echo "== system packages"
PKGS="build-essential time curl python3-venv python3-dev"
if [ "$APT" = 1 ]; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq $PKGS
else
  for c in gcc make /usr/bin/time; do command -v "$c" >/dev/null || { echo "missing $c"; exit 1; }; done
fi

echo "== python environment: $VENV"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q -r "$BUNDLE/env/requirements.txt"

echo "== fastapriori 0.2.0 ($LIB)"
if [ "$LIB" = pypi ]; then
  python -m pip install -q --only-binary fastapriori "fastapriori==0.2.0"
else
  if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
  fi
  rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal
  # build outside the bundle (it may be on a read-only or shared drive)
  CARGO_TARGET_DIR="$(mktemp -d)" RUSTUP_TOOLCHAIN="$RUST_TOOLCHAIN" \
    python -m pip install -q "$BUNDLE/lib/fastapriori-0.2.0"
fi
python - <<'PY'
import importlib.metadata as md
import fastapriori._fastapriori_rs as rs
v = md.version("fastapriori")
missing = [f for f in ("rust_compute_pairs_v2", "rust_eclat_pipeline_seeded") if not hasattr(rs, f)]
print("fastapriori", v, "| extension", rs.__file__)
assert v == "0.2.0", v
assert not missing, f"extension lacks {missing}"
PY

echo "== Borgelt binaries (apriori 6.31, eclat 5.24, fpgrowth 6.21) and LCM 5.3"
BUILD="$(mktemp -d)"
mkdir -p "$BUNDLE/tools/bin"
for a in apriori eclat fpgrowth; do
  mkdir -p "$BUILD/$a" && tar -xzf "$BUNDLE/tools/borgelt/$a.tar.gz" -C "$BUILD/$a"
  make -s -C "$BUILD/$a/$a/src" "$a" >/dev/null
  install -m 755 "$BUILD/$a/$a/src/$a" "$BUNDLE/tools/bin/$a"
done
python -m zipfile -e "$BUNDLE/tools/lcm/lcm53.zip" "$BUILD/lcm"
make -s -C "$BUILD/lcm" >/dev/null
install -m 755 "$BUILD/lcm/lcm" "$BUNDLE/tools/bin/lcm"
rm -rf "$BUILD"
ls -l "$BUNDLE/tools/bin"

echo "== verify shipped files"
# the repository copy has no data/: check the data files that are present, count the others
SUMS="$(mktemp)"
ABSENT=0
while IFS= read -r line; do
  f="${line#*  }"
  if [ "${f#data/}" != "$f" ] && [ ! -e "$BUNDLE/$f" ]; then
    ABSENT=$((ABSENT + 1))
  else
    printf '%s\n' "$line"
  fi
done < "$BUNDLE/SHA256SUMS" > "$SUMS"
if ! ( cd "$BUNDLE" && sha256sum --quiet -c "$SUMS" ); then
  echo "ERROR: shipped files differ from SHA256SUMS (see above); the bundle is not the published one." >&2
  exit 1
fi
rm -f "$SUMS"
echo "all shipped files match SHA256SUMS ($ABSENT data files absent)"

cat <<EOF

Setup complete. Next:
  source $VENV/bin/activate
  cd $BUNDLE/notebooks
  jupyter nbconvert --to notebook --execute --inplace 00_setup_and_verify.ipynb
See README.md for the order of the notebooks and how to start a new campaign.
EOF
