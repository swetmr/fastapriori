#!/usr/bin/env bash
# Rebuild the eight real datasets from their public sources (see data/SOURCES.md) and verify each
# file against SHA256SUMS. Files that are already present and correct are left alone; a present
# file with a different hash is an error (nothing is ever overwritten).
#
#   source ~/.venvs/fastapriori/bin/activate          # needs pandas + openpyxl (env/requirements.txt)
#   bash env/fetch_real_data.sh                                   # fill data/real/
#   bash env/fetch_real_data.sh --dest /tmp/real                  # rebuild elsewhere, e.g. to check
#   bash env/fetch_real_data.sh --instacart-prior PATH/order_products__prior.csv
#
# Instacart is not shipped and needs a Kaggle account: download it from
# https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis
# and pass
# order_products__prior.csv with --instacart-prior, or have the `kaggle` CLI configured.
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$BUNDLE/data/real"
PRIOR=""
PY="${PYTHON:-python}"
while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --instacart-prior) PRIOR="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
mkdir -p "$DEST"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
UA="Mozilla/5.0 (X11; Linux x86_64) fastapriori-artifacts"   # SPMF refuses curl's default agent
SPMF=https://www.philippe-fournier-viger.com/spmf/publicdatasets
MISSING=()

expected() { grep "  data/real/$1\$" "$BUNDLE/SHA256SUMS" | cut -c1-64; }
have_ok() {  # 0 = present and verified, 1 = absent; exits on a mismatching file
  local f="$DEST/$1"
  [ -e "$f" ] || return 1
  if [ "$(sha256sum "$f" | cut -c1-64)" = "$(expected "$1")" ]; then
    echo "ok       $1 (already present)"; return 0
  fi
  echo "ERROR    $f exists but differs from SHA256SUMS; move it away first" >&2; exit 1
}
install_verified() {  # $1 = target name, $2 = candidate file
  if [ "$(sha256sum "$2" | cut -c1-64)" != "$(expected "$1")" ]; then
    echo "ERROR    $1: rebuilt file does not match SHA256SUMS (source changed?)" >&2; exit 1
  fi
  mv "$2" "$DEST/$1"; echo "fetched  $1 (verified)"
}
download() { curl -fsSL -A "$UA" --retry 3 --max-time 1800 -o "$2" "$1"; }

# --- plain downloads, byte-identical ----------------------------------------------------
while read -r name url; do
  have_ok "$name" && continue
  download "$url" "$WORK/$name" || { echo "ERROR    $name: download failed ($url)" >&2; exit 1; }
  install_verified "$name" "$WORK/$name"
done <<EOF
groceries.csv https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv
BMS1.txt $SPMF/BMS1_itemset_mining.txt
BMS2.txt $SPMF/BMS2_itemset_mining.txt
retail.txt $SPMF/retail.txt
kosarak.dat $SPMF/kosarak.dat
chainstore.txt $SPMF/chainstoreFIM.txt
EOF

# --- Online Retail II (UCI 502), sheet "Year 2009-2010" ---------------------------------
if ! have_ok online-retail.csv; then
  download "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip" "$WORK/or2.zip" \
    || { echo "ERROR    online-retail.csv: download from UCI failed" >&2; exit 1; }
  "$PY" - "$WORK/or2.zip" "$WORK/online-retail.csv" <<'PY'
import sys, zipfile
import pandas as pd
zp, out = sys.argv[1:3]
with zipfile.ZipFile(zp) as z, z.open("online_retail_II.xlsx") as f:
    x = pd.read_excel(f, sheet_name="Year 2009-2010", dtype={"Invoice": str, "StockCode": str})
x["InvoiceDate"] = pd.to_datetime(x["InvoiceDate"]).dt.strftime("%m/%d/%y %I:%M %p")
x["Customer ID"] = x["Customer ID"].astype("Int64")
x["Price"] = [r[:-2] if (r := repr(float(v))).endswith(".0") else r for v in x["Price"]]
x.drop(columns=["Country"]).to_csv(out, index=False, lineterminator="\n")
PY
  install_verified online-retail.csv "$WORK/online-retail.csv"
fi

# --- Instacart (Kaggle) -----------------------------------------------------------------
if ! have_ok instacart_orders.csv; then
  if [ -z "$PRIOR" ] && command -v kaggle >/dev/null 2>&1; then
    kaggle datasets download -d psparks/instacart-market-basket-analysis -f order_products__prior.csv -p "$WORK" \
      && { [ -f "$WORK/order_products__prior.csv" ] || unzip -q -o "$WORK/order_products__prior.csv.zip" -d "$WORK" 2>/dev/null \
           || "$PY" -m zipfile -e "$WORK/order_products__prior.csv.zip" "$WORK"; } \
      && PRIOR="$WORK/order_products__prior.csv" || true
  fi
  if [ -n "$PRIOR" ] && [ -f "$PRIOR" ]; then
    "$PY" - "$PRIOR" "$WORK/instacart_orders.csv" <<'PY'
import sys
import pandas as pd
src, out = sys.argv[1:3]
d = pd.read_csv(src, usecols=["order_id", "product_id"])
d.rename(columns={"order_id": "txn_id", "product_id": "item"}).to_csv(out, index=False, lineterminator="\r\n")
PY
    install_verified instacart_orders.csv "$WORK/instacart_orders.csv"
  else
    echo "skipped  instacart_orders.csv: needs Kaggle's order_products__prior.csv (--instacart-prior PATH)"
    MISSING+=(instacart_orders.csv)
  fi
fi

if [ ${#MISSING[@]} -gt 0 ]; then
  echo "incomplete: ${MISSING[*]}" >&2; exit 3
fi
echo "all 8 real datasets present and verified in $DEST"
