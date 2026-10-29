#!/usr/bin/env bash
# Tải dữ liệu M5 từ Kaggle vào data/raw/.
# Yêu cầu:
#   1. Có tài khoản Kaggle và đã bấm "Join Competition / I Understand and Accept"
#      tại https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules
#   2. Có tệp API token kaggle.json (Kaggle > Settings > API > Create New Token)
#      đặt tại ~/.kaggle/kaggle.json  (KHÔNG commit tệp này lên GitHub)
# Cách dùng:  bash scripts/download_m5.sh
set -euo pipefail

DEST="${1:-data/raw}"
mkdir -p "$DEST"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Cài Kaggle CLI..."
  pip install -q kaggle
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "Không thấy ~/.kaggle/kaggle.json. Xem README mục 'Dữ liệu'." >&2
  exit 1
fi
chmod 600 "$HOME/.kaggle/kaggle.json"

kaggle competitions download -c m5-forecasting-accuracy -p "$DEST"
unzip -o "$DEST/m5-forecasting-accuracy.zip" -d "$DEST"
rm -f "$DEST/m5-forecasting-accuracy.zip"

for f in calendar.csv sell_prices.csv sales_train_evaluation.csv; do
  if [ ! -f "$DEST/$f" ]; then
    echo "Thiếu $f sau khi giải nén" >&2
    exit 1
  fi
done
echo "Đã tải dữ liệu M5 vào $DEST"
ls -lh "$DEST"
