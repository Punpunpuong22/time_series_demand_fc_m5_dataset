# Calendar-aware Daily Retail Demand Forecasting on M5

Mã nguồn cho đề tài **“Dự báo nhu cầu bán lẻ theo ngày có xét đặc trưng lịch đều đặn và bất thường: so sánh có kiểm soát thông tin giữa LightGBM và LSTM trên bộ dữ liệu M5 (Walmart)”**.

> Regular versus irregular calendar effects in daily retail demand forecasting: an information-controlled comparison of LightGBM and LSTM on the M5 data.

Repo này hiện thực toàn bộ mục 6–7 của báo cáo đề xuất: dữ liệu và phân loại ngày lịch, chia dữ liệu rolling-origin 13 cửa sổ, cân bằng tập thông tin giữa hai mô hình, thiết kế giai thừa 2 mô hình × 3 bộ đặc trưng, độ đo RMSSE phân tầng theo loại ngày, kiểm định thống kê, diễn giải và mô phỏng tồn kho.

---

## 1. Câu hỏi nghiên cứu

| Mã | Câu hỏi | Tệp kết quả |
| --- | --- | --- |
| RQ1 | Khi cùng tập thông tin, LightGBM và LSTM khác nhau thế nào so với baseline? | `tables/overall.csv`, `dm_tests.csv`, `bootstrap_wrmsse_diff.csv` |
| RQ2 | Đặc trưng lịch đóng góp bao nhiêu cho mỗi họ mô hình, có tương tác không? | `tables/by_daytype.csv`, notebook 03 (mô hình hỗn hợp) |
| RQ3 | Lợi thế tương đối thay đổi theo loại ngày và nhóm nhu cầu thế nào? | `tables/by_daytype.csv`, `by_demand_class.csv` |
| RQ4 | Khác biệt độ chính xác có thành khác biệt tồn kho không? | `tables/inventory.csv` |

## 2. Cấu trúc repo

```
m5-calendar-forecasting/
├── README.md                  # tệp này
├── LICENSE, CITATION.cff      # giấy phép MIT, cách trích dẫn
├── pyproject.toml             # cài gói: pip install -e .
├── requirements.txt
├── configs/
│   ├── base.yaml              # đường dẫn, tập con, cửa sổ, seed, tồn kho
│   ├── experiments.yaml       # ma trận thí nghiệm B0, B1, L-*, S-*, R1, R2
│   ├── lgbm.yaml              # siêu tham số LightGBM + không gian tìm kiếm
│   └── lstm.yaml              # siêu tham số LSTM + không gian tìm kiếm
├── scripts/
│   ├── download_m5.sh         # tải M5 từ Kaggle
│   └── make_synthetic.py      # sinh dữ liệu GIẢ LẬP để chạy thử
├── src/m5cal/
│   ├── data.py                # đọc, kiểm tra lược đồ, xây ma trận chuỗi × ngày
│   ├── daytypes.py            # đặc trưng lịch, nhãn loại ngày (Bảng 9)
│   ├── sampling.py            # ADI/CV², lấy mẫu phân tầng
│   ├── splits.py              # 2 cửa sổ tinh chỉnh + 13 cửa sổ đánh giá (Bảng 10)
│   ├── features.py            # đặc trưng LightGBM và cửa sổ LSTM (chống rò rỉ)
│   ├── models/                # baselines, lgbm, lstm, deepar
│   ├── metrics.py             # MAE, RMSE, RMSSE, WRMSSE, RMSSE phân tầng
│   ├── stats_tests.py         # Diebold–Mariano (HLN), BH, Friedman–Nemenyi, bootstrap, mixed model
│   ├── explain.py             # permutation importance theo nhóm, TreeSHAP
│   ├── inventory.py           # mô phỏng order-up-to
│   └── pipeline.py            # điều phối chạy và đánh giá
├── experiments/
│   ├── run.py                 # chạy thí nghiệm
│   └── evaluate.py            # sinh mọi bảng kết quả
├── tests/                     # kiểm thử, gồm bộ chống rò rỉ test_leakage.py
├── notebooks/                 # 01 EDA, 02 chạy thí nghiệm, 03 phân tích (Google Colab)
├── docs/
│   ├── HUONG_DAN_GIT.md       # hướng dẫn đưa repo lên GitHub từng bước
│   └── NHAT_KY_THI_NGHIEM.md  # mẫu nhật ký thí nghiệm
└── .github/workflows/tests.yml  # tự chạy kiểm thử mỗi lần push
```

Nguyên tắc: **toàn bộ logic nằm trong `src/m5cal/`**, notebook chỉ gọi hàm. Nhờ vậy mã được kiểm thử, dễ đọc lại và cho kết quả giống nhau giữa các lần chạy.

## 6. Thí nghiệm

```bash
# Chạy thử toàn bộ quy trình trên dữ liệu GIẢ LẬP (vài giây cho baseline)
python scripts/make_synthetic.py --out data/synthetic
python experiments/run.py --exp B0 B1 L-HCP S-HCP --seeds 1 --raw-dir data/synthetic --n-series 30 --max-eval 2 --out results_smoke
python experiments/evaluate.py --raw-dir data/synthetic --out results_smoke --pairs L-HCP:S-HCP

# Dữ liệu thật
python experiments/run.py --exp B0 B1                          # baseline
python experiments/run.py --exp L-H L-HC L-HCP --seeds 1 2 3 4 5
python experiments/run.py --exp S-H S-HC S-HCP --seeds 1 2 3 4 5
python experiments/run.py --exp R2 --seeds 1 2 3               # độ bền: lag ≥ 28
python experiments/evaluate.py                                 # bảng vào results/tables/
```

| Mã | Mô hình | Bộ đặc trưng | Ghi chú |
| --- | --- | --- | --- |
| B0 | Seasonal Naive | — | benchmark của M5 |
| B1 | Croston–SBA | — | baseline cho nhu cầu gián đoạn |
| L-H, L-HC, L-HCP | LightGBM trực tiếp 4 nhóm tầm | H / HC / HCP | lag tối thiểu 7, 14, 21, 28 |
| S-H, S-HC, S-HCP | LSTM encoder–decoder | H / HC / HCP | 28 đầu ra một lần |
| R1 | DeepAR (GluonTS) | HCP | thử nghiệm, cần `pip install "gluonts[torch]"` |
| R2 | LightGBM lag ≥ 28 | HCP | thiết lập phổ biến trong M5 |

H = lịch sử nhu cầu + định danh; HC = + lịch đều đặn, lịch bất thường, SNAP; HCP = + giá.

**Cửa sổ**: tinh chỉnh d_1522–d_1577 (2 cửa sổ); đánh giá d_1578–d_1913 (12 cửa sổ); kiểm tra cuối d_1914–d_1941. Huấn luyện lại sau mỗi 3 cửa sổ; giữa hai lần, mô hình vẫn nhận doanh số mới nhất đến mốc làm đầu vào.

**Cân bằng thông tin**: LightGBM dùng một booster cho mỗi nhóm tầm `[a, b]` với lag tối thiểu `b`; đặt `horizon_groups: daily` trong `experiments.yaml` để có 28 booster với lag tối thiểu đúng bằng h (cân bằng tuyệt đối với LSTM, tốn thời gian hơn).

## 7. Kiểm thử chống rò rỉ thông tin

`tests/test_leakage.py` thay đổi ngẫu nhiên toàn bộ doanh số và giá **sau** mốc dự báo, rồi kiểm tra rằng không đặc trưng nào thay đổi theo. Cụ thể:

| Kiểm thử | Bảo đảm |
| --- | --- |
| `test_lgbm_prediction_features_ignore_future_sales` | Đặc trưng dòng dự báo của LightGBM không phụ thuộc doanh số sau mốc (3 chiến lược × 2 bộ đặc trưng) |
| `test_lgbm_training_features_ignore_future` | Dòng huấn luyện không chứa thông tin sau mốc |
| `test_price_normalisation_uses_only_past` | Giá tham chiếu chỉ tính trên dữ liệu đến mốc |
| `test_last_known_price_mode_blocks_future_prices` | Chế độ độ nhạy `last_known` không thấy giá tương lai |
| `test_insufficient_lag_raises` | Lag nhỏ hơn tầm dự báo bị chặn bằng lỗi |
| `test_masked_sales_hides_future_and_prelaunch` | Ma trận tính đặc trưng không chứa ngày sau mốc |
| `test_seq_inputs_ignore_future_sales` | Đầu vào và hệ số chuẩn hóa của LSTM chỉ dùng quá khứ |
| `test_seq_training_pairs_targets_before_cutoff` | Mục tiêu huấn luyện LSTM luôn ≤ mốc |
| `test_eval_models_never_trained_on_eval_targets` | Mô hình của cửa sổ k không bao giờ huấn luyện trên cửa sổ k |

Bộ kiểm thử còn gồm `test_splits.py`, `test_data_daytypes.py`, `test_metrics_stats.py` (giá trị độ đo tính tay, DM, Friedman, bootstrap, tồn kho) và `test_models_smoke.py` (chạy thử LightGBM và LSTM).

## 8. Tái lập

* Seed cố định cho numpy, random, PyTorch và LightGBM (`seeds` trong `base.yaml`).
* Danh sách chuỗi đã lấy mẫu lưu tại `results/series_ids.csv` và được dùng lại ở mọi lần chạy.
* Mỗi lần chạy ghi `results/runs/<exp>__seed<k>.json` gồm cấu hình, số vòng/epoch, thời gian, phiên bản thư viện, GPU.
* Ghi các quyết định và lần chạy thất bại vào `docs/NHAT_KY_THI_NGHIEM.md`.

## 9. Trạng thái kiểm chứng của mã

* Phần lõi (dữ liệu, loại ngày, cửa sổ, đặc trưng, baseline, độ đo, kiểm định, tồn kho) đã chạy và qua 47 kiểm thử trên dữ liệu **giả lập** định dạng M5; quy trình baseline → đánh giá đã chạy trọn vẹn.
* `models/lgbm.py` và `models/lstm.py` đã được viết và có kiểm thử chạy thử (`test_models_smoke.py`), nhưng **cần được chạy lần đầu trên Colab** (nơi có LightGBM và PyTorch) trước khi dùng cho kết quả. Nếu có lỗi, ghi lại thông báo lỗi vào nhật ký thí nghiệm.
* `models/deepar.py` (R1) là mã thử nghiệm, phụ thuộc phiên bản GluonTS.
* Kết quả trên dữ liệu giả lập **không** phải kết quả nghiên cứu.

## 10. Trích dẫn

Xem `CITATION.cff`. Dữ liệu: Makridakis, Spiliotis & Assimakopoulos (2022), *International Journal of Forecasting* 38(4), doi:10.1016/j.ijforecast.2021.11.013.

## 11. Giấy phép

Mã nguồn theo giấy phép MIT. Dữ liệu M5 thuộc quy định của cuộc thi trên Kaggle.
