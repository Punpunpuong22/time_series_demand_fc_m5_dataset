# Đánh giá tác động của các yếu tố lịch và sự kiện trong dự báo nhu cầu bán lẻ trên các mô hình học máy

Mã nguồn cho đề tài **“Phân tích, đánh giá tác động của các yếu tố lịch và sự kiện trong bài toán dự báo nhu cầu bán lẻ trên các mô hình học máy”**, dữ liệu M5 (Walmart).

> Impact of calendar and event factors on retail demand forecasting across machine-learning model families.

Thông tin ngày tháng có mặt trong hầu hết các bài toán dự báo bán lẻ, nhưng thường chỉ nằm ở bước tiền xử lý mà không được đánh giá riêng. Ngày tháng là một biến liên tục, tăng mãi theo thời gian; muốn mô hình dùng được thì phải **phân rã** nó (thứ, ngày, tháng, mùa) và **gắn ngữ cảnh** cho nó (sự kiện, lễ hội, chính sách). Đề tài đo tác động của ba nhóm thuộc tính sinh ra từ ngày tháng trên bốn họ mô hình dự báo.

**Mở thí nghiệm chính trên Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Punpunpuong22/time_series_demand_fc_m5_dataset/blob/main/notebooks/main/m5_calendar_impact_4models_v4.ipynb)

---

## 0. Đọc trước: repo gồm ba phần

| Phần | Nội dung | Trạng thái |
| --- | --- | --- |
| **Thí nghiệm chính (v4)** | `notebooks/main/m5_calendar_impact_4models_v4.ipynb` và các mô-đun `calendar_groups.py`, `arima_exog.py`, `nbeats_exog.py`, `group_effects.py` | Mã hoàn chỉnh, đã chạy thử bằng dữ liệu và mô hình giả lập; **cần chạy trên Colab** để có kết quả |
| **Thí điểm (v3)** | `notebooks/main/m5_calendar_value_lightgbm_lstm_v3.ipynb`: LightGBM và LSTM, bộ A/B/C theo thiết kế cũ | Mục 0–15 đã chạy, kết quả dùng làm bằng chứng thí điểm |
| **Khung mở rộng** | `src/m5cal/` (data, splits, features, models, pipeline…), `experiments/`, `notebooks/01–03`: tầm 28 ngày, rolling-origin | Đã qua kiểm thử trên dữ liệu giả lập, chưa chạy trên dữ liệu thật |

## 1. Thiết kế nghiên cứu

### Ba nhóm yếu tố lịch

| Nhóm | Nội dung | Ví dụ thuộc tính |
| --- | --- | --- |
| G1 · Nội sinh (phái sinh từ ngày) | Phân rã ngày tháng thành các chu kỳ | thứ, ngày trong tháng, tuần trong năm, tháng, mùa, cuối tuần |
| G2 · Ngoại sinh (sự kiện, lễ hội) | Ngữ cảnh gắn với ngày | loại sự kiện (quốc gia, tôn giáo, văn hóa, thể thao), ngày lễ, trước sự kiện, khoảng cách tới/từ sự kiện |
| G3 · Chính sách trợ cấp | Tác động của chính sách SNAP | ngày phát SNAP, ngày thứ mấy trong đợt phát |

Thời tiết cũng là ngữ cảnh ngoại sinh quan trọng, nhưng M5 chỉ cho biết bang của cửa hàng, không có vị trí hay dữ liệu thời tiết; đây là hướng mở rộng.

### Bộ biến (ablation)

S0 = chỉ lịch sử nhu cầu · S1 = S0 + G1 · S2 = S1 + G2 · S3 = S2 + G3 · D1 = S3 − G1 · D2 = S3 − G2. So sánh lồng nhau cho tác động **tăng thêm**; bỏ một nhóm khỏi S3 cho tác động **biên**.

### Bốn họ mô hình, mỗi họ một đại diện

| Họ | Đại diện | Cách nhận biến lịch |
| --- | --- | --- |
| Thống kê cổ điển | ARIMA có biến ngoại sinh (SARIMAX, statsmodels) | biến giả của thứ, tháng, sự kiện, SNAP |
| Cây tăng cường | LightGBM (Tweedie) | cột số |
| Mạng hồi quy | LSTM hai nhánh (Keras) | kênh trong cửa sổ 28 ngày và nhánh biết trước của ngày dự báo |
| Kiến trúc mới | N-BEATS có biến ngoại sinh (NBEATSx, neuralforecast) | `futr_exog_list` |

Mỗi mô hình được chọn từ 3 cấu hình trên tập kiểm định (LightGBM: số lá; LSTM: số đơn vị; N-BEATS: độ rộng MLP; ARIMA: bậc theo AIC). Mạng nơ-ron chạy 3 seed.

### Câu hỏi nghiên cứu

| Mã | Câu hỏi | Trả lời ở đâu (notebook v4) |
| --- | --- | --- |
| RQ1 | Mỗi nhóm yếu tố lịch (G1, G2, G3) tác động bao nhiêu đến sai số dự báo, có ý nghĩa thống kê không? | Mục 14.2, 14.3 |
| RQ2 | Tác động đó khác nhau thế nào giữa bốn họ mô hình? | Mục 14.2 (hình nhiệt mô hình × nhóm), 14.5 |
| RQ3 | Tác động tập trung vào loại ngày nào, ngày SNAP hay ngày thường? | Mục 14.3, 14.4 |
| RQ4 | Chi phí tính toán của mỗi mô hình? | Mục 14.5 |

Tầm dự báo 1 ngày tới: đây là điều kiện sạch nhất để tách tác động của lịch, vì dự báo nhiều bước kéo theo lựa chọn chiến lược (đệ quy, trực tiếp, nhiều đầu ra) và sai số tích lũy theo tầm, hai yếu tố này chi phối độ chính xác và khác nhau giữa các họ mô hình (Ben Taieb và cộng sự, 2012; Ben Taieb & Atiya, 2016).

### Kết quả thí điểm (v3, thiết kế cũ, chỉ LightGBM và LSTM)

| Mô hình | WAPE % bộ A (lịch sử) | WAPE % bộ B (+ lịch và sự kiện) | WAPE % bộ C (+ SNAP và giá) |
| --- | --- | --- | --- |
| LightGBM | 61,68 | 59,63 | 59,57 |
| LSTM (3 seed) | 61,13 | 60,63 | 60,67 |

Bộ B của thiết kế cũ gộp G1 và G2; bộ C gộp G3 với giá. Thiết kế v4 tách riêng ba nhóm và bỏ giá.

## 2. Cấu trúc repo

```
time_series_demand_fc_m5_dataset/
├── notebooks/main/
│   ├── m5_calendar_impact_4models_v4.ipynb      # THÍ NGHIỆM CHÍNH: 4 mô hình × 3 nhóm yếu tố lịch
│   └── m5_calendar_value_lightgbm_lstm_v3.ipynb  # thí điểm (thiết kế cũ, đã chạy)
├── src/m5cal/
│   ├── calendar_groups.py   # ba nhóm yếu tố lịch, ba cách biểu diễn, bộ S0–S3, D1, D2
│   ├── arima_exog.py        # ARIMA có biến ngoại sinh, dự báo một bước không huấn luyện lại
│   ├── nbeats_exog.py       # NBEATSx qua neuralforecast, cross-validation bước 1 ngày
│   ├── group_effects.py     # tác động tăng thêm / biên của từng nhóm, trên mọi ngày và ngày liên quan
│   ├── supplementary.py     # DM, bootstrap khối tuần, permutation importance, đo thời gian
│   └── ...                  # khung mở rộng (data, splits, features, models, pipeline...)
├── tests/                   # kiểm thử, gồm test_calendar_groups.py, test_supplementary.py, test_leakage.py
├── docs/                    # HUONG_DAN_GIT.md, NHAT_KY_THI_NGHIEM.md
└── ...
```

## 3. Chạy thí nghiệm chính trên Google Colab

1. Đưa repo lên GitHub theo [docs/HUONG_DAN_GIT.md](docs/HUONG_DAN_GIT.md).
2. Colab → tab **GitHub** → mở `notebooks/main/m5_calendar_impact_4models_v4.ipynb`.
3. Runtime → Change runtime type → **T4 GPU**.
4. Ở ô cấu hình, đặt `USE_DRIVE = True` để lưu dự báo của từng mô hình sang Google Drive: phiên Colab bị ngắt thì chạy lại, các mô hình đã xong sẽ được đọc lại thay vì huấn luyện lại.
5. Có thể chạy từng họ mô hình bằng `RUN = {...}` và bật/tắt bộ bỏ một nhóm bằng `DROP_ONE = {...}` (mặc định LSTM chỉ chạy bốn bộ lồng nhau vì tốn thời gian).
6. Runtime → Run all. Ô N-BEATS tự cài `neuralforecast` nếu chưa có.
7. Ghi kết quả vào `docs/NHAT_KY_THI_NGHIEM.md`; hướng dẫn đưa vào báo cáo ở mục 16 của notebook.

Thời gian ước tính trên T4: ARIMA 20–60 phút (CPU, song song), LightGBM vài phút, LSTM 1–2 giờ, N-BEATS 30–60 phút.

## 4. Khung mở rộng: chạy trên Google Colab

1. Đưa repo lên GitHub theo [docs/HUONG_DAN_GIT.md](docs/HUONG_DAN_GIT.md).
2. Mở `notebooks/01_setup_and_eda.ipynb` trên Colab: vào https://colab.research.google.com → tab **GitHub** → dán đường dẫn repo → chọn notebook.
3. Ô đầu tiên đã đặt sẵn `GITHUB_USER = "Punpunpuong22"` và `REPO = "time_series_demand_fc_m5_dataset"`.
4. Runtime → Change runtime type → **T4 GPU**. Chạy lần lượt từng ô.
5. Sau đó dùng `02_run_experiments.ipynb` để chạy thí nghiệm và `03_results_analysis.ipynb` để phân tích.

Kết quả được lưu trong Google Drive (`MyDrive/M5_project/results`) nên không mất khi Colab ngắt kết nối.

## 5. Chạy trên máy tính và kiểm thử

```bash
git clone https://github.com/Punpunpuong22/time_series_demand_fc_m5_dataset.git
cd time_series_demand_fc_m5_dataset
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m pytest -q                                    # chạy kiểm thử
```

Máy văn phòng cấu hình thấp chạy được baseline và LightGBM trên tập con nhỏ; LSTM nên chạy trên Colab GPU.

## 6. Dữ liệu

Dữ liệu M5 **không** được đưa lên repo (quy định của Kaggle và dung lượng ~450 MB). Có hai cách:

* **Kaggle API**: đăng nhập Kaggle, đồng ý luật cuộc thi tại https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules, tạo API token (Settings → API → Create New Token) để có `kaggle.json`, đặt vào `~/.kaggle/`, rồi chạy `bash scripts/download_m5.sh`.
* **Google Drive**: sao chép `calendar.csv`, `sell_prices.csv`, `sales_train_evaluation.csv` vào `MyDrive/M5_project/data_raw/`; notebook sẽ tự nối thư mục này vào `data/raw`.

Không bao giờ commit `kaggle.json` (đã có trong `.gitignore`).

## 7. Khung mở rộng: ma trận thí nghiệm (đề cương ban đầu, chưa chạy trên dữ liệu thật)

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

## 8. Kiểm thử chống rò rỉ thông tin (khung mở rộng)

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

## 9. Tái lập

* Seed cố định cho numpy, random, PyTorch và LightGBM (`seeds` trong `base.yaml`).
* Danh sách chuỗi đã lấy mẫu lưu tại `results/series_ids.csv` và được dùng lại ở mọi lần chạy.
* Mỗi lần chạy ghi `results/runs/<exp>__seed<k>.json` gồm cấu hình, số vòng/epoch, thời gian, phiên bản thư viện, GPU.
* Ghi các quyết định và lần chạy thất bại vào `docs/NHAT_KY_THI_NGHIEM.md`.

## 10. Trạng thái kiểm chứng của mã

* **Thí nghiệm chính v4**: 9 kiểm thử riêng cho nhóm yếu tố lịch, ARIMAX, NBEATSx và bảng tác động (ARIMAX và NBEATSx kiểm thử bằng lớp giả lập vì môi trường kiểm thử không có statsmodels và neuralforecast). Notebook v4 đã chạy trọn vẹn trên dữ liệu M5 giả lập với statsmodels, LightGBM và neuralforecast giả lập; ô LSTM cần TensorFlow nên chưa chạy thử ngoài Colab. Toàn bộ bộ kiểm thử: 63 đạt, 1 bỏ qua.
* **Thí điểm v3**: mục 0–15 là notebook đã chạy (giữ nguyên kết quả). Các hàm của mục 16 (`supplementary.py`) có 6 kiểm thử riêng; mục 16 đã được chạy thử với mô hình giả lập (thay LightGBM và TensorFlow) để kiểm tra luồng dữ liệu, riêng phần dò cấu hình LSTM (16.4) cần TensorFlow nên **chưa được chạy thử** ngoài Colab.
* **Khung mở rộng**: phần lõi (dữ liệu, loại ngày, cửa sổ, đặc trưng, baseline, độ đo, kiểm định, tồn kho) đã qua kiểm thử trên dữ liệu **giả lập** định dạng M5; quy trình baseline → đánh giá đã chạy trọn vẹn. Toàn bộ bộ kiểm thử: 54 đạt, 1 bỏ qua (LSTM của khung mở rộng cần PyTorch).
* `models/lgbm.py` và `models/lstm.py` đã được viết và có kiểm thử chạy thử (`test_models_smoke.py`), nhưng **cần được chạy lần đầu trên Colab** (nơi có LightGBM và PyTorch) trước khi dùng cho kết quả. Nếu có lỗi, ghi lại thông báo lỗi vào nhật ký thí nghiệm.
* `models/deepar.py` (R1) là mã thử nghiệm, phụ thuộc phiên bản GluonTS.
* Kết quả trên dữ liệu giả lập **không** phải kết quả nghiên cứu.

## 11. Trích dẫn

Xem `CITATION.cff`. Dữ liệu: Makridakis, Spiliotis & Assimakopoulos (2022), *International Journal of Forecasting* 38(4), doi:10.1016/j.ijforecast.2021.11.013.

## 12. Giấy phép

Mã nguồn theo giấy phép MIT. Dữ liệu M5 thuộc quy định của cuộc thi trên Kaggle.
