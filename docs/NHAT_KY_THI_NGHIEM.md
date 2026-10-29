# Nhật ký thí nghiệm

Ghi mỗi lần chạy, mỗi quyết định và mỗi lần thất bại để không lặp lại cách làm đã hỏng và để viết mục Phương pháp chính xác.
Mỗi mục ghi kèm mã commit (`git log --oneline -1`) để biết kết quả ứng với phiên bản mã nào.

## Mẫu một mục

```
### 2026-MM-DD · <mã thí nghiệm> · commit <abc1234>
Mục đích:
Cấu hình (tệp/tham số đã đổi):
Môi trường: Colab T4 / CPU, thời gian chạy, bộ nhớ đỉnh
Kết quả chính (đường dẫn tệp trong results/tables):
Vấn đề gặp phải và cách xử lý:
Quyết định / bước tiếp theo:
```

## Các quyết định đã chốt (từ báo cáo đề xuất)

| Ngày | Quyết định | Lý do |
| --- | --- | --- |
| 2026-10-24 | LightGBM trực tiếp theo 4 nhóm tầm, lag tối thiểu 7/14/21/28 | Cân bằng tập thông tin với LSTM; R2 (lag ≥ 28) giữ làm độ nhạy |
| 2026-10-24 | 13 cửa sổ đánh giá d_1578–d_1941, huấn luyện lại mỗi 3 cửa sổ | Đủ ngày sự kiện để phân tầng; vừa sức Colab |
| 2026-10-24 | ADI/CV² chỉ tính đến d_1521 | Chọn mẫu không dùng thông tin giai đoạn đánh giá |
| 2026-10-24 | Diễn giải chung bằng permutation importance theo nhóm | Saliency trên RNN kém tin cậy; TreeSHAP chỉ bổ sung cho LightGBM |
| 2026-10-25 | Đổi tên đề tài: trọng tâm là giá trị của thông tin lịch; LightGBM và LSTM là công cụ đo | Góp ý của giảng viên: so sánh trực tiếp hai họ không công bằng vì LightGBM được thiết kế cho dữ liệu bảng |
| 2026-10-25 | Thí nghiệm chính là notebook v3 (CA_1 FOODS, 100 mặt hàng, 1 ngày tới); pipeline 28 ngày trong `src/` là khung mở rộng | Báo cáo và slide dựa trên notebook đã chạy |
| 2026-10-25 | Giữ LSTM-A đọc thẳng cửa sổ 28 ngày log doanh số, không dùng lag dựng tay | Mỗi mô hình nhận đầu vào ở dạng tự nhiên của nó |
| 2026-10-25 | (v0.3.0) Đổi tên đề tài theo góp ý: đánh giá tác động của yếu tố lịch và sự kiện trên các mô hình học máy | Lịch luôn có trong tiền xử lý nhưng ít được đánh giá riêng |
| 2026-10-25 | (v0.3.0) Ba nhóm yếu tố lịch: G1 nội sinh, G2 ngoại sinh (sự kiện, lễ hội), G3 chính sách (SNAP); bỏ giá khỏi thiết kế | Bám theo bản chất ngày tháng: phân rã và gắn ngữ cảnh |
| 2026-10-25 | (v0.3.0) Bốn họ mô hình, mỗi họ một đại diện: ARIMA (có biến ngoại sinh), LightGBM, LSTM, N-BEATS (NBEATSx) | Không cần biện minh vì sao chọn riêng hai mô hình; kết luận khái quát hơn |
| 2026-10-25 | (v0.3.0) Bộ S0–S3 lồng nhau và D1, D2 bỏ một nhóm; nhãn loại ngày chỉ dùng để phân tầng đánh giá | Đo tác động tăng thêm và tác động biên của từng nhóm |
| 2026-10-25 | (v0.3.0) Giữ tầm dự báo 1 ngày | Tránh nhiễu do chiến lược đa bước và sai số tích lũy theo tầm |
| 2026-10-25 | Bổ sung: LSTM chọn từ 3 cấu hình (32/16, 64/32, 128/64); bootstrap khối tuần; permutation theo nhóm; DM LightGBM-A với LSTM-A; đo thời gian | Cân bằng ngân sách tinh chỉnh; trả lời RQ2–RQ4 |

## Nhật ký chạy

<!-- Thêm mục mới ở trên cùng -->

### 2026-10-25 · Notebook v4 (4 mô hình × 3 nhóm yếu tố lịch) · commit <điền sau khi chạy>
Mục đích: thí nghiệm chính theo thiết kế mới
Cấu hình: RUN, DROP_ONE, LSTM_GRID, NB_GRID như mặc định; USE_DRIVE = True
Môi trường: <Colab T4 / CPU>, thời gian từng mô hình (bảng v4_times.csv)
Kết quả chính: v4_results_main.csv, v4_group_effects.csv, v4_bootstrap_marginal.csv, fig_group_effects_heatmap.png
Vấn đề gặp phải và cách xử lý:
Quyết định / bước tiếp theo:

### 2026-10-25 · Mục 16 notebook v3 · commit <điền sau khi chạy>
Mục đích: thí nghiệm bổ sung (16.1–16.5)
Cấu hình: RUN_LSTM_TUNING = True, N_BOOT = 2000, PI_REPEATS = 3
Môi trường: <Colab T4 / CPU>, thời gian chạy, bộ nhớ đỉnh
Kết quả chính: supp_*.csv trong OUT_DIR (chép sang Drive)
Vấn đề gặp phải và cách xử lý:
Quyết định / bước tiếp theo:

### 2026-10 · Notebook v2 (mục 0–15) · trước khi có repo
Mục đích: thí nghiệm chính, ablation A/B/C cho LightGBM và LSTM
Cấu hình: CA_1, FOODS, 100 mặt hàng (5 nhóm doanh số × 20), chia 70/15/15, LOOKBACK 28, HORIZON 1, LSTM 3 seed
Môi trường: Colab CPU; LSTM 661–1.577 giây mỗi seed
Kết quả chính: LightGBM-C WAPE 59,57%, LSTM-C 60,67%, MA28 64,90%; A→B: LightGBM −3,33% WAPE (DM p < 0,0001), LSTM −0,83%; B→C không có ý nghĩa
Vấn đề gặp phải và cách xử lý: cặp LightGBM-A với LSTM-A chưa kiểm định DM; LSTM chưa được tinh chỉnh tương đương → bổ sung ở mục 16
Quyết định / bước tiếp theo: chạy mục 16 trên T4
