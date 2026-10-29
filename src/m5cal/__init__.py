"""m5cal — Calendar-aware daily retail demand forecasting on the M5 data.

Gói này chứa toàn bộ logic của nghiên cứu: đọc dữ liệu, xây dựng đặc trưng,
chia cửa sổ rolling-origin, mô hình (Seasonal Naive, Croston-SBA, LightGBM,
LSTM), độ đo, kiểm định thống kê, diễn giải và mô phỏng tồn kho.
Notebook Colab chỉ gọi các hàm trong gói này.
"""

__version__ = "0.3.0"
