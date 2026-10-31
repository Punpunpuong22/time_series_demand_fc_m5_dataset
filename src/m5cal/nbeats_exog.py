"""N-BEATS có biến ngoại sinh (NBEATSx, thư viện neuralforecast) cho dự báo 1 ngày tới.

NBEATSx (Olivares và cộng sự, 2023) mở rộng N-BEATS (Oreshkin và cộng sự, 2020) để nhận biến biết trước
(`futr_exog_list`). Dự báo cho tập kiểm tra được tạo bằng `cross_validation` với h = 1, bước 1 ngày,
không huấn luyện lại (`refit=False`): mô hình học một lần trên phần trước tập kiểm tra, dùng `val_size`
ngày cuối của phần đó để dừng sớm, rồi dự báo từng ngày của tập kiểm tra.

`nf_cls`, `model_cls`, `loss` được truyền vào để kiểm thử được mà không cần neuralforecast.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_PARAMS = dict(input_size=28, mlp_width=256, max_steps=1000, val_check_steps=50,
                      early_stop_patience_steps=5, batch_size=32, windows_batch_size=1024,
                      learning_rate=1e-3, scaler_type="standard", accelerator="auto")


def build_nf_frame(df: pd.DataFrame, exog_cols, id_col: str = "item_id", date_col: str = "date",
                   y_col: str = "demand") -> pd.DataFrame:
    exog_cols = list(exog_cols)
    out = df[[id_col, date_col, y_col] + exog_cols].rename(columns={id_col: "unique_id", date_col: "ds", y_col: "y"})
    out["ds"] = pd.to_datetime(out["ds"])
    out[exog_cols] = out[exog_cols].astype("float32")
    return out.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def nbeatsx_cross_validation(df_nf: pd.DataFrame, exog_cols, test_size: int, val_size: int, seed: int = 1,
                             params: dict | None = None, nf_cls=None, model_cls=None, loss=None) -> pd.DataFrame:
    """Trả về (unique_id, ds, pred) cho `test_size` ngày cuối của mỗi chuỗi."""
    p = dict(DEFAULT_PARAMS, **(params or {}))
    if nf_cls is None or model_cls is None or loss is None:  # pragma: no cover - cần neuralforecast
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MSE
        from neuralforecast.models import NBEATSx
        nf_cls, model_cls, loss = nf_cls or NeuralForecast, model_cls or NBEATSx, loss or MSE()
    exog_cols = list(exog_cols)
    # Chuỗi quá ngắn (mặt hàng lên kệ muộn) làm cross_validation báo lỗi: bỏ ra, dòng kiểm tra của chúng
    # được điền bằng mô hình cơ sở khi ghép (xem `attach`).
    lengths = df_nf.groupby("unique_id")["ds"].transform("size")
    short = lengths < test_size + val_size + 1
    if short.any():
        print(f"  NBEATSx: bỏ {df_nf.loc[short, 'unique_id'].nunique()} chuỗi ngắn hơn {test_size + val_size + 1} ngày")
        df_nf = df_nf[~short]
    model = model_cls(h=1, input_size=p["input_size"], futr_exog_list=exog_cols or None,
                      stack_types=["identity", "identity", "identity"], n_blocks=[1, 1, 1],
                      mlp_units=3 * [[p["mlp_width"], p["mlp_width"]]], loss=loss,
                      max_steps=p["max_steps"], val_check_steps=p["val_check_steps"],
                      early_stop_patience_steps=p["early_stop_patience_steps"], batch_size=p["batch_size"],
                      windows_batch_size=p["windows_batch_size"], learning_rate=p["learning_rate"],
                      scaler_type=p["scaler_type"], random_seed=seed, start_padding_enabled=True,
                      enable_progress_bar=False, accelerator=p["accelerator"])
    nf = nf_cls(models=[model], freq="D")
    cv = nf.cross_validation(df=df_nf, n_windows=None, test_size=test_size, val_size=val_size,
                             step_size=1, refit=False)
    cv = cv.reset_index() if "unique_id" not in cv.columns else cv
    name = [c for c in cv.columns if c not in ("unique_id", "ds", "cutoff", "y", "index")][0]
    return cv[["unique_id", "ds", name]].rename(columns={name: "pred"})


def attach(test_eval: pd.DataFrame, preds: pd.DataFrame, col: str, id_col: str = "item_id",
           date_col: str = "date", fallback_col: str | None = None, max_missing: float = 0.05) -> pd.DataFrame:
    """Ghép dự báo vào khung đánh giá theo (mặt hàng, ngày).

    Dòng thiếu dự báo (ví dụ mặt hàng lên kệ quá muộn để ước lượng mô hình) được điền bằng cột
    `fallback_col` (mô hình cơ sở) nếu tỉ lệ thiếu không vượt `max_missing`; ngược lại báo lỗi.
    Số dòng đã điền được lưu trong `out.attrs["filled"]`.
    """
    p = preds.rename(columns={"unique_id": id_col, "ds": date_col, "pred": col})
    p[date_col] = pd.to_datetime(p[date_col]).astype("datetime64[ns]")
    p = p.drop_duplicates([id_col, date_col])
    out = test_eval.drop(columns=[col], errors="ignore").merge(p[[id_col, date_col, col]], on=[id_col, date_col], how="left")
    missing = int(out[col].isna().sum())
    if missing:
        frac = missing / len(out)
        if fallback_col is None or fallback_col not in out.columns or frac > max_missing:
            raise ValueError(f"{col}: thiếu dự báo cho {missing} dòng kiểm tra ({100 * frac:.1f}%)")
        print(f"  {col}: {missing} dòng ({100 * frac:.2f}%) không có dự báo, điền bằng {fallback_col}")
        out[col] = out[col].fillna(out[fallback_col])
    out[col] = out[col].clip(lower=0)
    out.attrs = dict(getattr(test_eval, "attrs", {}) or {})
    out.attrs.setdefault("filled", {})[col] = missing
    return out
