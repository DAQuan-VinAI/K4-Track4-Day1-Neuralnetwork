"""data.py — nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES = 54
N_CLASSES = 7


def _check(X, y, name: str) -> None:
    assert X.ndim == 2 and X.shape[1] == N_FEATURES, f"{name}: X phải có shape (N, 54), hiện {X.shape}"
    assert X.dtype == np.float32, f"{name}: X phải là float32, hiện {X.dtype}"
    assert y.shape == (X.shape[0],), f"{name}: y phải có shape (N,), hiện {y.shape}"
    assert y.dtype == np.int64, f"{name}: y phải là int64, hiện {y.dtype}"
    assert y.min() >= 0 and y.max() <= N_CLASSES - 1, f"{name}: nhãn phải nằm trong 0..6"


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    tr = np.load(f"{processed_dir}/train.npz")
    ev = np.load(f"{processed_dir}/eval.npz")
    X_train_full, y_train_full = tr["X"], tr["y"]
    X_eval, y_eval, eval_row_id = ev["X"], ev["y"], ev["row_id"]
    _check(X_train_full, y_train_full, "train")
    _check(X_eval, y_eval, "eval")
    assert eval_row_id.shape == y_eval.shape and len(np.unique(eval_row_id)) == len(eval_row_id)
    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed)
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Không tính trên val/eval: thống kê của chúng lọt vào đầu vào của mô hình là rò rỉ thông tin,
    làm điểm val/eval lạc quan hơn so với dữ liệu thật sự chưa thấy.
    """
    num = X_tr[:, :N_NUMERIC].astype(np.float64)  # float64 để mean/std của ~370k mẫu không mất chính xác
    mean = num.mean(axis=0)
    std = num.std(axis=0)
    std[std == 0] = 1.0  # cột hằng: giữ nguyên thay vì chia cho 0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    out = X.copy()
    out[:, :N_NUMERIC] = (out[:, :N_NUMERIC] - mean) / std
    return out


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id, mean, std
    """
    X_full, y_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)

    mean, std = fit_standardizer(X_tr)  # chỉ trên phần train còn lại
    X_tr, X_val, X_eval = (apply_standardizer(a, mean, std) for a in (X_tr, X_val, X_eval))

    data = {
        "X_tr": torch.tensor(X_tr, dtype=torch.float32, device=device),
        "y_tr": torch.tensor(y_tr, dtype=torch.int64, device=device),
        "X_val": torch.tensor(X_val, dtype=torch.float32, device=device),
        "y_val": torch.tensor(y_val, dtype=torch.int64, device=device),
        "X_eval": torch.tensor(X_eval, dtype=torch.float32, device=device),
        "y_eval": torch.tensor(y_eval, dtype=torch.int64, device=device),
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
    }

    counts = np.bincount(y_tr, minlength=N_CLASSES)
    majority = int(counts.argmax())
    print(f"train {tuple(X_tr.shape)} | val {tuple(X_val.shape)} | eval {tuple(X_eval.shape)} | device {device}")
    print(f"luôn đoán lớp đa số (lớp {majority}): val accuracy = {(y_val == majority).mean():.4f}")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Hoán vị được sinh trên thiết bị của `generator` (CPU nếu không truyền) rồi chuyển sang thiết bị
    của X, nên cùng seed cho cùng thứ tự lô dù chạy trên CPU hay GPU.
    Lô cuối nhỏ hơn batch_size vẫn được dùng (không bỏ mẫu nào); loss lấy trung bình trong lô nên
    lô nhỏ chỉ làm gradient nhiễu hơn một chút ở đúng một bước mỗi epoch.
    """
    n = X.shape[0]
    if shuffle:
        perm = torch.randperm(n, generator=generator,
                              device=generator.device if generator is not None else "cpu").to(X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
