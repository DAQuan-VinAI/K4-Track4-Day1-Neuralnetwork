"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

N_CLASSES = 7

# Cấu hình mặc định = BASELINE (M-base). `lr` được chọn bằng val trong notebook rồi truyền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    scheduler=None,            # None | "cosine" (ghi vào notes của bảng nếu dùng)
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán. Cùng công thức với scripts/evaluate.py.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits, ở chế độ eval()."""
    model.eval()
    preds = [model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, X.shape[0], batch_size)]
    return torch.cat(preds)


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y, giống nn.MSELoss: không có hệ số 1/2 và lấy trung bình
               trên MỌI phần tử (B x 7), không chỉ trên B.
    reduction="sum" trả về tổng (dùng trong evaluate để cộng dồn qua các lô).
    """
    logits = logits.float()  # dưới autocast logits có thể là fp16/bf16; loss luôn tính ở fp32
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        target = F.one_hot(y, num_classes=logits.shape[1]).to(logits.dtype)
        return F.mse_loss(logits, target, reduction=reduction)
    raise ValueError(f"loss phải là 'ce' hoặc 'mse', hiện là {loss_name!r}")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt), no_grad và fp32.

    Dùng cho: train loss (trên toàn bộ train), val, và eval cuối cùng.
    """
    model.eval()
    n = X.shape[0]
    total_loss = 0.0
    cm = torch.zeros(N_CLASSES * N_CLASSES, dtype=torch.int64, device=X.device)
    for i in range(0, n, batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        logits = model(xb)
        total_loss += compute_loss(logits, yb, loss_name, reduction="sum").item()
        pred = logits.argmax(dim=1)
        cm += torch.bincount(yb * N_CLASSES + pred, minlength=N_CLASSES * N_CLASSES)
    cm = cm.reshape(N_CLASSES, N_CLASSES).cpu().numpy()
    denom = n * N_CLASSES if loss_name == "mse" else n  # MSE: trung bình trên mọi phần tử
    return {"loss": total_loss / denom, "acc": float(np.trace(cm) / n), "macro_f1": macro_f1_from_confusion(cm)}


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def run_experiment(cfg: dict, data: dict, verbose: bool = False) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình; khoá thiếu lấy từ DEFAULT_CFG
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, ... trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                     "grad_norm", "grad_norm_max", "clip_frac", "epoch_time_s"},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged", ...},
         "best_state": state_dict (trên CPU) của epoch có val_loss thấp nhất}

    Quy ước đo:
      - train_loss đo ở eval() trên TOÀN BỘ train sau mỗi epoch, nên cùng thang với val_loss.
      - grad_norm là chuẩn L2 toàn cục TRƯỚC khi clip, trung bình các bước của epoch; grad_norm_max là
        bước lớn nhất (thấy "gai"); clip_frac là tỉ lệ bước có chuẩn vượt ngưỡng clip.
      - epoch_time_s chỉ tính vòng lặp huấn luyện (không tính phần đánh giá cuối epoch).
      - val_acc / val_macro_f1 trong summary lấy ở best_epoch (dừng sớm theo val loss).
    Hàm này KHÔNG đụng tới X_eval.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    cfg["hidden"] = tuple(cfg["hidden"])
    if cfg["lr"] is None:
        raise ValueError("cfg['lr'] chưa được đặt")
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    device = X_tr.device
    precision, loss_name, clip_norm = cfg["precision"], cfg["loss"], cfg["clip_norm"]
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"precision không hợp lệ: {precision!r}")
    if precision != "fp32" and device.type not in ("cuda", "mps"):
        raise RuntimeError(f"precision={precision} cần GPU (cuda hoặc mps); thiết bị hiện tại là {device.type}")

    set_seed(cfg["seed"])
    model = MLP(hidden=cfg["hidden"], dropout=cfg["dropout"], init=cfg["init"]).to(device)
    if cfg["hidden"] in EXPECTED_PARAMS:
        assert count_params(model) == EXPECTED_PARAMS[cfg["hidden"]], "số tham số sai quy định"
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])
    steps_per_epoch = math.ceil(X_tr.shape[0] / cfg["batch"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], total_steps=steps_per_epoch * cfg["epochs"])
    use_scaler = precision == "fp16"
    scaler = torch.amp.GradScaler(device.type) if use_scaler else None
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    generator = torch.Generator().manual_seed(cfg["seed"])  # thứ tự xáo lô, tách khỏi RNG khởi tạo

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Loss bước 0: TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7 với CE và khởi tạo hợp lý
    step0_loss = evaluate(model, X_val, y_val, loss_name)["loss"]

    keys = ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
            "grad_norm", "grad_norm_max", "clip_frac", "epoch_time_s")
    history = {k: [] for k in keys}
    best_val_loss, best_epoch, best_state = math.inf, None, None
    diverged, diverged_epoch, skipped_steps = False, None, 0
    mps_mem = 0  # MPS không có API đo đỉnh bộ nhớ: lấy mức cấp phát lớn nhất đo ở cuối mỗi epoch huấn luyện

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        grad_norms = []
        _sync(device)
        t0 = time.perf_counter()
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
            if amp_dtype is None:
                loss = compute_loss(model(xb), yb, loss_name)
            else:
                with torch.autocast(device.type, dtype=amp_dtype):  # chỉ bọc forward + loss
                    loss = compute_loss(model(xb), yb, loss_name)
            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # đưa gradient về thang thật TRƯỚC khi đo/cắt
            else:
                loss.backward()
            gn = clip_gradients(model.parameters(), clip_norm)  # chuẩn TRƯỚC khi cắt
            if not math.isfinite(gn):
                if use_scaler:
                    # FP16 tràn số ở bước này: GradScaler tự bỏ qua bước và giảm hệ số nhân
                    skipped_steps += 1
                else:
                    diverged = True
                    break
            else:
                grad_norms.append(gn)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
        _sync(device)
        epoch_time = time.perf_counter() - t0
        if device.type == "mps":
            mps_mem = max(mps_mem, torch.mps.current_allocated_memory())

        if not diverged:
            tr = evaluate(model, X_tr, y_tr, loss_name)
            va = evaluate(model, X_val, y_val, loss_name)
            diverged = not (math.isfinite(tr["loss"]) and math.isfinite(va["loss"]))
        if diverged:
            diverged_epoch = epoch
            print(f"[{cfg['exp_id']}] loss/gradient thành NaN/inf ở epoch {epoch}: dừng sớm")
            break

        gn_arr = np.asarray(grad_norms) if grad_norms else np.asarray([float("nan")])
        history["epoch"].append(epoch)
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])
        history["val_macro_f1"].append(va["macro_f1"])
        history["grad_norm"].append(float(gn_arr.mean()))
        history["grad_norm_max"].append(float(gn_arr.max()))
        history["clip_frac"].append(float((gn_arr > clip_norm).mean()) if clip_norm is not None else 0.0)
        history["epoch_time_s"].append(epoch_time)
        if va["loss"] < best_val_loss:
            best_val_loss, best_epoch = va["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"[{cfg['exp_id']}] epoch {epoch:2d} train {tr['loss']:.4f} val {va['loss']:.4f} "
                  f"acc {va['acc']:.4f} f1 {va['macro_f1']:.4f} gn {gn_arr.mean():.3f} {epoch_time:.1f}s")

    has_epochs = best_epoch is not None
    i_best = best_epoch - 1 if has_epochs else None
    summary = {
        "step0_loss": step0_loss,
        "best_val_loss": best_val_loss if has_epochs else None,
        "best_epoch": best_epoch,
        "final_train_loss": history["train_loss"][-1] if has_epochs else None,
        "final_val_loss": history["val_loss"][-1] if has_epochs else None,
        "val_acc": history["val_acc"][i_best] if has_epochs else None,
        "val_macro_f1": history["val_macro_f1"][i_best] if has_epochs else None,
        "time_per_epoch_s": float(np.mean(history["epoch_time_s"])) if has_epochs else None,
        "peak_mem_MB": (torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda"
                        else mps_mem / 2**20 if device.type == "mps" else None),
        "diverged": diverged,
        "diverged_epoch": diverged_epoch,
        "amp_skipped_steps": skipped_steps,
        "n_params": count_params(model),
        "steps_per_epoch": steps_per_epoch,
        "device": device.type,
    }
    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    """
    row_id, preds = np.asarray(row_id), np.asarray(preds)
    assert row_id.shape == preds.shape, "row_id và preds phải cùng độ dài"
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1, "pred phải nằm trong 0..6"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame({"row_id": row_id.astype(np.int64), "pred": preds.astype(np.int64)}).to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str,
               repo_root: str | None = None, out_json: str | None = None) -> dict | None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Nếu truyền repo_root: chạy luôn `python scripts/evaluate.py --pred <pred_path> [--out <out_json>]`
    từ thư mục gốc repo và trả về nội dung out_json (dict). Ngược lại trả về None.
    """
    if result.get("best_state") is None:
        raise ValueError(f"{cfg.get('exp_id')}: không có best_state (lần chạy bị phân kỳ ngay từ đầu?)")
    device = data["X_eval"].device
    cfg = {**DEFAULT_CFG, **cfg}
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"], init=cfg["init"])
    model.load_state_dict(result["best_state"])
    model.to(device)
    preds = predict(model, data["X_eval"])  # fp32, eval mode (không dropout)
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)

    if repo_root is None:
        return None
    cmd = [sys.executable, "scripts/evaluate.py", "--pred", os.path.abspath(pred_path)]
    if out_json is not None:
        cmd += ["--out", os.path.abspath(out_json)]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"scripts/evaluate.py báo lỗi:\n{proc.stdout}\n{proc.stderr}")
    if out_json is None:
        return None
    with open(out_json) as f:
        return json.load(f)
