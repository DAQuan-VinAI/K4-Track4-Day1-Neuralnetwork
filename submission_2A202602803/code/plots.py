"""plots.py — ảnh biểu đồ của từng thí nghiệm và ảnh chồng theo nhóm.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt

# Bảng màu phân loại, gán theo thứ tự cố định (tối đa 8 đường trên một trục)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_GRID = dict(color="#d9d8d2", linewidth=0.6)
_RC = {"axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
       "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8}

METRIC_LABELS = {
    "train_loss": "train loss (eval mode)", "val_loss": "val loss", "val_acc": "val accuracy",
    "val_macro_f1": "val macro-F1", "grad_norm": "grad norm trung bình (trước clip)",
    "grad_norm_max": "grad norm lớn nhất (trước clip)", "epoch_time_s": "thời gian epoch (s)",
}


def _cfg_text(cfg: dict) -> str:
    hidden = "-".join(str(h) for h in cfg["hidden"])
    parts = [f"{cfg['optimizer']} lr={cfg['lr']:g}", f"batch={cfg['batch']}", f"loss={cfg['loss']}",
             f"hidden={hidden}", f"init={cfg['init']}", f"dropout={cfg['dropout']:g}",
             f"clip={cfg['clip_norm'] if cfg['clip_norm'] is not None else 'none'}",
             f"{cfg['precision']}", f"wd={cfg['weight_decay']:g}", f"seed={cfg['seed']}"]
    if cfg.get("scheduler"):
        parts.append(f"sched={cfg['scheduler']}")
    return " · ".join(parts)


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có 3 ô:
         (1) train_loss và val_loss theo epoch
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm trung bình và lớn nhất của epoch (đo TRƯỚC khi clip)
    Đường đứt dọc đánh dấu best_epoch (val loss thấp nhất).
    """
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        status = "PHÂN KỲ ở epoch %s" % s.get("diverged_epoch") if s["diverged"] else \
            f"best epoch {s['best_epoch']}: val loss {s['best_val_loss']:.4f}, " \
            f"acc {s['val_acc']:.4f}, macro-F1 {s['val_macro_f1']:.4f}"
        fig.suptitle(f"{cfg['exp_id']}  [{cfg['group']}]  —  {status}\n{_cfg_text(cfg)}", fontsize=10)
        ep = h["epoch"]

        if not ep:  # phân kỳ ngay trong epoch đầu: không có điểm nào để vẽ
            for ax in axes:
                ax.text(0.5, 0.5, f"loss/gradient thành NaN/inf\nở epoch {s.get('diverged_epoch')}\n"
                        f"(loss bước 0 = {s['step0_loss']:.4g})", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            axes[0].set_title("loss"); axes[1].set_title("val accuracy / macro-F1"); axes[2].set_title("grad norm")
            _save(fig, path)
            return

        ax = axes[0]
        ax.plot(ep, h["train_loss"], color=PALETTE[0], lw=2, marker="o", ms=3, label="train loss (eval mode)")
        ax.plot(ep, h["val_loss"], color=PALETTE[1], lw=2, marker="o", ms=3, label="val loss")
        ax.set_title(f"Loss ({cfg['loss'].upper()}), loss bước 0 = {s['step0_loss']:.4f}")
        ax.set_ylabel("loss")

        ax = axes[1]
        ax.plot(ep, h["val_acc"], color=PALETTE[0], lw=2, marker="o", ms=3, label="val accuracy")
        ax.plot(ep, h["val_macro_f1"], color=PALETTE[1], lw=2, marker="o", ms=3, label="val macro-F1")
        ax.set_title("Val accuracy và macro-F1")
        ax.set_ylabel("điểm (0–1)")

        ax = axes[2]
        ax.plot(ep, h["grad_norm"], color=PALETTE[0], lw=2, marker="o", ms=3, label="trung bình epoch")
        ax.plot(ep, h["grad_norm_max"], color=PALETTE[1], lw=2, marker="o", ms=3, label="bước lớn nhất")
        if cfg["clip_norm"] is not None:
            ax.axhline(cfg["clip_norm"], color="#52514e", lw=1, ls=":", label=f"ngưỡng clip c={cfg['clip_norm']:g}")
        ax.set_yscale("log")
        ax.set_title("Chuẩn L2 gradient toàn cục (trước clip)")
        ax.set_ylabel("grad norm (thang log)")

        for ax in axes:
            if s["best_epoch"] is not None:
                ax.axvline(s["best_epoch"], color="#52514e", lw=1, ls="--", label="best epoch")
            ax.set_xlabel("epoch")
            ax.grid(True, **_GRID)
            ax.legend(frameon=False)
        fig.tight_layout()
        _save(fig, path)


def plot_compare(results: list[dict], metric, path: str, title: str = "", labels: list[str] | None = None) -> None:
    """Vẽ chồng một hoặc nhiều chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm:
    mỗi chỉ số một ô, mỗi thí nghiệm một đường, chú thích bằng exp_id (hoặc `labels`).

    Dùng cho ảnh figures/compare_<nhóm>.png. Tối đa 8 thí nghiệm một ảnh (mỗi đường một màu cố định).
    Lần chạy phân kỳ ngay từ epoch 1 không có điểm nào để vẽ, chỉ hiện trong chú thích.
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    assert len(results) <= len(PALETTE), f"tối đa {len(PALETTE)} đường trên một ảnh, hiện {len(results)}"
    labels = labels or [r["cfg"]["exp_id"] for r in results]
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4.2), squeeze=False)
        for ax, m in zip(axes[0], metrics):
            for r, label, color in zip(results, labels, PALETTE):
                h = r["history"]
                if r["summary"]["diverged"]:
                    label += " (phân kỳ)"
                ax.plot(h["epoch"], h[m], color=color, lw=2, marker="o", ms=3, label=label)
            if m.startswith("grad_norm"):
                ax.set_yscale("log")
            ax.set_title(METRIC_LABELS.get(m, m))
            ax.set_xlabel("epoch")
            ax.set_ylabel(METRIC_LABELS.get(m, m))
            ax.grid(True, **_GRID)
        axes[0][-1].legend(frameon=False, loc="best")
        if title:
            fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        _save(fig, path)


def plot_lr_sweep(groups: dict[str, list[dict]], path: str, metric: str = "val_macro_f1", title: str = "") -> None:
    """Độ nhạy với lr: trục x là lr (thang log), trục y là `metric` ở best epoch, mỗi bộ tối ưu một đường.

    groups: {tên bộ tối ưu: [result, ...]}. Lần chạy phân kỳ (không có metric) bị bỏ khỏi đường.
    """
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for (name, results), color in zip(groups.items(), PALETTE):
            pts = sorted((r["cfg"]["lr"], r["summary"][metric]) for r in results
                         if r["summary"][metric] is not None)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=2, marker="o", ms=6, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("lr (thang log)")
        ax.set_ylabel(f"{METRIC_LABELS.get(metric, metric)} ở best epoch")
        ax.set_title(title or "Độ nhạy với lr của từng bộ tối ưu")
        ax.grid(True, **_GRID)
        ax.legend(frameon=False)
        fig.tight_layout()
        _save(fig, path)
