import os
import time
import json
import math
import yaml
import glob
import random
import shutil
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSD, SSDHead
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    HAS_TORCHMETRICS = True
except Exception:
    HAS_TORCHMETRICS = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False


# -------------------------
# Utils
# -------------------------
def now_str() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Rendimiento vs reproducibilidad
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True  # <- mejor rendimiento en GPU con shapes fijos

def read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def abs_path_from_yaml(yaml_path: str, maybe_rel: str) -> str:
    if os.path.isabs(maybe_rel):
        return maybe_rel
    base = os.path.dirname(os.path.abspath(yaml_path))
    return os.path.abspath(os.path.join(base, maybe_rel))

def list_images(img_dir: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(img_dir, e)))
    return sorted(files)

def img_to_label_path(img_path: str, labels_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(labels_dir, stem + ".txt")

def check_dataset_yolo(yaml_path: str) -> Dict[str, str]:
    cfg = read_yaml(yaml_path)

    required_keys = ["train", "val", "nc", "names"]
    for k in required_keys:
        if k not in cfg:
            raise ValueError(f"data.yaml no contiene la clave requerida: {k}")

    train_img = abs_path_from_yaml(yaml_path, cfg["train"])
    val_img   = abs_path_from_yaml(yaml_path, cfg["val"])
    test_img  = abs_path_from_yaml(yaml_path, cfg.get("test", "")) if cfg.get("test") else ""

    def infer_labels_dir(images_dir: str) -> str:
        if images_dir.endswith(os.sep + "images"):
            return images_dir[:-len("images")] + "labels"
        return images_dir.replace(os.sep + "images", os.sep + "labels")

    train_lbl = infer_labels_dir(train_img)
    val_lbl   = infer_labels_dir(val_img)
    test_lbl  = infer_labels_dir(test_img) if test_img else ""

    for d in [train_img, val_img, train_lbl, val_lbl]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"No existe el directorio: {d}")

    nc = int(cfg["nc"])
    names = cfg["names"]
    if not isinstance(names, list) or len(names) != nc:
        raise ValueError(f"Inconsistencia: nc={nc} pero names tiene len={len(names)}")

    return {
        "yaml": os.path.abspath(yaml_path),
        "train_images": train_img,
        "val_images": val_img,
        "test_images": test_img,
        "train_labels": train_lbl,
        "val_labels": val_lbl,
        "test_labels": test_lbl,
        "nc": str(nc),
        "names": json.dumps(names),
    }


# -------------------------
# Dataset YOLO -> Torchvision detection
# -------------------------
class YoloDetectionDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, nc: int, img_size: Optional[int] = None):
        self.images = list_images(images_dir)
        self.labels_dir = labels_dir
        self.nc = nc
        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def _read_labels(self, label_path: str, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        boxes, labels = [], []
        if not os.path.exists(label_path):
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, bw, bh = parts
                cls = int(float(cls))
                if cls < 0 or cls >= self.nc:
                    continue

                xc = float(xc) * w
                yc = float(yc) * h
                bw = float(bw) * w
                bh = float(bh) * h

                x1 = max(0.0, min(xc - bw / 2.0, w - 1))
                y1 = max(0.0, min(yc - bh / 2.0, h - 1))
                x2 = max(0.0, min(xc + bw / 2.0, w - 1))
                y2 = max(0.0, min(yc + bh / 2.0, h - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels.append(cls + 1)  # 0 reservado para background

        if len(boxes) == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)

        return np.array(boxes, np.float32), np.array(labels, np.int64)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        label_path = img_to_label_path(img_path, self.labels_dir)
        boxes, labels = self._read_labels(label_path, w, h)

        if self.img_size is not None:
            new_w = new_h = int(self.img_size)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            if boxes.shape[0] > 0:
                boxes[:, [0, 2]] *= (new_w / w)
                boxes[:, [1, 3]] *= (new_h / h)

        img_t = F.to_tensor(img)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        if boxes.shape[0] > 0:
            area = (target["boxes"][:, 2] - target["boxes"][:, 0]) * (target["boxes"][:, 3] - target["boxes"][:, 1])
        else:
            area = torch.zeros((0,), dtype=torch.float32)
        target["area"] = area
        return img_t, target

def collate_fn(batch):
    return tuple(zip(*batch))


# -------------------------
# Models
# -------------------------
def build_fasterrcnn_resnet50(num_classes: int, pretrained: bool = True):
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def build_ssd_mobilenetv2(num_classes: int, pretrained_backbone: bool = True, img_size: int = 320):
    # En lugar de construir el extractor de características a mano, 
    # usamos SSDLite (basado en MobileNetV3), que ya maneja las múltiples escalas necesarias.
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone="DEFAULT" if pretrained_backbone else None,
        num_classes=num_classes
    )
    return model

# -------------------------
# Train (Torchvision detection)
# -------------------------
@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_iters: int = 800
    num_workers: int = 8
    img_size: int = 640
    amp: bool = True
    grad_clip: float = 1.0
    early_stop_patience: int = 10
    save_dir: str = "runs_torch"
    seed: int = 42

def build_optimizer(model, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

def warmup_lr_lambda(current_step: int, warmup_steps: int):
    if current_step >= warmup_steps:
        return 1.0
    return float(current_step) / float(max(1, warmup_steps))

def get_autocast_dtype(device: torch.device):
    # L4 (Ada) soporta BF16; suele ser más estable que FP16.
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

@torch.no_grad()
def evaluate_map(model, loader, device) -> Dict[str, float]:
    model.eval()
    if not HAS_TORCHMETRICS:
        return {"map": float("nan"), "map50": float("nan"), "map75": float("nan")}

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    for images, targets in tqdm(loader, desc="Evaluating", leave=False):
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)

        preds, gts = [], []
        for out, tgt in zip(outputs, targets):
            preds.append({
                "boxes": out["boxes"].detach().cpu(),
                "scores": out["scores"].detach().cpu(),
                "labels": out["labels"].detach().cpu(),
            })
            gts.append({
                "boxes": tgt["boxes"].detach().cpu(),
                "labels": tgt["labels"].detach().cpu(),
            })
        metric.update(preds, gts)

    res = metric.compute()
    return {
        "map": float(res["map"].item()),
        "map50": float(res["map_50"].item()),
        "map75": float(res["map_75"].item()),
    }

def train_torch_detection(model_name: str, model, train_loader, val_loader, device, cfg: TrainConfig):
    ensure_dir(cfg.save_dir)
    run_name = f"{model_name}_{now_str()}"
    out_dir = os.path.join(cfg.save_dir, run_name)
    ensure_dir(out_dir)

    log_path = os.path.join(out_dir, "train_log.jsonl")

    model.to(device)
    optimizer = build_optimizer(model, cfg.lr, cfg.weight_decay)

    total_steps = cfg.epochs * len(train_loader)

    def lr_factor(step):
        warm = warmup_lr_lambda(step, cfg.warmup_iters)
        if step < cfg.warmup_iters:
            return warm
        progress = (step - cfg.warmup_iters) / float(max(1, total_steps - cfg.warmup_iters))
        cosine = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))
        return cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    autocast_dtype = get_autocast_dtype(device)

    best_map50 = -1.0
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")
    patience = 0

    # (Opcional) torch.compile (acelera en PyTorch 2.x; si te da problemas, ponlo en False)
    use_compile = (device.type == "cuda")
    if use_compile:
        try:
            model = torch.compile(model)  # PyTorch 2.x
        except Exception as e:
            print(f"[WARN] torch.compile no disponible/compatible: {e}")

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        epoch_loss = 0.0

        for images, targets in pbar:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda"), dtype=autocast_dtype):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += float(loss.item())
            global_step += 1
            pbar.set_postfix(loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"]))

        epoch_loss /= max(1, len(train_loader))

        metrics = evaluate_map(model, val_loader, device)
        m50 = metrics.get("map50", float("nan"))

        record = {"epoch": epoch, "train_loss": epoch_loss, **metrics, "lr": float(optimizer.param_groups[0]["lr"])}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "cfg": cfg.__dict__, "metrics": metrics}, last_path)

        improved = (not math.isnan(m50)) and (m50 > best_map50)
        if improved:
            best_map50 = m50
            patience = 0
            shutil.copy(last_path, best_path)
        else:
            patience += 1

        print(f"[{model_name}] epoch={epoch} loss={epoch_loss:.4f} mAP50={m50:.4f} best={best_map50:.4f} patience={patience}/{cfg.early_stop_patience}")

        if patience >= cfg.early_stop_patience:
            print(f"[EARLY STOP] No mejora en {cfg.early_stop_patience} epochs.")
            break

    print(f"✅ Best checkpoint: {best_path}")
    return out_dir, best_path


# -------------------------
# YOLOv8 training
# -------------------------
def train_yolov8(yaml_path: str, variant: str, out_dir: str, epochs: int, imgsz: int, batch: int, device: str, seed: int):
    if not HAS_ULTRALYTICS:
        raise RuntimeError("Ultralytics no está instalado. pip install ultralytics")

    ensure_dir(out_dir)
    model = YOLO(f"yolov8{variant}.pt")

    # Para 32GB RAM, evita cache=True (RAM). Mejor cache en disco.
    # workers: con 8 CPUs, 8 workers suele ir bien.
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        seed=seed,
        patience=25,
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        weight_decay=0.0005,
        cos_lr=True,
        amp=True,
        cache="disk",
        workers=8,
        close_mosaic=10,
        project=out_dir,
        name=f"yolov8{variant}_{now_str()}",
        exist_ok=False,
        verbose=True
    )

    model.val(data=yaml_path, imgsz=imgsz, device=device)

    # Export ONNX opcional
    try:
        model.export(format="onnx", opset=12, dynamic=True)
    except Exception as e:
        print(f"[WARN] Export ONNX falló: {e}")

    return results


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data.yaml")
    parser.add_argument("--model", type=str, required=True,
                        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "ssd_mnv2", "fasterrcnn_r50"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=0, help="0 = auto recomendado por L4")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO imgsz")
    parser.add_argument("--img_size", type=int, default=640, help="Torchvision resize")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0, help="0 = auto por modelo")
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default="runs")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true", help="más reproducible, un poco más lento")
    args = parser.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)

    paths = check_dataset_yolo(args.data)
    nc = int(paths["nc"])
    names = json.loads(paths["names"])
    print("📦 Dataset OK:", {"nc": nc, "names": names})

    # ---------------- Auto defaults para NVIDIA L4 24GB ----------------
    # (Puedes override con --batch o --lr)
    AUTO = {
        "yolov8n": {"batch": 32, "imgsz": 640, "epochs": args.epochs},
        "yolov8s": {"batch": 24, "imgsz": 640, "epochs": args.epochs},
        "yolov8m": {"batch": 16, "imgsz": 640, "epochs": args.epochs},
        "yolov8l": {"batch": 8,  "imgsz": 640, "epochs": args.epochs},
        "ssd_mnv2": {"batch": 32, "img_size": 320, "lr": 1e-3, "epochs": max(80, args.epochs)},
        "fasterrcnn_r50": {"batch": 4, "img_size": 640, "lr": 5e-4, "epochs": min(60, args.epochs)},
    }

    if args.model.startswith("yolov8"):
        variant = args.model.replace("yolov8", "")
        batch = args.batch if args.batch > 0 else AUTO[args.model]["batch"]
        imgsz = args.imgsz if args.imgsz else AUTO[args.model]["imgsz"]
        epochs = args.epochs

        out_dir = os.path.join(args.out, "yolo")
        train_yolov8(paths["yaml"], variant, out_dir, epochs, imgsz, batch, args.device, args.seed)
        return

    # Torchvision device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 PyTorch device: {device}, bf16_supported={torch.cuda.is_available() and torch.cuda.is_bf16_supported()}")

    # Ajustes auto (si no te pasan valores)
    batch = args.batch if args.batch > 0 else AUTO[args.model]["batch"]
    img_size = args.img_size if args.img_size else AUTO[args.model]["img_size"]
    lr = args.lr if args.lr > 0 else AUTO[args.model]["lr"]

    train_ds = YoloDetectionDataset(paths["train_images"], paths["train_labels"], nc=nc, img_size=img_size)
    val_ds   = YoloDetectionDataset(paths["val_images"],   paths["val_labels"],   nc=nc, img_size=img_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, batch // 2), shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
        collate_fn=collate_fn
    )

    num_classes = nc + 1

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=batch,
        lr=lr,
        weight_decay=args.wd,
        num_workers=args.num_workers,
        img_size=img_size,
        amp=args.amp or True,  # recomendado
        save_dir=os.path.join(args.out, "torch"),
        seed=args.seed
    )

    if args.model == "fasterrcnn_r50":
        model = build_fasterrcnn_resnet50(num_classes=num_classes, pretrained=True)
        train_torch_detection("fasterrcnn_r50", model, train_loader, val_loader, device, cfg)
        return

    if args.model == "ssd_mnv2":
        model = build_ssd_mobilenetv2(num_classes=num_classes, pretrained_backbone=True, img_size=img_size)
        train_torch_detection("ssd_mnv2", model, train_loader, val_loader, device, cfg)
        return

    raise ValueError("Modelo no soportado.")

if __name__ == "__main__":
    main()