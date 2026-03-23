from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import dill
import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


ENTRY_SUFFIX = ".pkl"
TARGET_SR = 16000
PATCH_FRAMES = 96
N_MELS = 64
EPS = 1e-6


def seed_everything(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


@dataclass
class EntryRecord:
	filename: str
	label: str
	start: int
	end: int
	samplerate: int


def load_metadata_entries(entries_dir: Path, max_entries: int | None = None) -> List[EntryRecord]:
	files = sorted(entries_dir.glob(f"*{ENTRY_SUFFIX}"))
	if not files:
		raise FileNotFoundError(f"No {ENTRY_SUFFIX} files found in: {entries_dir}")

	records: List[EntryRecord] = []
	skipped = 0

	for file_path in files:
		try:
			with file_path.open("rb") as f:
				raw = dill.load(f)
		except Exception:
			skipped += 1
			continue

		try:
			start = int(raw["start"])
			end = int(raw["end"])
			samplerate = int(raw["samplerate"])
			filename = str(raw["filename"])
			label = str(raw["label"])
		except Exception:
			skipped += 1
			continue

		if end <= start or samplerate <= 0:
			skipped += 1
			continue

		# Use only filename/label/segment metadata and ignore precomputed features.
		records.append(
			EntryRecord(
				filename=filename,
				label=label,
				start=start,
				end=end,
				samplerate=samplerate,
			)
		)

		if max_entries is not None and len(records) >= max_entries:
			break

	if not records:
		raise RuntimeError(
			"No valid metadata records were loaded from pre_processed_entries."
		)

	print(f"Loaded {len(records)} metadata records (skipped {skipped}).")
	return records


def build_label_mapping(records: List[EntryRecord]) -> Dict[str, int]:
	labels = sorted({record.label for record in records})
	return {label: idx for idx, label in enumerate(labels)}


class YAMNetEntriesDataset(Dataset):
	def __init__(
		self,
		entries: List[EntryRecord],
		label_to_idx: Dict[str, int],
		audio_root: Path,
		sample_rate: int = TARGET_SR,
		patch_frames: int = PATCH_FRAMES,
		n_mels: int = N_MELS,
		random_crop: bool = False,
	) -> None:
		self.entries = entries
		self.label_to_idx = label_to_idx
		self.audio_root = audio_root
		self.sample_rate = sample_rate
		self.patch_frames = patch_frames
		self.n_mels = n_mels
		self.random_crop = random_crop

	def __len__(self) -> int:
		return len(self.entries)

	def _audio_path(self, filename: str) -> Path:
		audio_path = self.audio_root / filename
		if audio_path.exists():
			return audio_path

		fallback = self.audio_root.parent / "data" / "audio_data" / filename
		if fallback.exists():
			return fallback

		return audio_path

	def _load_signal(self, entry: EntryRecord) -> np.ndarray:
		audio_path = self._audio_path(entry.filename)
		offset_seconds = entry.start / entry.samplerate
		duration_seconds = (entry.end - entry.start) / entry.samplerate

		try:
			signal, _ = librosa.load(
				str(audio_path),
				sr=self.sample_rate,
				mono=True,
				offset=offset_seconds,
				duration=duration_seconds,
			)
		except Exception:
			signal = np.zeros(self.sample_rate, dtype=np.float32)

		if signal.size == 0:
			signal = np.zeros(self.sample_rate, dtype=np.float32)

		return np.asarray(signal, dtype=np.float32)

	def _signal_to_log_mel(self, signal: np.ndarray) -> np.ndarray:
		mel = librosa.feature.melspectrogram(
			y=signal,
			sr=self.sample_rate,
			n_fft=400,
			hop_length=160,
			win_length=400,
			n_mels=self.n_mels,
			fmin=125,
			fmax=7500,
			power=2.0,
		)
		log_mel = np.log(mel + EPS).T  # [frames, mel_bins]

		frames = log_mel.shape[0]
		if frames < self.patch_frames:
			pad = self.patch_frames - frames
			log_mel = np.pad(log_mel, ((0, pad), (0, 0)), mode="constant")
		elif frames > self.patch_frames:
			if self.random_crop:
				start = random.randint(0, frames - self.patch_frames)
			else:
				start = (frames - self.patch_frames) // 2
			log_mel = log_mel[start : start + self.patch_frames]

		return log_mel.astype(np.float32)

	def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
		entry = self.entries[idx]
		signal = self._load_signal(entry)
		patch = self._signal_to_log_mel(signal)

		x = torch.from_numpy(patch).unsqueeze(0)  # [1, T, M]
		y = torch.tensor(self.label_to_idx[entry.label], dtype=torch.long)
		return x, y


class DepthwiseSeparableConv2d(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
		super().__init__()
		self.depthwise = nn.Conv2d(
			in_channels,
			in_channels,
			kernel_size=3,
			stride=stride,
			padding=1,
			groups=in_channels,
			bias=False,
		)
		self.bn_dw = nn.BatchNorm2d(in_channels)
		self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
		self.bn_pw = nn.BatchNorm2d(out_channels)
		self.act = nn.ReLU(inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.depthwise(x)
		x = self.bn_dw(x)
		x = self.act(x)
		x = self.pointwise(x)
		x = self.bn_pw(x)
		x = self.act(x)
		return x


class YAMNetBackbone(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.stem = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
		)

		cfg = [
			(64, 1),
			(128, 2),
			(128, 1),
			(256, 2),
			(256, 1),
			(512, 2),
			(512, 1),
			(512, 1),
			(512, 1),
			(512, 1),
			(512, 1),
			(1024, 2),
			(1024, 1),
		]

		blocks: List[nn.Module] = []
		in_channels = 32
		for out_channels, stride in cfg:
			blocks.append(
				DepthwiseSeparableConv2d(
					in_channels=in_channels,
					out_channels=out_channels,
					stride=stride,
				)
			)
			in_channels = out_channels

		self.blocks = nn.ModuleList(blocks)
		self.pool = nn.AdaptiveAvgPool2d((1, 1))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.stem(x)
		for block in self.blocks:
			x = block(x)
		x = self.pool(x)
		return torch.flatten(x, 1)


class FineTunableYAMNet(nn.Module):
	def __init__(self, num_classes: int, dropout: float = 0.2) -> None:
		super().__init__()
		self.backbone = YAMNetBackbone()
		self.dropout = nn.Dropout(dropout)
		self.classifier = nn.Linear(1024, num_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.backbone(x)
		x = self.dropout(x)
		return self.classifier(x)

	def freeze_backbone(self) -> None:
		for p in self.backbone.parameters():
			p.requires_grad = False

	def unfreeze_all(self) -> None:
		for p in self.backbone.parameters():
			p.requires_grad = True

	def unfreeze_last_n_blocks(self, n_blocks: int) -> None:
		self.freeze_backbone()
		modules: List[nn.Module] = [self.backbone.stem] + list(self.backbone.blocks)
		n_blocks = max(0, min(n_blocks, len(modules)))
		for module in modules[-n_blocks:]:
			for p in module.parameters():
				p.requires_grad = True

	def parameter_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
		backbone_params = [
			p for p in self.backbone.parameters() if p.requires_grad
		]
		head_params = [p for p in self.classifier.parameters() if p.requires_grad]

		groups: list[dict] = []
		if backbone_params:
			groups.append({"params": backbone_params, "lr": backbone_lr})
		if head_params:
			groups.append({"params": head_params, "lr": head_lr})
		return groups


def make_class_weights(entries: List[EntryRecord], label_to_idx: Dict[str, int]) -> torch.Tensor:
	counts = np.zeros(len(label_to_idx), dtype=np.float64)
	for entry in entries:
		counts[label_to_idx[entry.label]] += 1

	counts = np.maximum(counts, 1.0)
	inv = counts.sum() / (len(counts) * counts)
	return torch.tensor(inv, dtype=torch.float32)


def train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	criterion: nn.Module,
	device: torch.device,
) -> float:
	model.train()
	running_loss = 0.0
	batch_count = 0

	for x, y in loader:
		x = x.to(device)
		y = y.to(device)

		optimizer.zero_grad(set_to_none=True)
		logits = model(x)
		loss = criterion(logits, y)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		batch_count += 1

	return running_loss / max(batch_count, 1)


@torch.no_grad()
def evaluate(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
) -> tuple[float, float, float]:
	model.eval()
	running_loss = 0.0
	batch_count = 0
	all_true: List[int] = []
	all_pred: List[int] = []

	for x, y in loader:
		x = x.to(device)
		y = y.to(device)

		logits = model(x)
		loss = criterion(logits, y)

		running_loss += loss.item()
		batch_count += 1

		pred = torch.argmax(logits, dim=1)
		all_true.extend(y.cpu().tolist())
		all_pred.extend(pred.cpu().tolist())

	val_loss = float(running_loss / max(batch_count, 1))
	val_acc = float(accuracy_score(all_true, all_pred)) if all_true else 0.0
	val_f1 = (
		float(f1_score(all_true, all_pred, average="weighted", zero_division=0))
		if all_true
		else 0.0
	)
	return val_loss, val_acc, val_f1


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Fine-tune a PyTorch YAMNet-style model from pre_processed_entries metadata."
	)
	parser.add_argument("--entries-dir", type=str, default="./data/pre_processed_entries")
	parser.add_argument("--audio-dir", type=str, default="./dataset")
	parser.add_argument("--save-path", type=str, default="./yamnet_finetuned_best.pt")
	parser.add_argument("--max-entries", type=int, default=None)

	parser.add_argument("--val-ratio", type=float, default=0.2)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--seed", type=int, default=42)

	parser.add_argument("--epochs-head", type=int, default=5)
	parser.add_argument("--epochs-finetune", type=int, default=20)
	parser.add_argument("--unfreeze-blocks", type=int, default=4)

	parser.add_argument("--lr-head", type=float, default=1e-3)
	parser.add_argument("--lr-backbone", type=float, default=1e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--dropout", type=float, default=0.2)

	parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	seed_everything(args.seed)

	entries_dir = Path(args.entries_dir)
	audio_dir = Path(args.audio_dir)

	records = load_metadata_entries(entries_dir=entries_dir, max_entries=args.max_entries)
	label_to_idx = build_label_mapping(records)
	idx_to_label = {idx: label for label, idx in label_to_idx.items()}

	labels_as_idx = [label_to_idx[r.label] for r in records]
	indices = np.arange(len(records))

	train_idx, val_idx = train_test_split(
		indices,
		test_size=args.val_ratio,
		random_state=args.seed,
		stratify=labels_as_idx,
	)

	train_entries = [records[i] for i in train_idx]
	val_entries = [records[i] for i in val_idx]

	print(
		f"Split: train={len(train_entries)} | val={len(val_entries)} | classes={len(label_to_idx)}"
	)
	print(f"Labels: {sorted(label_to_idx.keys())}")

	train_dataset = YAMNetEntriesDataset(
		entries=train_entries,
		label_to_idx=label_to_idx,
		audio_root=audio_dir,
		random_crop=True,
	)
	val_dataset = YAMNetEntriesDataset(
		entries=val_entries,
		label_to_idx=label_to_idx,
		audio_root=audio_dir,
		random_crop=False,
	)

	if args.device == "auto":
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	else:
		device = torch.device(args.device)

	print(f"Using device: {device}")

	pin_memory = device.type == "cuda"
	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=pin_memory,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=pin_memory,
	)

	model = FineTunableYAMNet(num_classes=len(label_to_idx), dropout=args.dropout).to(device)

	class_weights = make_class_weights(train_entries, label_to_idx).to(device)
	criterion = nn.CrossEntropyLoss(weight=class_weights)

	best = {
		"val_loss": float("inf"),
		"val_acc": 0.0,
		"val_f1": 0.0,
		"epoch": 0,
		"state": None,
	}
	global_epoch = 0

	# Phase 1: train classifier head only.
	model.freeze_backbone()
	optimizer = torch.optim.AdamW(
		model.parameter_groups(backbone_lr=args.lr_backbone, head_lr=args.lr_head),
		weight_decay=args.weight_decay,
	)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode="min",
		factor=0.5,
		patience=2,
	)

	print("\nPhase 1/2: head-only fine-tuning")
	for phase_epoch in range(1, args.epochs_head + 1):
		global_epoch += 1
		train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
		val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
		scheduler.step(val_loss)

		if val_loss < best["val_loss"]:
			best.update(
				{
					"val_loss": val_loss,
					"val_acc": val_acc,
					"val_f1": val_f1,
					"epoch": global_epoch,
					"state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
				}
			)

		print(
			f"[Head {phase_epoch:02d}/{args.epochs_head}] "
			f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
			f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f}"
		)

	# Phase 2: unfreeze last N backbone blocks + head.
	model.unfreeze_last_n_blocks(args.unfreeze_blocks)
	optimizer = torch.optim.AdamW(
		model.parameter_groups(backbone_lr=args.lr_backbone, head_lr=args.lr_head),
		weight_decay=args.weight_decay,
	)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode="min",
		factor=0.5,
		patience=3,
	)

	print(f"\nPhase 2/2: unfreeze last {args.unfreeze_blocks} backbone blocks")
	for phase_epoch in range(1, args.epochs_finetune + 1):
		global_epoch += 1
		train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
		val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
		scheduler.step(val_loss)

		if val_loss < best["val_loss"]:
			best.update(
				{
					"val_loss": val_loss,
					"val_acc": val_acc,
					"val_f1": val_f1,
					"epoch": global_epoch,
					"state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
				}
			)

		print(
			f"[Tune {phase_epoch:02d}/{args.epochs_finetune}] "
			f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
			f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f}"
		)

	if best["state"] is None:
		raise RuntimeError("Training finished without a valid checkpoint.")

	model.load_state_dict(best["state"])
	final_val_loss, final_val_acc, final_val_f1 = evaluate(model, val_loader, criterion, device)

	checkpoint = {
		"model_state_dict": model.state_dict(),
		"label_to_idx": label_to_idx,
		"idx_to_label": idx_to_label,
		"best_epoch": best["epoch"],
		"best_val_loss": best["val_loss"],
		"best_val_acc": best["val_acc"],
		"best_val_f1": best["val_f1"],
		"final_val_loss": final_val_loss,
		"final_val_acc": final_val_acc,
		"final_val_f1": final_val_f1,
		"config": vars(args),
	}

	save_path = Path(args.save_path)
	save_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(checkpoint, save_path)

	labels_path = save_path.with_suffix(".labels.json")
	with labels_path.open("w", encoding="utf-8") as f:
		json.dump(idx_to_label, f, indent=2)

	print("\nTraining finished.")
	print(f"Best checkpoint epoch: {best['epoch']}")
	print(
		f"Best val metrics: loss={best['val_loss']:.4f} | "
		f"acc={best['val_acc']:.4f} | f1={best['val_f1']:.4f}"
	)
	print(
		f"Final val metrics: loss={final_val_loss:.4f} | "
		f"acc={final_val_acc:.4f} | f1={final_val_f1:.4f}"
	)
	print(f"Saved model: {save_path}")
	print(f"Saved label map: {labels_path}")


if __name__ == "__main__":
	main()
