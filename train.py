import os
import sys
import copy
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from config import Config
from utils.vocab import build_vocab, build_glove_matrix
from datasets.dataset import RefCOCODataset, build_dataloader
from models.model import CoordinateSequenceDecoder
from evaluate import evaluate

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.step_count = 0
        self.shadow = {name: param.clone().detach()
                       for name, param in model.state_dict().items()}

    def update(self, model):
        decay = min(self.decay, (self.step_count + 1) / (self.step_count + 10))
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in self.shadow:
                    if not param.is_floating_point():
                        self.shadow[name].copy_(param)
                        continue
                    self.shadow[name].mul_(decay).add_(param, alpha=1 - decay)
        self.step_count += 1

    def apply(self, model):
        self.backup = {name: param.clone()
                       for name, param in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model):
        model.load_state_dict(self.backup, strict=True)
        del self.backup 
        self.backup = None


def build_scheduler(optimizer, config):
    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return (epoch + 1) / (config.warmup_epochs + 1)
        elif epoch < config.decay_epoch:
            return 1.0
        else:
            return config.decay_ratio

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(model, ema, optimizer, scheduler, epoch, accuracy, best_accuracy, config):
    os.makedirs(config.work_dir, exist_ok=True)
    raw_model = model.module if hasattr(model, 'module') else model

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy': accuracy,
        'best_accuracy': best_accuracy,
    }
    if ema is not None:
        checkpoint['ema_shadow'] = ema.shadow

    latest_path = os.path.join(config.work_dir, 'latest.pth')
    torch.save(checkpoint, latest_path)

    if accuracy >= best_accuracy:
        best_path = os.path.join(config.work_dir, 'best.pth')
        torch.save(checkpoint, best_path)
        print(f"  ★ New best model saved! Acc: {accuracy:.2f}%")

def train_one_epoch(model, dataloader, optimizer, device, epoch, config, ema=None):
    model.train()
    total_loss = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, (imgs, ref_inds, gt_bboxes, img_shapes) in enumerate(dataloader):
        imgs = imgs.to(device)
        ref_inds = ref_inds.to(device)
        gt_bboxes = gt_bboxes.to(device)
        img_shapes = img_shapes.to(device)

        loss = model(imgs, ref_inds, img_shapes, gt_bbox=gt_bboxes)

        if loss.dim() > 0:
            loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()

        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        if ema is not None:
            # [CŨ] ema.update(model)
            raw_model = model.module if hasattr(model, 'module') else model
            ema.update(raw_model)

        # Tracking
        total_loss += loss.item()
        num_batches += 1

        # Log
        if (batch_idx + 1) % config.log_interval == 0:
            avg = total_loss / num_batches
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {avg:.4f} | LR: {lr:.6f} | Time: {elapsed:.1f}s")

    avg_loss = total_loss / num_batches
    return avg_loss


# Main

def main():
    config = Config

    # Seed
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build vocabulary
    print("Building vocabulary")
    token2idx, idx2token = build_vocab(config.ann_file)
    print(f"Vocabulary size: {len(token2idx)}")

    # Load GloVe embeddings
    print("Loading GloVe embeddings")
    try:
        import gensim.downloader as api
        glove_model = api.load("glove-wiki-gigaword-300")
        glove_matrix = build_glove_matrix(token2idx, glove_model, config.glove_dim)
        del glove_model
        import gc; gc.collect()
    except ImportError:
        print("gensim chưa cài. Dùng random embeddings")
        print("Cài gensim: pip install gensim")
        glove_matrix = torch.randn(len(token2idx), config.glove_dim) * 0.01
        glove_matrix[0] = 0

    # Create datasets
    print("STEP 3: Creating datasets")
    train_dataset = RefCOCODataset(
        config.ann_file, config.img_dir, 'train',
        token2idx, config.max_token, config.img_size
    )
    val_dataset = RefCOCODataset(
        config.ann_file, config.img_dir, 'val',
        token2idx, config.max_token, config.img_size
    )

    train_loader = build_dataloader(
        train_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers
    )
    val_loader = build_dataloader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers
    )

    # Build model
    print("Building model")
    model = CoordinateSequenceDecoder(config, glove_matrix).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {train_params:,}")

    # Optimizer + Scheduler
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0,
        amsgrad=True,
    )
    scheduler = build_scheduler(optimizer, config)

    ema = EMA(model, decay=config.ema_decay) if config.ema else None

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs with DataParallel!")
        model = nn.DataParallel(model)
    else:
        print(f"Using 1 GPU")

    # Resume from checkpoint
    start_epoch = 0
    best_accuracy = 0.0
    latest_ckpt = os.path.join(config.work_dir, 'latest.pth')
    if os.path.exists(latest_ckpt):
        print(f"\nResuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        # Load vào model gốc (bên trong DataParallel nếu có)
        raw_model = model.module if hasattr(model, 'module') else model
        raw_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_accuracy = ckpt.get('best_accuracy', 0.0)
        if ema is not None and 'ema_shadow' in ckpt:
            ema.shadow = ckpt['ema_shadow']
        print(f"Resumed from epoch {start_epoch}, best acc: {best_accuracy:.2f}%")

    # Training loop
    print("Start training")

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()

        # Train
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, config, ema
        )

        # Evaluate
        print(f"Evaluating epoch {epoch+1}")
        if ema is not None:
            raw_model = model.module if hasattr(model, 'module') else model
            ema.apply(raw_model)
            val_acc, val_iou = evaluate(model, val_loader, device, desc="val (EMA)")
            ema.restore(raw_model)
        else:
            val_acc, val_iou = evaluate(model, val_loader, device, desc="val")

        # Save checkpoint
        save_checkpoint(
            model, ema, optimizer, scheduler,
            epoch, val_acc, best_accuracy, config
        )
        best_accuracy = max(best_accuracy, val_acc)

        # Step scheduler
        scheduler.step()
    
        gc.collect()
        torch.cuda.empty_cache()

        # Epoch summary
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        mem_alloc = torch.cuda.memory_allocated() / 1024**2
        mem_reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.epochs} Summary:")
        print(f"  Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Best: {best_accuracy:.2f}% | LR: {lr:.6f} | Time: {epoch_time:.0f}s")
        print(f"  GPU Memory: {mem_alloc:.0f}MB allocated / {mem_reserved:.0f}MB reserved")
        print(f"{'='*60}\n")

    print(f"Training finished. Best accuracy: {best_accuracy:.2f}%")
    print(f"Checkpoints saved at: {config.work_dir}")


if __name__ == "__main__":
    main()
