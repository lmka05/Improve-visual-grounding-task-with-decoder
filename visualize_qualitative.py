import os
import sys
import argparse
import random
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches, Patch
from config import Config
from models import SeqTRDet
from datasets import RefCOCODataset
from datasets.dataset import resize_image_keep_ratio
from utils import build_vocab, build_glove_matrix
from evaluate import compute_iou_batch

def draw_bbox_on_axes(ax, bbox, color, linewidth=3, label=None):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = x2 - x1
    height = y2 - y1

    rect = patches.Rectangle(
        (x1, y1), width, height,
        linewidth=linewidth,
        edgecolor=color,
        facecolor='none',
        label=label,
    )

    ax.add_patch(rect)

def load_model(checkpoint_path, config, device):
    print("Building vocabulary")
    token2idx, idx2token = build_vocab(config.ann_file)
    print(f"Vocab size: {len(token2idx)}")

    try:
        import gensim.downloader as api
        print("Loading GloVe embeddings")
        glove_model = api.load("glove-wiki-gigaword-300")
        glove_matrix = build_glove_matrix(token2idx, glove_model, config.glove_dim)
    except ImportError:
        print("gensim chưa cài. Dùng random")
        glove_matrix = torch.randn(len(token2idx), config.glove_dim) * 0.01
        glove_matrix[0] = 0

    # Khởi tạo model 
    print("Building model")
    model = SeqTRDet(config, glove_matrix).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'ema_shadow' in ckpt:
        print("Using EMA weights (better performance)")
        model.load_state_dict(ckpt['ema_shadow'], strict=True)
    else:
        print("Using standard model weights")
        model.load_state_dict(ckpt['model_state_dict'], strict=True)

    epoch = ckpt.get('epoch', '?')
    print(f"Loaded from epoch {epoch}")

    model.eval()

    return model, token2idx, idx2token


# Inference 1 samples
def inference_single(model, dataset, index, device):
    img, ref_inds, gt_bbox, img_meta = dataset[index]
    img_batch = img.unsqueeze(0).to(device)
    ref_batch = ref_inds.unsqueeze(0).to(device)
    img_shapes = torch.tensor([[
        img_meta['pad_shape'][0],   
        img_meta['pad_shape'][1],   
        img_meta['img_shape'][0],   
        img_meta['img_shape'][1],   
    ]], dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_bbox = model(img_batch, ref_batch, img_shapes, gt_bbox=None)
    pred_bbox = pred_bbox.squeeze(0).cpu()

    iou = compute_iou_batch(
        pred_bbox.unsqueeze(0), 
        gt_bbox.unsqueeze(0)    
    ).item()

    result = {
        'image_id': img_meta['image_id'],
        'expression': img_meta['expression'],
        'gt_bbox': gt_bbox.numpy(),           
        'pred_bbox': pred_bbox.numpy(),      
        'iou': iou,
        'img_shape': img_meta['img_shape'],  
        'scale': img_meta['scale_factor'][0],
    }

    return result

def load_display_image(image_id, img_dir, img_size):
    img_path = os.path.join(img_dir, "COCO_train2014_%012d.jpg" % image_id)
    pil_img = Image.open(img_path).convert('RGB')
    img_np = np.array(pil_img)
    img_resized, _ = resize_image_keep_ratio(img_np, img_size)

    return img_resized

def create_qualitative_figure(results, img_dir, img_size, output_path,
                               ncols=4, figscale=5):
    n = len(results)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * figscale, nrows * figscale),
        squeeze=False,
    )

    for i, result in enumerate(results):
        row = i // ncols  
        col = i % ncols   
        ax = axes[row][col]
        img_display = load_display_image(result['image_id'], img_dir, img_size)
        ax.imshow(img_display)
        gt_bbox = clip_bbox(result['gt_bbox'], img_display.shape)
        draw_bbox_on_axes(
            ax, gt_bbox,
            color='red',      
            linewidth=3,
            label='Ground Truth',
        )

        pred_bbox = clip_bbox(result['pred_bbox'], img_display.shape)
        draw_bbox_on_axes(
            ax, pred_bbox,
            color='blue',
            linewidth=3,
            label='Prediction',
        )

        expr = result['expression']
        if len(expr) > 50:
            expr = expr[:47] + "..."

        label_char = chr(ord('a') + i)
        title = f"({label_char}) {expr}"
        iou_text = f"IoU: {result['iou']:.2f}"
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8, wrap=True)
        ax.text(
            8, 8, iou_text,
            fontsize=11,
            fontweight='bold',
            color='white',
            verticalalignment='top',
            bbox=dict(
                boxstyle='round,pad=0.3', 
                facecolor='black',         
                alpha=0.7,                 
            ),
        )
        ax.axis('off')
    for i in range(n, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row][col].axis('off')

    legend_elements = [
        Patch(facecolor='none', edgecolor='blue', linewidth=2, label='Prediction'),
        Patch(facecolor='none', edgecolor='red', linewidth=2, label='Ground Truth'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',          
        ncol=2,                        
        fontsize=13,
        frameon=True,                
        edgecolor='gray',
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.5, -0.02),   
    )

    plt.tight_layout()
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)

    print(f"Đã lưu figure: {output_path}")

def clip_bbox(bbox, img_shape):
    h, w = img_shape[:2]
    clipped = bbox.copy()
    clipped[0] = np.clip(clipped[0], 0, w - 1)
    clipped[2] = np.clip(clipped[2], 0, w - 1)
    clipped[1] = np.clip(clipped[1], 0, h - 1)
    clipped[3] = np.clip(clipped[3], 0, h - 1)

    return clipped

# Lưu từng ảnh riêng lẻ
def save_individual_images(results, img_dir, img_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for i, result in enumerate(results):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # Load ảnh
        img_display = load_display_image(result['image_id'], img_dir, img_size)
        ax.imshow(img_display)

        # Vẽ GT bbox (đỏ)
        gt_bbox = clip_bbox(result['gt_bbox'], img_display.shape)
        draw_bbox_on_axes(ax, gt_bbox, color='red', linewidth=3)

        # Vẽ Pred bbox (xanh)
        pred_bbox = clip_bbox(result['pred_bbox'], img_display.shape)
        draw_bbox_on_axes(ax, pred_bbox, color='blue', linewidth=3)

        # Tiêu đề
        ax.set_title(
            f"{result['expression']}\nIoU: {result['iou']:.2f}",
            fontsize=11, wrap=True,
        )
        ax.axis('off')

        # Lưu file
        out_path = os.path.join(output_dir, f"qualitative_{i}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"Đã lưu {len(results)} ảnh riêng lẻ vào: {output_dir}/")


# PHẦN 8: HÀM MAIN 

def main():
    parser = argparse.ArgumentParser(
        description='SeqTR Detection — Qualitative Results Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python visualize_qualitative.py --checkpoint work_dir/best.pth
  python visualize_qualitative.py --checkpoint best.pth --split testA --num-samples 4
  python visualize_qualitative.py --checkpoint best.pth --indices 0 10 42 99
        """,
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Đường dẫn tới checkpoint (.pth). Ví dụ: work_dir/best.pth',
    )
    parser.add_argument(
        '--split', type=str, default='testA',
        choices=['val', 'testA', 'testB'],
        help='Split để lấy ảnh visualize (mặc định: testA)',
    )
    parser.add_argument(
        '--num-samples', type=int, default=4,
        help='Số ảnh cần visualize (mặc định: 4)',
    )
    parser.add_argument(
        '--indices', nargs='+', type=int, default=None,
        help='Chỉ định index cụ thể thay vì random. Ví dụ: --indices 0 10 42 99',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed cho reproducibility (mặc định: 42)',
    )
    parser.add_argument(
        '--output', type=str, default='qualitative_results.png',
        help='Đường dẫn file output (mặc định: qualitative_results.png)',
    )
    parser.add_argument(
        '--save-individual', action='store_true',
        help='Lưu thêm từng ảnh riêng lẻ (ngoài figure grid)',
    )
    parser.add_argument(
        '--ncols', type=int, default=4,
        help='Số cột trong figure grid (mặc định: 4)',
    )

    args = parser.parse_args()

    # set up
    print("Set up")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Num samples:{args.num_samples}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")

    # Load model
    model, token2idx, idx2token = load_model(args.checkpoint, config, device)

    # Tạo dataset
    dataset = RefCOCODataset(
        config.ann_file, config.img_dir, args.split,
        token2idx, config.max_token, config.img_size,
    )
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Chọn samples")

    if args.indices is not None:
        selected_indices = args.indices
        print(f"Chế độ: chỉ định index -> {selected_indices}")
    else:
        random.seed(args.seed)
        selected_indices = random.sample(
            range(len(dataset)),
            min(args.num_samples, len(dataset)),
        )
        print(f"Chế độ: random (seed={args.seed}) → {selected_indices}")

    print(f"Inference trên {len(selected_indices)} samples")

    results = []
    for i, idx in enumerate(selected_indices):
        print(f"[{i+1}/{len(selected_indices)}] Sample index={idx}...", end=" ")
        result = inference_single(model, dataset, idx, device)
        results.append(result)
        print(f"IoU={result['iou']:.3f}  expr=\"{result['expression']}\"")

    create_qualitative_figure(
        results=results,
        img_dir=config.img_dir,
        img_size=config.img_size,
        output_path=args.output,
        ncols=args.ncols,
    )

    # Lưu ảnh riêng lẻ (tùy chọn)
    if args.save_individual:
        print(f"Lưu ảnh riêng lẻ")
        individual_dir = os.path.splitext(args.output)[0] + "_individual"
        save_individual_images(results, config.img_dir, config.img_size, individual_dir)

    print("Hoàn thành")
    print(f"Figure grid: {args.output}")
    if args.save_individual:
        print(f"Ảnh riêng:{individual_dir}/")

    # In bảng tổng kết IoU
    print(f"{'Index':>6}  {'IoU':>6}  Expression")
    print(f"{'─'*6}  {'─'*6}  {'─'*40}")
    for idx, res in zip(selected_indices, results):
        expr_short = res['expression'][:40]
        print(f"{idx:>6}  {res['iou']:>6.3f}  {expr_short}")

    avg_iou = np.mean([r['iou'] for r in results])
    print(f"Average IoU: {avg_iou:.3f}")


if __name__ == "__main__":
    main()
