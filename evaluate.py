import torch
from torch.utils.data import DataLoader


def compute_iou_batch(pred, gt):
    inter_x1 = torch.max(pred[:, 0], gt[:, 0])
    inter_y1 = torch.max(pred[:, 1], gt[:, 1])
    inter_x2 = torch.min(pred[:, 2], gt[:, 2])
    inter_y2 = torch.min(pred[:, 3], gt[:, 3])

    # Diện tích intersection (clamp 0 nếu không giao nhau)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Diện tích mỗi bbox
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])

    # Union = pred + gt - intersection
    union_area = pred_area + gt_area - inter_area

    # IoU (thêm eps tránh chia 0)
    iou = inter_area / (union_area + 1e-6)

    return iou


@torch.no_grad()
def evaluate(model, dataloader, device, desc="Evaluating"):
    model.eval()

    total_correct = 0
    total_samples = 0
    total_iou = 0.0

    for batch_idx, (imgs, ref_inds, gt_bboxes, img_shapes) in enumerate(dataloader):
        imgs = imgs.to(device)
        ref_inds = ref_inds.to(device)
        gt_bboxes = gt_bboxes.to(device)
        img_shapes = img_shapes.to(device)

        pred_bboxes = model(imgs, ref_inds, img_shapes, gt_bbox=None)
        iou = compute_iou_batch(pred_bboxes, gt_bboxes)  # [B]
        correct = (iou >= 0.5).sum().item()
        total_correct += correct
        total_samples += imgs.shape[0]
        total_iou += iou.sum().item()

    accuracy = total_correct / total_samples * 100
    avg_iou = total_iou / total_samples * 100

    print(f"[{desc}] Accuracy@IoU>=0.5: {accuracy:.2f}% | "
          f"Avg IoU: {avg_iou:.2f}% | "
          f"Samples: {total_samples}")

    return accuracy, avg_iou


# # Test
# if __name__ == "__main__":
#     print("Test evaluate functions")
#     pred = torch.tensor([[0, 0, 100, 100],
#                           [0, 0, 50, 50]], dtype=torch.float32)
#     gt = torch.tensor([[0, 0, 100, 100],
#                         [25, 25, 75, 75]], dtype=torch.float32)

#     iou = compute_iou_batch(pred, gt)
#     print(f"IoU: {iou}")
#     assert abs(iou[0].item() - 1.0) < 1e-5, "IoU case 1 sai!"
#     print("evaluate test passed")
