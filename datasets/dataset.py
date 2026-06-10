import os
import re
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from utils.vocab import clean_expression, tokenize_expression

# Resize ảnh
def resize_image_keep_ratio(img, max_size):
    h,w = img.shape[:2]
    scale = max_size / max(h,w)
    new_h , new_w = int(h*scale), int(w*scale)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((new_w, new_h))
    resized_img = np.array(pil_img)

    return resized_img, scale

# Pad ảnh
def pad_image_to_square(img, target_size, pad_value =0):
        h,w = img.shape[:2]
        padded = np.full((target_size,target_size,3), pad_value, dtype = img.dtype)
        padded[:h, :w, :] = img
        
        return padded

def normalize_image(img):
    return img.astype(np.float32)/255.0

def image_to_tensor(img):
    img = np.transpose(img, (2, 0, 1))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)

    return img

# Resize bounding box
def transform_bbox(bbox_xywh, scale, img_shape_after_resize):
    x, y, w, h = bbox_xywh

    # Scale tọa độ theo tỉ lệ resize
    x1 = x * scale
    y1 = y * scale
    x2 = (x + w) * scale
    y2 = (y + h) * scale

    # Clip tọa độ để không vượt quá biên ảnh
    new_h, new_w = img_shape_after_resize
    x1 = np.clip(x1, 0, new_w - 1)
    y1 = np.clip(y1, 0, new_h - 1)
    x2 = np.clip(x2, 0, new_w - 1)
    y2 = np.clip(y2, 0, new_h - 1)

    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

class RefCOCODataset(Dataset):

    def __init__(self, ann_file, img_dir, split, token2idx, max_token=15, img_size=640):
        super().__init__()
        self.img_dir = img_dir
        self.split = split
        self.token2idx = token2idx
        self.max_token = max_token
        self.img_size = img_size
        anns_all = json.load(open(ann_file, 'r'))
        self.anns = anns_all[split]
        print(f"[{split}] Loaded {len(self.anns)} samples")

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, index):
        # Lấy 1 sample tại vị trí index.
        ann = self.anns[index]
        img_path = os.path.join(
            self.img_dir,
            "COCO_train2014_%012d.jpg" % ann['image_id']
        )

        pil_img = Image.open(img_path).convert('RGB')
        img = np.array(pil_img)
        ori_h, ori_w = img.shape[:2]
        img, scale = resize_image_keep_ratio(img, self.img_size)
        resized_h, resized_w = img.shape[:2]
        img = pad_image_to_square(img, self.img_size)
        img = normalize_image(img)      # [H, W, 3] float32 [0, 1]
        img = image_to_tensor(img)      # [3, H, W] tensor

        expressions = ann['expressions']
        if self.split == 'train':
            # Training: random chọn 1 câu (data augmentation cho text)
            expression = random.choice(expressions)
        else:
            # Val/Test: luôn chọn câu đầu tiên (để kết quả consistent)
            expression = expressions[0]

        ref_inds = tokenize_expression(expression, self.token2idx, self.max_token)

        gt_bbox = transform_bbox(
            ann['bbox'],
            scale=scale,
            img_shape_after_resize=(resized_h, resized_w)
        )

        img_meta = {
            'image_id': ann['image_id'],
            'expression': expression,
            'ori_shape': (ori_h, ori_w, 3),         # Kích thước ảnh gốc
            'img_shape': (resized_h, resized_w, 3),  # Kích thước sau resize
            'pad_shape': (self.img_size, self.img_size, 3),  # Kích thước sau pad
            'scale_factor': np.array([scale, scale, scale, scale], dtype=np.float32),
        }

        return img, ref_inds, gt_bbox, img_meta
    

def collate_fn(batch):
    imgs, ref_inds, gt_bboxes, img_metas = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    ref_inds = torch.stack(ref_inds, dim=0)
    gt_bboxes = torch.stack(gt_bboxes, dim=0)

    img_shapes = torch.tensor([
        [m['pad_shape'][0], m['pad_shape'][1],
         m['img_shape'][0], m['img_shape'][1]]
        for m in img_metas
    ], dtype=torch.float32)

    return imgs, ref_inds, gt_bboxes, img_shapes

def build_dataloader(dataset, batch_size, shuffle=True, num_workers=2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,            
        drop_last=(shuffle == True), 
    )

# if __name__ == "__main__":
#     from config import Config

#     print("Test Dataset")

#     # Build vocabulary
#     print("Building vocabulary")
#     token2idx, idx2token = build_vocab(Config.ann_file)
#     print(f"Vocabulary size: {len(token2idx)}")
#     print(f"Sample words: {list(token2idx.items())[:10]}")

#     # Tạo dataset
#     print("Creating dataset")
#     train_dataset = RefCOCODataset(
#         ann_file=Config.ann_file,
#         img_dir=Config.img_dir,
#         split='train',
#         token2idx=token2idx,
#         max_token=Config.max_token,
#         img_size=Config.img_size,
#     )

#     # Lấy 1 sample
#     print("Getting 1 sample")
#     img, ref_inds, gt_bbox, img_meta = train_dataset[0]
#     print(f"Image shape: {img.shape}")            # [3, 640, 640]
#     print(f"Image dtype: {img.dtype}")             # torch.float32
#     print(f"Image range: [{img.min():.2f}, {img.max():.2f}]")  # [0, 1]
#     print(f"Ref indices: {ref_inds}")              # [idx1, idx2, ..., 0, 0, 0]
#     print(f"Ref words: {[idx2token.get(i.item(), '?') for i in ref_inds if i > 0]}")
#     print(f"GT bbox: {gt_bbox}")                   # [x1, y1, x2, y2]
#     print(f"Image meta: {img_meta}")

#     # Test DataLoader
#     print("Testing DataLoader")
#     loader = build_dataloader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
#     batch = next(iter(loader))
#     imgs, refs, bboxes, metas = batch
#     print(f"Batch images: {imgs.shape}")    # [4, 3, 640, 640]
#     print(f"Batch refs: {refs.shape}")      # [4, 15]
#     print(f"Batch bboxes: {bboxes.shape}")  # [4, 4]
#     print(f"Batch metas: {len(metas)} dicts")  # 4
