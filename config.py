class Config:
    img_dir = "/kaggle/input/datasets/jeffaudi/coco-2014-dataset-for-yolov3/coco2014/images/train2014"
    ann_file = "/kaggle/input/datasets/minhkhoai/seqtr-annotations-weights/annotations/refcoco-unc/instances.json"

    img_size = 640
    max_token = 15
    glove_dim = 300

    # Tham số cho visual backbone
    backbone_out_channels = 1024 # số channel đầu ra của backbone (ResNet-50)

    # Tham số cho language branch
    gru_hidden = 512 

    # Tham số cho kiến trúc Transformer
    d_model = 256
    nhead = 8
    dim_feedforward = 1024 
    enc_layers = 6
    dec_layers = 3
    dropout = 0.1

    # Quantization
    num_bin = 1000
    vocab_size = 1001 # thêm 1 token cho End token

    # Tham số cho training
    batch_size = 16
    lr = 6.25e-5
    epochs = 60
    warmup_epochs = 5 
    decay_epoch = 50 
    decay_ratio = 0.1
    grad_clip = 0.15

    # EMA duy trì 1 bản sao "trung bình" của model weights.
    # ema_decay = 0.999 nghĩa là: shadow = 0.999 * shadow + 0.001 * current_weights
    ema = True
    ema_decay = 0.999

    # Soft target
    label_smoothing = 0.1

    # Ablation - pooling của nhánh ngôn ngữ
    pooling = "max"

    # Ablation - Token Weights
    token_weights = [2.0, 1.5, 1.0, 1.0, 0.5 ]

    # Logging & Checkpoints
    log_interval = 80

    # Random seed
    seed = 6666
    num_workers = 2
    
    # Thư mục lưu checkpoint & log.
    work_dir = "/kaggle/working/checkpoints"


    


