(base) sagemaker-user@default:~/CNN_A3$ python experiments/exp01_detection_baseline.py 

2026-04-22 13:22:32 - exp01_detection_baseline - INFO - ================================================================================
2026-04-22 13:22:32 - exp01_detection_baseline - INFO - STARTING EXPERIMENT: exp01_detection_baseline
2026-04-22 13:22:32 - exp01_detection_baseline - INFO - MODE: Using FULL DATASET
2026-04-22 13:22:32 - exp01_detection_baseline - INFO - ================================================================================
2026-04-22 13:22:32 - exp01_detection_baseline - INFO - 
[Step 1/5] Loading dataset configuration...
2026-04-22 13:22:32 - exp01_detection_baseline - INFO - Dataset config loaded from: data/processed/detection/dataset.yaml
2026-04-22 13:22:32 - exp01_detection_baseline - INFO - Dataset root: data/processed/detection
2026-04-22 13:22:32 - exp01_detection_baseline - INFO - Classes: 1 (['dog'])
2026-04-22 13:22:32 - exp01_detection_baseline - INFO - [Step 3/5] Initializing model and trainer...
2026-04-22 13:22:32 - exp01_detection_baseline - INFO - Model config: {'backbone': 'm', 'input_size': 640, 'confidence_threshold': 0.5, 'nms_iou_threshold': 0.45, 'pretrained': True}
2026-04-22 13:22:32 - exp01_detection_baseline - INFO - Training config: {'learning_rate': 0.001, 'batch_size': 16, 'epochs': 120, 'optimizer': 'adam', 'weight_decay': 0.0001, 'early_stopping_patience': 15, 'use_amp': True, 'gradient_accumulation_steps': 1, 'warmup_epochs': 10, 'scheduler': 'cosine'}
Experiment directory created: outputs/exp01_detection_baseline/run_20260422_132232
Using device: cuda
2026-04-22 13:22:32 - exp01_detection_baseline - INFO - 
[Step 3/5] Training model...
================================================================================
DETECTION MODEL TRAINING
================================================================================
Model config: {'backbone': 'm', 'input_size': 640, 'confidence_threshold': 0.5, 'nms_iou_threshold': 0.45, 'pretrained': True}
Training config: {'learning_rate': 0.001, 'batch_size': 16, 'epochs': 120, 'optimizer': 'adam', 'weight_decay': 0.0001, 'early_stopping_patience': 15, 'use_amp': True, 'gradient_accumulation_steps': 1, 'warmup_epochs': 10, 'scheduler': 'cosine'}
Output directory: outputs/exp01_detection_baseline/run_20260422_132232

  Starting fresh training...
Ultralytics 8.4.41 🚀 Python-3.12.9 torch-2.2.2+cu118 CUDA:0 (Tesla T4, 14912MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=0.5, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=data/processed/detection/dataset.yaml, degrees=0.0, deterministic=True, device=None, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=120, erasing=0.4, exist_ok=True, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.45, keras=False, kobj=1.0, line_width=None, lr0=0.001, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolov8m.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=run_20260422_132232, nbs=64, nms=False, opset=None, optimize=False, optimizer=ADAM, overlap_mask=True, patience=15, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/home/sagemaker-user/CNN_A3/outputs/exp01_detection_baseline/run_20260422_132232, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=10, warmup_momentum=0.8, weight_decay=0.0001, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments                     
  0                  -1  1      1392  ultralytics.nn.modules.conv.Conv             [3, 48, 3, 2]                 
  1                  -1  1     41664  ultralytics.nn.modules.conv.Conv             [48, 96, 3, 2]                
  2                  -1  2    111360  ultralytics.nn.modules.block.C2f             [96, 96, 2, True]             
  3                  -1  1    166272  ultralytics.nn.modules.conv.Conv             [96, 192, 3, 2]               
  4                  -1  4    813312  ultralytics.nn.modules.block.C2f             [192, 192, 4, True]           
  5                  -1  1    664320  ultralytics.nn.modules.conv.Conv             [192, 384, 3, 2]              
  6                  -1  4   3248640  ultralytics.nn.modules.block.C2f             [384, 384, 4, True]           
  7                  -1  1   1991808  ultralytics.nn.modules.conv.Conv             [384, 576, 3, 2]              
  8                  -1  2   3985920  ultralytics.nn.modules.block.C2f             [576, 576, 2, True]           
  9                  -1  1    831168  ultralytics.nn.modules.block.SPPF            [576, 576, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  2   1993728  ultralytics.nn.modules.block.C2f             [960, 384, 2]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  2    517632  ultralytics.nn.modules.block.C2f             [576, 192, 2]                 
 16                  -1  1    332160  ultralytics.nn.modules.conv.Conv             [192, 192, 3, 2]              
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 18                  -1  2   1846272  ultralytics.nn.modules.block.C2f             [576, 384, 2]                 
 19                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  2   4207104  ultralytics.nn.modules.block.C2f             [960, 576, 2]                 
 22        [15, 18, 21]  1   3776275  ultralytics.nn.modules.head.Detect           [1, 16, None, [192, 384, 576]]
Model summary: 170 layers, 25,856,899 parameters, 25,856,883 gradients, 79.1 GFLOPs

Transferred 469/475 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 100% ━━━━━━━━━━━━ 5.3MB 345.4MB/s 0.0s
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 101.0±30.3 MB/s, size: 93.4 KB)
train: Scanning /home/sagemaker-user/CNN_A3/data/processed/detection/labels/train... 2461 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 2461/2461 1.1Kit/s 2.3s
train: New cache created: /home/sagemaker-user/CNN_A3/data/processed/detection/labels/train.cache
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 81.7±46.8 MB/s, size: 104.4 KB)
val: Scanning /home/sagemaker-user/CNN_A3/data/processed/detection/labels/val... 2462 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 2462/2462 995.2it/s 2.5s
val: New cache created: /home/sagemaker-user/CNN_A3/data/processed/detection/labels/val.cache
optimizer: Adam(lr=0.001, momentum=0.937) with parameter groups 77 weight(decay=0.0), 84 weight(decay=0.0001), 83 bias(decay=0.0)
Plotting labels to /home/sagemaker-user/CNN_A3/outputs/exp01_detection_baseline/run_20260422_132232/labels.jpg... 
2026/04/22 13:22:45 INFO mlflow.tracking.fluent: Experiment with name '/Shared/Ultralytics' does not exist. Creating a new experiment.
2026/04/22 13:22:45 WARNING mlflow.utils.autologging_utils: MLflow sklearn autologging is known to be compatible with 0.24.1 <= scikit-learn <= 1.6.1, but the installed version is 1.7.2. If you encounter errors during autologging, try upgrading / downgrading scikit-learn to a compatible version, or try upgrading MLflow.
2026/04/22 13:22:46 INFO mlflow.tracking.fluent: Autologging successfully enabled for sklearn.
2026/04/22 13:22:46 WARNING mlflow.utils.autologging_utils: MLflow transformers autologging is known to be compatible with 4.35.2 <= transformers <= 4.51.2, but the installed version is 4.57.6. If you encounter errors during autologging, try upgrading / downgrading transformers to a compatible version, or try upgrading MLflow.
2026/04/22 13:22:46 INFO mlflow.bedrock: Enabled auto-tracing for Bedrock. Note that MLflow can only trace boto3 service clients that are created after this call. If you have already created one, please recreate the client by calling `boto3.client`.
2026/04/22 13:22:46 INFO mlflow.tracking.fluent: Autologging successfully enabled for boto3.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1776864167.151442   11327 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1776864167.157259   11327 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2026/04/22 13:22:50 WARNING mlflow.utils.autologging_utils: MLflow keras autologging is known to be compatible with 3.0.2 <= keras <= 3.9.2, but the installed version is 3.13.2. If you encounter errors during autologging, try upgrading / downgrading keras to a compatible version, or try upgrading MLflow.
2026/04/22 13:22:50 INFO mlflow.tracking.fluent: Autologging successfully enabled for keras.
2026/04/22 13:22:50 INFO mlflow.tracking.fluent: Autologging successfully enabled for tensorflow.
2026/04/22 13:22:55 INFO mlflow.tracking.fluent: Autologging successfully enabled for transformers.
MLflow: logging run_id(34a19c4171f64fa8b7c904dd068da362) to runs/mlflow
MLflow: view at http://127.0.0.1:5000 with 'mlflow server --backend-store-uri runs/mlflow'
MLflow: disable with 'yolo settings mlflow=False'
Image sizes 640 train, 640 val
Using 4 dataloader workers
Logging results to /home/sagemaker-user/CNN_A3/outputs/exp01_detection_baseline/run_20260422_132232
Starting training for 120 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/120      6.38G      1.313      1.365      1.567         35        640: 39% ━━━━╸─────── 60/154 2.2it/s 31.1s<43.7sCorrupt JPEG data: premature end of data segment
      1/120      6.38G      1.269      1.165       1.52         39        640: 69% ━━━━━━━━──── 106/154 2.1it/s 53.1s<22.5sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
      1/120      6.38G      1.248       1.08      1.509         30        640: 100% ━━━━━━━━━━━━ 154/154 2.0it/s 1:16
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.1it/s 24.9s
                   all       2462       2541      0.904      0.402      0.392      0.226

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/120      6.56G      1.229     0.9107      1.506         45        640: 86% ━━━━━━━━━━── 133/154 2.1it/s 1:04<10.0sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment
      2/120      6.56G      1.225     0.9135      1.501         26        640: 100% ━━━━━━━━━━━━ 154/154 2.1it/s 1:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.9s
                   all       2462       2541      0.944       0.85      0.834      0.539

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/120      6.67G      1.198     0.8775      1.498         44        640: 23% ━━╸───────── 36/154 2.1it/s 17.4s<55.4sCorrupt JPEG data: premature end of data segment
      3/120      6.67G      1.183     0.8616      1.485         38        640: 40% ━━━━╸─────── 62/154 2.2it/s 29.3s<42.2sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
      3/120      6.67G      1.215     0.8898      1.494         39        640: 98% ━━━━━━━━━━━╸ 151/154 2.2it/s 1:11<1.4sCorrupt JPEG data: premature end of data segment
      3/120      6.67G      1.217     0.8901      1.496         28        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.7s
                   all       2462       2541      0.926      0.739      0.723      0.427

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/120      6.66G      1.198     0.9214      1.481         41        640: 34% ━━━━──────── 52/154 2.2it/s 25.2s<47.4sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
      4/120      6.66G      1.237     0.9325      1.506         30        640: 100% ━━━━━━━━━━━━ 154/154 2.1it/s 1:12
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.5it/s 22.2s
                   all       2462       2541      0.916      0.732      0.718      0.419

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/120      6.67G      1.271      1.006       1.55         32        640: 27% ━━━───────── 42/154 2.2it/s 19.7s<52.0sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
      5/120      6.67G      1.263     0.9846      1.542         32        640: 49% ━━━━━╸────── 76/154 2.2it/s 35.7s<35.8sCorrupt JPEG data: premature end of data segment
      5/120      6.67G      1.281     0.9856      1.539         23        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.9s
                   all       2462       2541      0.941      0.802      0.794      0.483

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/120      6.66G      1.248     0.9511        1.5         34        640: 47% ━━━━━╸────── 73/154 1.9it/s 34.1s<41.8sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
      6/120      6.66G      1.258     0.9481      1.505         37        640: 58% ━━━━━━━───── 90/154 2.2it/s 41.9s<29.2sCorrupt JPEG data: premature end of data segment
      6/120      6.66G      1.245     0.9275      1.493         31        640: 98% ━━━━━━━━━━━╸ 151/154 2.0it/s 1:10<1.5sCorrupt JPEG data: premature end of data segment
      6/120      6.66G      1.243     0.9261      1.492         25        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.5it/s 21.9s
                   all       2462       2541      0.969      0.801      0.802      0.535

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/120      6.58G      1.168     0.8543      1.439         36        640: 29% ━━━───────── 44/154 2.2it/s 20.9s<50.6sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
      7/120      6.58G      1.203     0.8806      1.472         34        640: 60% ━━━━━━━───── 93/154 2.2it/s 43.2s<27.7s

      7/120      6.58G      1.203     0.8819      1.472         40        640: 61% ━━━━━━━───── 94/154 2.2it/s 43.6s<27.1s
      7/120      6.58G      1.224     0.8969      1.491         27        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.1it/s 24.6s
                   all       2462       2541      0.935        0.9      0.898      0.572

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/120      6.66G      1.237     0.9443      1.535         31        640: 10% ━─────────── 16/154 2.2it/s 7.7s<1:03Corrupt JPEG data: premature end of data segment
      8/120      6.66G      1.225      0.898      1.483         41        640: 72% ━━━━━━━━╸─── 111/154 2.2it/s 51.2s<19.3sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
      8/120      6.66G      1.219     0.8956      1.475         25        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 22% ━━╸───────── 17/77 3.9it/s 4.4s<15.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 23% ━━╸───────── 18/77 3.8it/s 4.7s<15.4s

                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.0s
                   all       2462       2541      0.912      0.891      0.883       0.56

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/120      6.64G       1.22     0.9008      1.471         30        640: 13% ━╸────────── 20/154 2.0it/s 10.0s<1:06Corrupt JPEG data: premature end of data segment
      9/120      6.64G      1.231     0.8881       1.48         42        640: 27% ━━━───────── 42/154 2.2it/s 20.1s<51.6sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
      9/120      6.64G      1.209     0.8658      1.467         34        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.4s
                   all       2462       2541      0.928      0.928      0.911      0.599

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/120      6.65G      1.189     0.8166      1.461         41        640: 2% ──────────── 3/154 1.5it/s 1.8s<1:44Corrupt JPEG data: premature end of data segment
     10/120      6.65G      1.153     0.8097      1.429         39        640: 36% ━━━━──────── 55/154 2.2it/s 25.3s<44.4sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     10/120      6.65G      1.165     0.8209      1.427         24        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.5it/s 22.1s
                   all       2462       2541      0.943      0.876      0.866      0.553

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/120      6.58G      1.176     0.8302       1.47         31        640: 36% ━━━━──────── 55/154 2.2it/s 25.5s<44.9sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     11/120      6.58G       1.18     0.8314      1.454         33        640: 76% ━━━━━━━━━─── 117/154 2.2it/s 53.7s<16.9sCorrupt JPEG data: premature end of data segment
     11/120      6.58G      1.178     0.8317      1.453         22        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 23.8s
                   all       2462       2541      0.955      0.898      0.889      0.605

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/120      6.65G      1.161     0.8132      1.451         27        640: 25% ━━━───────── 39/154 2.1it/s 18.3s<54.3sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     12/120      6.65G      1.142     0.7931      1.428         40        640: 57% ━━━━━━╸───── 88/154 2.2it/s 40.5s<29.6sCorrupt JPEG data: premature end of data segment
     12/120      6.65G      1.136     0.7832      1.416         24        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 24.1s
                   all       2462       2541      0.957       0.88      0.867      0.547

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/120      6.56G      1.101     0.7497        1.4         41        640: 67% ━━━━━━━━──── 103/154 2.2it/s 47.1s<23.0sCorrupt JPEG data: premature end of data segment
     13/120      6.56G      1.101     0.7519      1.404         35        640: 77% ━━━━━━━━━─── 119/154 2.2it/s 54.3s<15.9sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     13/120      6.57G      1.111     0.7559      1.412         34        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.3s
                   all       2462       2541      0.977      0.899      0.893      0.598

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/120      6.64G      1.179     0.7597      1.466         39        640: 14% ━╸────────── 21/154 2.2it/s 10.2s<1:02Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
     14/120      6.64G      1.118     0.7458      1.407         38        640: 60% ━━━━━━━───── 93/154 2.2it/s 42.8s<27.7sCorrupt JPEG data: premature end of data segment
     14/120      6.64G      1.127     0.7501      1.416         30        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.4s
                   all       2462       2541      0.931      0.942      0.936      0.616

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/120      6.57G        1.1     0.7261      1.382         35        640: 64% ━━━━━━━╸──── 98/154 2.2it/s 45.2s<25.0sCorrupt JPEG data: premature end of data segment
     15/120      6.57G      1.099     0.7239      1.384         35        640: 84% ━━━━━━━━━━── 130/154 2.2it/s 59.6s<10.8sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     15/120      6.57G        1.1     0.7281      1.387         23        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.1it/s 24.5s
                   all       2462       2541      0.959       0.95       0.95      0.652

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/120      6.66G      1.101     0.7268      1.384         42        640: 19% ━━────────── 30/154 2.2it/s 14.2s<56.8sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     16/120      6.66G      1.082     0.7268      1.372         39        640: 86% ━━━━━━━━━━── 132/154 2.2it/s 1:01<9.8sCorrupt JPEG data: premature end of data segment
     16/120      6.66G      1.082      0.724      1.372         22        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.4s
                   all       2462       2541      0.979       0.93      0.922      0.636

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/120      6.56G      1.069     0.7204      1.357         29        640: 16% ━╸────────── 25/154 2.2it/s 11.7s<58.2sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     17/120      6.56G      1.068     0.6964      1.354         29        640: 84% ━━━━━━━━━━── 130/154 2.2it/s 59.8s<10.8sCorrupt JPEG data: premature end of data segment
     17/120      6.56G      1.064      0.693      1.354         32        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.3s
                   all       2462       2541      0.975      0.945      0.941       0.66

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/120      6.65G      1.077     0.6923      1.363         33        640: 74% ━━━━━━━━╸─── 114/154 2.2it/s 52.2s<18.0sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     18/120      6.65G      1.078     0.6922      1.363         39        640: 75% ━━━━━━━━╸─── 115/154 2.2it/s 52.6s<17.5sCorrupt JPEG data: premature end of data segment
     18/120      6.65G      1.078     0.6972      1.366         31        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.4s
                   all       2462       2541      0.957      0.941      0.941      0.655

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/120      6.58G      1.058     0.6769      1.343         34        640: 58% ━━━━━━╸───── 89/154 2.2it/s 41.1s<29.3sCorrupt JPEG data: premature end of data segment
     19/120      6.58G      1.051     0.6741       1.34         30        640: 86% ━━━━━━━━━━── 132/154 2.2it/s 1:00<9.9sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     19/120      6.59G      1.042     0.6688      1.334         29        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 38% ━━━━╸─────── 29/77 3.4it/s 9.6s<14.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 24.2s
                   all       2462       2541      0.969      0.955      0.952      0.677

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/120      6.64G      1.003     0.6147      1.297         34        640: 46% ━━━━━╸────── 71/154 2.2it/s 33.1s<37.3sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     20/120      6.64G      1.014     0.6182      1.305         32        640: 68% ━━━━━━━━──── 105/154 2.2it/s 48.7s<22.3sCorrupt JPEG data: premature end of data segment
     20/120      6.64G      1.027     0.6292      1.318         31        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 25% ━━╸───────── 19/77 3.6it/s 5.1s<16.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 26% ━━━───────── 20/77 3.5it/s 5.4s<16.2s

                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.4s
                   all       2462       2541      0.979      0.942      0.943       0.67

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/120      6.57G      1.037     0.6509      1.333         40        640: 65% ━━━━━━━╸──── 100/154 2.2it/s 45.8s<24.4sCorrupt JPEG data: premature end of data segment
Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
     21/120      6.57G      1.032     0.6429      1.327         17        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.1it/s 24.5s
                   all       2462       2541      0.976      0.958      0.953      0.684

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/120      6.66G      1.024     0.6092      1.337         35        640: 16% ━╸────────── 24/154 2.2it/s 11.3s<58.7s
     22/120      6.66G      1.024     0.6097       1.34         32        640: 16% ━╸────────── 25/154 2.2it/s 11.8s<59.6s

     22/120      6.66G      1.021     0.6337      1.334         37        640: 30% ━━━╸──────── 46/154 2.2it/s 21.2s<48.7sCorrupt JPEG data: premature end of data segment
     22/120      6.66G      1.021     0.6306      1.326         32        640: 80% ━━━━━━━━━╸── 123/154 2.2it/s 56.4s<14.0sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     22/120      6.66G      1.024     0.6307      1.327         23        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.4s
                   all       2462       2541      0.974      0.954      0.953      0.682

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/120      6.57G      1.018      0.638      1.323         37        640: 64% ━━━━━━━╸──── 99/154 2.2it/s 45.3s<24.9sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     23/120      6.57G      1.019     0.6391      1.324         35        640: 66% ━━━━━━━╸──── 102/154 2.2it/s 46.6s<23.2sCorrupt JPEG data: premature end of data segment
     23/120      6.57G      1.011     0.6311      1.319         30        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.4s
                   all       2462       2541      0.979      0.943      0.943      0.683

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/120      6.65G     0.9965      0.612      1.312         35        640: 15% ━╸────────── 23/154 2.1it/s 11.5s<1:03Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
     24/120      6.65G     0.9878     0.6042       1.31         35        640: 68% ━━━━━━━━──── 104/154 2.2it/s 48.3s<22.7sCorrupt JPEG data: premature end of data segment
     24/120      6.65G     0.9831     0.6039      1.304         19        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 24.3s
                   all       2462       2541       0.96      0.956      0.952      0.689

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/120      6.57G     0.9777     0.6006      1.291         30        640: 47% ━━━━━╸────── 73/154 2.2it/s 33.9s<36.3s

     25/120      6.57G     0.9779     0.5998      1.291         39        640: 48% ━━━━━╸────── 74/154 2.2it/s 34.3s<35.7s

     25/120      6.57G     0.9787     0.6014      1.292         41        640: 49% ━━━━━╸────── 75/154 2.2it/s 34.8s<36.0s


     25/120      6.57G     0.9775      0.602      1.291         24        640: 49% ━━━━━╸────── 76/154 2.2it/s 35.2s<35.3s

     25/120      6.57G     0.9931     0.6038      1.302         39        640: 75% ━━━━━━━━╸─── 115/154 2.0it/s 53.1s<19.6sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     25/120      6.57G      1.007     0.6127      1.313         38        640: 85% ━━━━━━━━━━── 131/154 2.2it/s 1:00<10.5sCorrupt JPEG data: premature end of data segment
     25/120      6.57G      1.006     0.6138      1.314         34        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.9s
                   all       2462       2541       0.98      0.939      0.933      0.673

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/120      6.66G      1.018     0.6599      1.332         28        640: 3% ──────────── 5/154 1.8it/s 2.8s<1:22Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
     26/120      6.66G     0.9981      0.616      1.309         34        640: 79% ━━━━━━━━━╸── 122/154 2.2it/s 56.2s<14.6sCorrupt JPEG data: premature end of data segment
     26/120      6.66G     0.9848     0.6069      1.301         28        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.5it/s 22.2s
                   all       2462       2541      0.986      0.928      0.923      0.673

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/120      6.56G     0.9829     0.6179      1.289         37        640: 53% ━━━━━━────── 82/154 2.2it/s 37.7s<33.2sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     27/120      6.56G     0.9852      0.616      1.289         35        640: 57% ━━━━━━╸───── 88/154 2.1it/s 40.6s<31.2sCorrupt JPEG data: premature end of data segment
     27/120      6.56G      0.982     0.6166      1.288         35        640: 97% ━━━━━━━━━━━╸ 150/154 2.2it/s 1:09<1.8sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     27/120      6.56G      0.983     0.6163      1.288         29        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 23.8s
                   all       2462       2541      0.969      0.952      0.952      0.685

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/120      6.65G     0.9666     0.6005      1.283         35        640: 18% ━━────────── 27/154 2.2it/s 12.6s<56.8sCorrupt JPEG data: premature end of data segment
     28/120      6.65G     0.9624     0.5884      1.272         38        640: 96% ━━━━━━━━━━━╸ 148/154 2.2it/s 1:08<2.7s

     28/120      6.65G     0.9624      0.589      1.272         28        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.1s
                   all       2462       2541      0.982      0.952      0.953      0.681

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/120      6.57G     0.9971     0.5813      1.303         32        640: 13% ━╸────────── 20/154 2.1it/s 9.8s<1:04Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
     29/120      6.57G     0.9653     0.5715      1.274         38        640: 34% ━━━━──────── 52/154 2.1it/s 24.3s<48.0sCorrupt JPEG data: premature end of data segment
     29/120      6.57G     0.9769     0.5814      1.282         38        640: 94% ━━━━━━━━━━━─ 144/154 2.2it/s 1:06<4.5sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     29/120      6.57G     0.9775      0.583      1.283         25        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 40% ━━━━╸─────── 31/77 3.1it/s 9.3s<14.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 42% ━━━━╸─────── 32/77 3.2it/s 9.6s<14.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.1s
                   all       2462       2541       0.98      0.954      0.953      0.692

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/120      6.65G     0.9531     0.5764      1.263         29        640: 49% ━━━━━╸────── 75/154 2.2it/s 34.9s<36.7sCorrupt JPEG data: premature end of data segment
     30/120      6.65G     0.9597     0.5661       1.27         31        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.7s
                   all       2462       2541      0.989      0.961      0.964      0.699

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/120      6.57G     0.9576     0.5764      1.262         43        640: 9% ━─────────── 14/154 2.0it/s 7.0s<1:09Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
     31/120      6.57G     0.9571     0.5701      1.283         38        640: 30% ━━━╸──────── 46/154 2.2it/s 21.6s<48.4s


     31/120      6.57G     0.9567     0.5687       1.28         30        640: 31% ━━━╸──────── 47/154 2.2it/s 22.1s<48.9s

     31/120      6.57G     0.9591     0.5694      1.283         32        640: 31% ━━━╸──────── 48/154 2.2it/s 22.5s<48.0s
     31/120      6.57G     0.9528     0.5686      1.278         46        640: 68% ━━━━━━━━──── 105/154 2.2it/s 48.5s<22.0sCorrupt JPEG data: premature end of data segment
     31/120      6.57G     0.9479      0.566      1.275         27        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.1s
                   all       2462       2541      0.972      0.952      0.952      0.691

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/120      6.66G     0.9586     0.5602      1.294         33        640: 12% ━─────────── 19/154 2.2it/s 9.0s<1:01Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
     32/120      6.66G     0.9455     0.5563      1.275         37        640: 19% ━━────────── 29/154 2.2it/s 13.5s<56.6sCorrupt JPEG data: premature end of data segment
     32/120      6.66G     0.9465     0.5634      1.275         21        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 23.8s
                   all       2462       2541      0.983      0.964      0.963      0.711

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/120      6.56G      0.942     0.5723      1.285         26        640: 58% ━━━━━━╸───── 89/154 2.2it/s 41.2s<29.3sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     33/120      6.56G     0.9357     0.5669      1.282         30        640: 82% ━━━━━━━━━╸── 127/154 2.1it/s 58.4s<13.0sCorrupt JPEG data: premature end of data segment
     33/120      6.56G     0.9354     0.5677      1.282         35        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 29% ━━━───────── 22/77 3.6it/s 5.8s<15.2s


                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 30% ━━━╸──────── 23/77 3.6it/s 6.1s<15.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 91% ━━━━━━━━━━╸─ 70/77 3.4it/s 20.5s<2.1s

                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 97% ━━━━━━━━━━━╸ 75/77 3.6it/s 21.9s<0.6s



                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.5s
                   all       2462       2541      0.985      0.952      0.953      0.694

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/120      6.67G     0.9274     0.5443      1.263         42        640: 61% ━━━━━━━───── 94/154 2.2it/s 43.2s<27.0sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     34/120      6.67G     0.9265     0.5435      1.266         29        640: 77% ━━━━━━━━━─── 119/154 2.2it/s 54.5s<15.7sCorrupt JPEG data: premature end of data segment
     34/120      6.67G     0.9321     0.5452      1.265         30        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.6s
                   all       2462       2541      0.982      0.965      0.964      0.713

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/120      6.57G     0.8735      0.537      1.231         37        640: 8% ━─────────── 13/154 2.1it/s 6.4s<1:07Corrupt JPEG data: premature end of data segment
     35/120      6.57G     0.9021     0.5225      1.244         35        640: 36% ━━━━──────── 56/154 2.2it/s 26.0s<43.9sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     35/120      6.57G     0.9207      0.536      1.254         33        640: 92% ━━━━━━━━━━━─ 142/154 2.2it/s 1:05<5.4sCorrupt JPEG data: premature end of data segment
     35/120      6.57G     0.9248     0.5388      1.256         32        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 23.7s
                   all       2462       2541      0.981      0.951      0.953      0.693

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/120      6.65G     0.9108     0.5408      1.254         43        640: 55% ━━━━━━╸───── 84/154 2.2it/s 38.6s<31.3sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     36/120      6.65G     0.9089     0.5362      1.255         28        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.2s
                   all       2462       2541      0.986      0.963      0.964      0.703

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/120      6.56G     0.9603     0.5647      1.256         39        640: 14% ━╸────────── 21/154 2.2it/s 9.9s<59.5sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     37/120      6.56G     0.9409     0.5517      1.262         38        640: 85% ━━━━━━━━━━── 131/154 2.2it/s 59.7s<10.5sCorrupt JPEG data: premature end of data segment
     37/120      6.56G     0.9439       0.55      1.266         28        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.3s
                   all       2462       2541      0.982       0.96      0.954      0.703

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/120      6.65G     0.8627     0.5158      1.214         41        640: 9% ━─────────── 14/154 2.2it/s 7.2s<1:04Corrupt JPEG data: premature end of data segment
     38/120      6.65G     0.8832     0.5153      1.222         33        640: 34% ━━━━──────── 52/154 2.2it/s 24.3s<45.5s
     38/120      6.65G     0.8813      0.513      1.221         33        640: 34% ━━━━──────── 53/154 2.2it/s 24.8s<45.7s

     38/120      6.65G     0.8833     0.5161       1.22         43        640: 67% ━━━━━━━━──── 103/154 2.2it/s 47.7s<22.8sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     38/120      6.65G     0.8909     0.5171      1.223         30        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.0s
                   all       2462       2541      0.989      0.953      0.954      0.701

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/120      6.57G     0.9046     0.5216      1.246         32        640: 24% ━━╸───────── 37/154 2.2it/s 17.6s<52.8sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     39/120      6.57G     0.9104     0.5217      1.241         34        640: 32% ━━━╸──────── 50/154 2.2it/s 23.4s<46.6s
     39/120      6.57G      0.911     0.5211      1.242         42        640: 33% ━━━╸──────── 51/154 2.2it/s 23.9s<46.8s
     39/120      6.57G        0.9     0.5183      1.239         26        640: 86% ━━━━━━━━━━── 133/154 2.2it/s 1:01<9.5sCorrupt JPEG data: premature end of data segment
     39/120      6.57G     0.9044     0.5182      1.241         39        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.6s
                   all       2462       2541      0.986      0.966      0.963      0.709

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/120      6.65G     0.8959     0.5091      1.225         40        640: 18% ━━────────── 27/154 2.2it/s 12.7s<57.2sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     40/120      6.65G     0.9016     0.5156      1.238         40        640: 31% ━━━╸──────── 47/154 2.2it/s 21.7s<47.9sCorrupt JPEG data: premature end of data segment
     40/120      6.65G     0.9051     0.5242      1.241         24        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.7s
                   all       2462       2541      0.981      0.965      0.964      0.705

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/120      6.57G     0.8765      0.498      1.233         36        640: 8% ╸─────────── 12/154 2.2it/s 6.2s<1:06Corrupt JPEG data: premature end of data segment
     41/120      6.57G     0.8723     0.5133      1.232         32        640: 66% ━━━━━━━╸──── 101/154 2.2it/s 46.7s<23.8sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     41/120      6.57G     0.8786     0.5122      1.236         38        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 24.2s
                   all       2462       2541      0.985      0.962      0.963      0.706

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/120      6.65G      0.879      0.505      1.215         38        640: 24% ━━╸───────── 37/154 2.2it/s 17.4s<53.3sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     42/120      6.65G     0.8925     0.5065      1.222         37        640: 49% ━━━━━╸────── 75/154 2.2it/s 34.5s<35.6sCorrupt JPEG data: premature end of data segment
     42/120      6.65G     0.8956     0.5022      1.231         41        640: 91% ━━━━━━━━━━╸─ 140/154 2.2it/s 1:04<6.3sCorrupt JPEG data: premature end of data segment
     42/120      6.65G      0.895     0.5034       1.23         18        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 23.8s
                   all       2462       2541      0.985      0.967      0.964      0.717

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/120      6.57G     0.8805     0.5071      1.223         31        640: 63% ━━━━━━━╸──── 97/154 2.2it/s 44.4s<25.5sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     43/120      6.57G     0.8814     0.5065      1.222         29        640: 99% ━━━━━━━━━━━╸ 152/154 2.2it/s 1:09<0.9sCorrupt JPEG data: premature end of data segment
     43/120      6.57G     0.8819     0.5069      1.222         24        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.4s
                   all       2462       2541      0.986      0.965      0.963      0.714

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/120      6.66G     0.9329     0.5315      1.296         37        640: 5% ╸─────────── 8/154 2.1it/s 4.1s<1:09Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
     44/120      6.66G      0.873     0.4969      1.208         31        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 14% ━╸────────── 11/77 3.7it/s 2.9s<17.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 16% ━╸────────── 12/77 3.7it/s 3.2s<17.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.4s
                   all       2462       2541      0.976      0.976      0.973      0.717

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/120      6.58G     0.8463      0.485      1.191         32        640: 51% ━━━━━━────── 78/154 2.0it/s 36.2s<37.1sCorrupt JPEG data: premature end of data segment
     45/120      6.58G     0.8522     0.4884        1.2         38        640: 64% ━━━━━━━╸──── 99/154 2.2it/s 46.0s<25.1sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     45/120      6.58G     0.8581     0.4881      1.201         24        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.3s
                   all       2462       2541      0.978      0.975      0.974      0.718

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/120      6.65G     0.8337     0.4765      1.186         41        640: 40% ━━━━╸─────── 62/154 2.2it/s 28.7s<41.7sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     46/120      6.65G     0.8508     0.4784        1.2         28        640: 84% ━━━━━━━━━━── 130/154 2.1it/s 59.9s<11.5sCorrupt JPEG data: premature end of data segment
     46/120      6.65G     0.8481     0.4743      1.197         30        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.4s
                   all       2462       2541      0.983      0.972      0.974      0.718

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size


     47/120      6.57G     0.8658     0.4908      1.226         32        640: 21% ━━╸───────── 33/154 2.2it/s 15.3s<54.2sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     47/120      6.57G     0.8683      0.495      1.221         34        640: 75% ━━━━━━━━╸─── 115/154 2.2it/s 52.7s<17.7sCorrupt JPEG data: premature end of data segment
     47/120      6.57G     0.8656     0.4968      1.216         40        640: 97% ━━━━━━━━━━━╸ 149/154 2.2it/s 1:08<2.2sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     47/120      6.57G     0.8658     0.4975      1.216         26        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.0it/s 25.3s
                   all       2462       2541      0.986      0.969      0.963      0.707

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/120      6.65G     0.8471     0.4963      1.215         28        640: 7% ╸─────────── 11/154 2.2it/s 5.4s<1:06Corrupt JPEG data: premature end of data segment
     48/120      6.65G     0.8479     0.4713        1.2         29        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.5s
                   all       2462       2541      0.986      0.957      0.954      0.714

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/120      6.56G     0.8333     0.4706      1.185         29        640: 17% ━━────────── 26/154 2.0it/s 13.1s<1:03
     49/120      6.56G     0.8361      0.475       1.19         37        640: 18% ━━────────── 27/154 2.1it/s 13.6s<1:02
     49/120      6.56G     0.8337     0.4714      1.189         33        640: 23% ━━╸───────── 36/154 2.2it/s 17.6s<53.1sCorrupt JPEG data: premature end of data segment
     49/120      6.56G     0.8356     0.4768      1.201         30        640: 62% ━━━━━━━───── 96/154 2.2it/s 44.7s<26.0sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     49/120      6.56G     0.8377     0.4747      1.198         40        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 24.4s
                   all       2462       2541      0.985      0.967      0.964      0.717

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/120      6.65G     0.8158     0.4634      1.188         40        640: 52% ━━━━━━────── 80/154 2.2it/s 37.0s<33.1sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     50/120      6.65G      0.818     0.4678      1.192         37        640: 73% ━━━━━━━━╸─── 113/154 2.2it/s 51.9s<18.7sCorrupt JPEG data: premature end of data segment
     50/120      6.65G     0.8204     0.4711       1.19         32        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.2s
                   all       2462       2541      0.985      0.968      0.964      0.717

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/120      6.56G       0.78     0.4296      1.157         32        640: 5% ╸─────────── 7/154 2.0it/s 3.6s<1:13Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
     51/120      6.56G     0.8129     0.4688      1.177         35        640: 68% ━━━━━━━━──── 105/154 2.2it/s 48.0s<22.1sCorrupt JPEG data: premature end of data segment
     51/120      6.57G     0.8212     0.4698      1.185         30        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.6s
                   all       2462       2541      0.983      0.978      0.974      0.723

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/120      6.66G     0.8599     0.5183      1.257         34        640: 6% ╸─────────── 10/154 2.2it/s 5.0s<1:07Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
     52/120      6.66G      0.812     0.4609      1.184         40        640: 65% ━━━━━━━╸──── 100/154 2.2it/s 46.0s<24.4sCorrupt JPEG data: premature end of data segment
     52/120      6.66G      0.811     0.4602      1.183         41        640: 71% ━━━━━━━━──── 109/154 2.2it/s 50.0s<20.4s
     52/120      6.66G     0.8108     0.4599      1.182         37        640: 71% ━━━━━━━━╸─── 110/154 2.2it/s 50.5s<19.8s

     52/120      6.66G      0.815      0.458       1.18         31        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 24.1s
                   all       2462       2541      0.987      0.973      0.974      0.726

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/120      6.57G     0.8402     0.4572      1.194         46        640: 12% ━─────────── 19/154 2.2it/s 9.0s<1:02Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
     53/120      6.57G     0.8274     0.4549      1.177         36        640: 32% ━━━╸──────── 50/154 2.2it/s 23.2s<46.6sCorrupt JPEG data: premature end of data segment
     53/120      6.57G     0.8299     0.4596      1.192         35        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 23.7s
                   all       2462       2541      0.987      0.976      0.974      0.725

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/120      6.65G     0.8017     0.4444      1.162         34        640: 46% ━━━━━╸────── 71/154 2.2it/s 32.9s<37.3sCorrupt JPEG data: premature end of data segment
     54/120      6.65G     0.8037     0.4471      1.166         34        640: 58% ━━━━━━━───── 90/154 2.2it/s 41.5s<28.9sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     54/120      6.65G     0.8006     0.4475      1.162         28        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.0s
                   all       2462       2541      0.989      0.966      0.964       0.72

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/120      6.56G     0.8195     0.4579       1.17         32        640: 69% ━━━━━━━━──── 106/154 2.2it/s 49.3s<21.4sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     55/120      6.56G     0.8189     0.4594      1.171         34        640: 75% ━━━━━━━━╸─── 115/154 2.2it/s 53.3s<17.6sCorrupt JPEG data: premature end of data segment
     55/120      6.56G     0.8117     0.4547      1.168         28        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 23.0s
                   all       2462       2541      0.983      0.978      0.974      0.726

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/120      6.65G     0.7954     0.4503       1.16         44        640: 40% ━━━━╸─────── 61/154 2.2it/s 28.2s<42.4sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     56/120      6.65G     0.7939      0.445      1.154         32        640: 60% ━━━━━━━───── 92/154 2.2it/s 42.2s<27.7sCorrupt JPEG data: premature end of data segment
     56/120      6.65G     0.7932     0.4386      1.153         45        640: 90% ━━━━━━━━━━╸─ 139/154 2.2it/s 1:04<6.8sCorrupt JPEG data: premature end of data segment
     56/120      6.65G     0.7935     0.4379      1.155         27        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.0it/s 25.4s
                   all       2462       2541      0.988      0.969      0.963      0.719

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/120      6.57G     0.8015      0.439       1.16         32        640: 34% ━━━━──────── 52/154 2.2it/s 24.4s<47.1sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     57/120      6.57G     0.7993     0.4462      1.169         30        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 24.3s
                   all       2462       2541      0.986      0.969      0.964      0.722

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/120      6.65G     0.8156      0.455      1.174         37        640: 12% ━─────────── 18/154 2.2it/s 8.6s<1:02Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
     58/120      6.65G     0.8101      0.453      1.169         29        640: 60% ━━━━━━━───── 92/154 2.2it/s 42.3s<27.8sCorrupt JPEG data: premature end of data segment
     58/120      6.65G     0.8076     0.4495      1.168         29        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.3s
                   all       2462       2541      0.987      0.976      0.974       0.72

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/120      6.57G     0.7835     0.4457       1.16         45        640: 13% ━╸────────── 20/154 2.2it/s 9.9s<1:01Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
     59/120      6.57G      0.802     0.4429       1.17         38        640: 62% ━━━━━━━───── 95/154 2.1it/s 43.9s<27.6sCorrupt JPEG data: premature end of data segment
     59/120      6.57G      0.793     0.4375      1.172         20        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 23.7s
                   all       2462       2541      0.987      0.973      0.974      0.726

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/120      6.66G     0.7691     0.4212      1.148         40        640: 67% ━━━━━━━━──── 103/154 2.2it/s 47.6s<22.9sCorrupt JPEG data: premature end of data segment
     60/120      6.66G     0.7693     0.4187      1.148         40        640: 82% ━━━━━━━━━╸── 127/154 2.2it/s 58.8s<12.1sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     60/120      6.66G     0.7736     0.4188      1.146         31        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.6s
                   all       2462       2541      0.988      0.966      0.964       0.72

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/120      6.57G     0.8362     0.4826      1.186         39        640: 0% ──────────── 0/154  0.5sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     61/120      6.58G     0.7888     0.4382      1.166         41        640: 86% ━━━━━━━━━━── 133/154 2.2it/s 1:01<9.5sCorrupt JPEG data: premature end of data segment
     61/120      6.58G     0.7861     0.4349      1.163         34        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.6s
                   all       2462       2541      0.986      0.971      0.974      0.728

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/120      6.64G     0.7726     0.4331      1.154         30        640: 43% ━━━━━─────── 66/154 2.2it/s 30.4s<39.5sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     62/120      6.64G     0.7731     0.4327      1.156         35        640: 45% ━━━━━─────── 70/154 2.2it/s 32.2s<37.8sCorrupt JPEG data: premature end of data segment
     62/120      6.64G     0.7698     0.4295      1.152         40        640: 97% ━━━━━━━━━━━╸ 149/154 2.2it/s 1:08<2.3sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     62/120      6.64G     0.7691     0.4298      1.149         29        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.1it/s 24.6s
                   all       2462       2541      0.988      0.974      0.974      0.728

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/120      6.57G     0.7892     0.4435      1.169         41        640: 12% ━─────────── 19/154 2.2it/s 9.0s<1:01Corrupt JPEG data: premature end of data segment
     63/120      6.58G     0.7737     0.4277      1.144         28        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 24.4s
                   all       2462       2541      0.986      0.966      0.964      0.724

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/120      6.64G     0.7804     0.4368      1.147         36        640: 31% ━━━╸──────── 48/154 2.2it/s 22.4s<47.3sCorrupt JPEG data: premature end of data segment
     64/120      6.64G     0.7611     0.4286      1.137         43        640: 49% ━━━━━╸────── 75/154 2.0it/s 34.9s<39.2sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     64/120      6.64G     0.7609      0.428      1.137         34        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.6s
                   all       2462       2541      0.989      0.976      0.974      0.729

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/120      6.56G     0.7466     0.4133      1.126         41        640: 25% ━━╸───────── 38/154 2.2it/s 17.8s<51.8s
     65/120      6.56G     0.7466     0.4122      1.125         34        640: 25% ━━━───────── 39/154 2.2it/s 18.3s<52.1s

     65/120      6.56G     0.7471     0.4133      1.126         39        640: 31% ━━━╸──────── 47/154 2.2it/s 21.9s<48.6sCorrupt JPEG data: premature end of data segment
     65/120      6.56G     0.7349     0.4076      1.124         47        640: 71% ━━━━━━━━──── 109/154 2.2it/s 50.5s<20.3sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     65/120      6.56G     0.7426     0.4112      1.131         37        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 24.2s
                   all       2462       2541      0.991      0.969      0.964       0.73

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/120      6.67G     0.7232     0.4112      1.128         36        640: 36% ━━━━──────── 55/154 2.2it/s 25.5s<44.4s

     66/120      6.67G     0.7229     0.4116      1.128         33        640: 48% ━━━━━╸────── 74/154 2.2it/s 34.1s<36.2sCorrupt JPEG data: premature end of data segment
     66/120      6.67G     0.7318     0.4172      1.133         37        640: 64% ━━━━━━━╸──── 98/154 2.2it/s 45.0s<25.4sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     66/120      6.67G     0.7457     0.4202      1.137         35        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.3s
                   all       2462       2541      0.986      0.978      0.974      0.722

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/120      6.56G     0.7392     0.4155       1.13         38        640: 44% ━━━━━─────── 67/154 2.2it/s 30.9s<39.7sCorrupt JPEG data: premature end of data segment
     67/120      6.56G     0.7405     0.4158      1.127         27        640: 61% ━━━━━━━───── 94/154 2.2it/s 43.1s<26.9s
     67/120      6.56G     0.7398     0.4158      1.126         39        640: 62% ━━━━━━━───── 95/154 2.2it/s 43.6s<26.8s

     67/120      6.56G     0.7371     0.4117      1.125         34        640: 80% ━━━━━━━━━╸── 123/154 2.2it/s 56.2s<14.3sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     67/120      6.56G     0.7405     0.4098      1.127         35        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.1it/s 24.6s
                   all       2462       2541      0.985      0.979      0.974      0.726

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/120      6.64G     0.7532     0.4285      1.134         40        640: 32% ━━━╸──────── 50/154 2.2it/s 23.0s<46.8sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     68/120      6.64G     0.7346     0.4164      1.127         39        640: 75% ━━━━━━━━━─── 116/154 2.2it/s 53.2s<17.1sCorrupt JPEG data: premature end of data segment
     68/120      6.64G     0.7348     0.4138      1.125         31        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.6s
                   all       2462       2541      0.989      0.969      0.964      0.728

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/120      6.56G     0.7457     0.4208      1.154         34        640: 7% ╸─────────── 11/154 2.0it/s 5.5s<1:10Corrupt JPEG data: premature end of data segment
     69/120      6.56G     0.7154      0.392      1.115         36        640: 23% ━━╸───────── 36/154 2.2it/s 17.5s<53.7sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     69/120      6.56G     0.7267     0.4008      1.117         32        640: 93% ━━━━━━━━━━━─ 143/154 2.2it/s 1:06<5.0sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     69/120      6.56G     0.7262     0.4014      1.118         27        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 79% ━━━━━━━━━╸── 61/77 3.0it/s 20.9s<5.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 81% ━━━━━━━━━╸── 62/77 3.1it/s 21.2s<4.8s

                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 82% ━━━━━━━━━╸── 63/77 3.2it/s 21.5s<4.3s

                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 83% ━━━━━━━━━╸── 64/77 3.3it/s 21.8s<3.9s

                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.0it/s 25.3s
                   all       2462       2541      0.989      0.978      0.974      0.725

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/120      6.64G     0.7428     0.4136      1.128         33        640: 38% ━━━━╸─────── 59/154 2.2it/s 27.4s<43.7sCorrupt JPEG data: premature end of data segment
     70/120      6.64G     0.7231     0.4036      1.115         25        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.8s
                   all       2462       2541      0.988      0.976      0.974      0.734

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/120      6.57G     0.7261       0.39      1.124         42        640: 43% ━━━━━─────── 66/154 2.2it/s 30.3s<39.3sCorrupt JPEG data: premature end of data segment
     71/120      6.57G     0.7253     0.3894      1.123         34        640: 45% ━━━━━─────── 70/154 2.2it/s 32.1s<37.6sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     71/120      6.57G     0.7209      0.386      1.118         30        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:11
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 23.8s
                   all       2462       2541      0.988      0.979      0.974      0.729

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/120      6.66G     0.7317     0.3886      1.129         38        640: 13% ━╸────────── 20/154 2.2it/s 9.5s<1:00Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
     72/120      6.66G     0.7212     0.3909      1.123         37        640: 19% ━━────────── 30/154 2.0it/s 14.3s<1:02Corrupt JPEG data: premature end of data segment
     72/120      6.66G     0.7204     0.3882      1.116         36        640: 95% ━━━━━━━━━━━─ 146/154 2.2it/s 1:07<3.6sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     72/120      6.66G     0.7195     0.3881      1.117         29        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.1it/s 24.9s
                   all       2462       2541       0.99      0.978      0.984      0.733

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/120      6.56G     0.7073     0.3934       1.11         34        640: 19% ━━────────── 29/154 2.2it/s 13.5s<55.9sCorrupt JPEG data: premature end of data segment
     73/120      6.56G     0.7032     0.3865      1.103         32        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.2s
                   all       2462       2541       0.99      0.972      0.974      0.731

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/120      6.64G     0.6939     0.3998      1.084         33        640: 2% ──────────── 3/154 1.3it/s 2.0s<1:57Corrupt JPEG data: premature end of data segment
     74/120      6.64G     0.6857      0.382      1.099         27        640: 20% ━━────────── 31/154 2.2it/s 14.8s<55.1s
     74/120      6.64G     0.7011     0.3796      1.111         34        640: 82% ━━━━━━━━━╸── 127/154 2.2it/s 58.2s<12.1sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     74/120      6.64G     0.7002     0.3785      1.109         25        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.1it/s 24.9s
                   all       2462       2541      0.991      0.974      0.974      0.732

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/120      6.57G     0.7067     0.3984      1.113         33        640: 9% ━─────────── 14/154 2.2it/s 6.8s<1:03Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
     75/120      6.57G     0.6884       0.38      1.098         35        640: 52% ━━━━━━────── 80/154 2.2it/s 37.0s<33.3sCorrupt JPEG data: premature end of data segment
     75/120      6.57G     0.6933     0.3827      1.103         41        640: 69% ━━━━━━━━──── 107/154 2.1it/s 49.5s<22.0s
     75/120      6.57G     0.6924     0.3808      1.102         30        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 23.8s
                   all       2462       2541      0.985      0.979      0.974       0.73

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/120      6.65G      0.685      0.377      1.093         31        640: 11% ━─────────── 17/154 2.2it/s 8.2s<1:03

     76/120      6.65G     0.6828     0.3801      1.092         30        640: 12% ━─────────── 18/154 2.2it/s 8.6s<1:01
     76/120      6.65G     0.6713     0.3721       1.09         31        640: 34% ━━━━──────── 52/154 2.1it/s 24.2s<49.6sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     76/120      6.65G     0.6796     0.3758       1.09         36        640: 49% ━━━━━╸────── 75/154 2.2it/s 34.8s<35.3sCorrupt JPEG data: premature end of data segment
     76/120      6.65G      0.696     0.3775      1.097         32        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.5s
                   all       2462       2541      0.988      0.972      0.974      0.722

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/120      6.57G     0.6654      0.367      1.038         36        640: 2% ──────────── 3/154 1.3it/s 2.1s<1:56Corrupt JPEG data: premature end of data segment
     77/120      6.57G     0.6895     0.3819      1.091         34        640: 29% ━━━╸──────── 45/154 2.2it/s 21.2s<48.7sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     77/120      6.57G     0.6925     0.3789      1.105         35        640: 46% ━━━━━╸────── 71/154 2.2it/s 32.9s<37.8s
     77/120      6.57G     0.6924     0.3795      1.104         39        640: 47% ━━━━━╸────── 72/154 2.2it/s 33.4s<37.1s

     77/120      6.57G     0.6917     0.3789      1.104         34        640: 47% ━━━━━╸────── 73/154 2.2it/s 33.8s<36.5s

     77/120      6.57G     0.6925     0.3788      1.105         36        640: 48% ━━━━━╸────── 74/154 2.2it/s 34.3s<35.8s
     77/120      6.57G     0.6909     0.3757      1.101         29        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.1it/s 24.5s
                   all       2462       2541      0.986      0.975      0.974      0.733

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/120      6.65G     0.7126     0.3862      1.092         41        640: 12% ━─────────── 19/154 2.2it/s 9.0s<1:01Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
     78/120      6.65G      0.695     0.3785      1.088         26        640: 55% ━━━━━━╸───── 84/154 2.2it/s 38.6s<31.3sCorrupt JPEG data: premature end of data segment
     78/120      6.65G     0.6862     0.3809      1.086         33        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.4s
                   all       2462       2541      0.987      0.967      0.964      0.725

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/120      6.56G     0.6758     0.3672      1.081         37        640: 40% ━━━━╸─────── 61/154 2.2it/s 28.5s<41.7sCorrupt JPEG data: premature end of data segment
     79/120      6.56G     0.6814     0.3685      1.084         40        640: 58% ━━━━━━━───── 90/154 2.2it/s 41.5s<28.6sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     79/120      6.57G     0.6729     0.3633       1.08         31        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 24.1s
                   all       2462       2541      0.988      0.979      0.974      0.726

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/120      6.65G     0.6715     0.3706      1.085         30        640: 5% ╸─────────── 7/154 2.0it/s 3.6s<1:12Corrupt JPEG data: 240 extraneous bytes before marker 0xd9
     80/120      6.65G     0.6818      0.367      1.084         37        640: 85% ━━━━━━━━━━── 131/154 2.2it/s 1:00<10.4sCorrupt JPEG data: premature end of data segment
     80/120      6.65G     0.6811     0.3686      1.085         25        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.4s
                   all       2462       2541      0.991       0.97      0.974      0.733

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/120      6.57G     0.6683     0.3582      1.073         37        640: 38% ━━━━╸─────── 59/154 2.2it/s 27.5s<43.5sCorrupt JPEG data: premature end of data segment
     81/120      6.57G     0.6677     0.3578      1.081         33        640: 86% ━━━━━━━━━━── 133/154 2.2it/s 1:02<9.4sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     81/120      6.57G     0.6647     0.3549      1.078         41        640: 92% ━━━━━━━━━━╸─ 141/154 2.2it/s 1:05<5.9s

     81/120      6.57G     0.6647     0.3546      1.077         46        640: 92% ━━━━━━━━━━━─ 142/154 2.2it/s 1:06<5.4s

     81/120      6.57G     0.6649     0.3547      1.078         32        640: 93% ━━━━━━━━━━━─ 143/154 2.2it/s 1:06<5.0s

     81/120      6.57G     0.6666     0.3557      1.077         25        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.2it/s 24.4s
                   all       2462       2541      0.991      0.968      0.964       0.73

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     82/120      6.66G     0.6647     0.3599      1.079         38        640: 28% ━━━───────── 43/154 2.2it/s 19.8s<49.8sCorrupt JPEG data: premature end of data segment
     82/120      6.66G     0.6652     0.3595      1.087         51        640: 74% ━━━━━━━━╸─── 114/154 2.2it/s 52.3s<18.1sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     82/120      6.66G      0.666      0.361      1.087         31        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.3it/s 23.6s
                   all       2462       2541      0.987      0.973      0.974       0.73

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     83/120      6.57G     0.6679     0.3515      1.088         40        640: 24% ━━╸───────── 37/154 2.2it/s 17.1s<52.2s

     83/120      6.57G     0.6591     0.3453      1.072         35        640: 34% ━━━━──────── 52/154 2.2it/s 23.8s<45.7sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     83/120      6.57G     0.6593     0.3452      1.073         32        640: 47% ━━━━━╸────── 73/154 2.1it/s 33.3s<37.7sCorrupt JPEG data: premature end of data segment
     83/120      6.57G     0.6553     0.3499      1.076         30        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.5s
                   all       2462       2541       0.99      0.971      0.974       0.73

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     84/120      6.65G     0.6624     0.3513      1.086         40        640: 7% ╸─────────── 11/154 2.2it/s 5.4s<1:05Corrupt JPEG data: premature end of data segment
     84/120      6.65G      0.677      0.361      1.084         42        640: 14% ━╸────────── 21/154 2.0it/s 10.4s<1:08
     84/120      6.65G     0.6778     0.3621      1.084         35        640: 14% ━╸────────── 22/154 2.0it/s 10.8s<1:05

     84/120      6.65G     0.6726     0.3595      1.079         38        640: 15% ━╸────────── 23/154 2.0it/s 11.4s<1:06

     84/120      6.65G     0.6497     0.3513      1.066         31        640: 45% ━━━━━─────── 69/154 2.2it/s 32.3s<38.5sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     84/120      6.65G     0.6536     0.3575      1.074         28        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.1it/s 24.5s
                   all       2462       2541      0.992      0.977      0.974      0.728





      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     85/120      6.57G     0.6427     0.3538      1.074         29        640: 43% ━━━━━─────── 66/154 2.1it/s 30.6s<41.3sCorrupt JPEG data: premature end of data segment
     85/120      6.57G      0.633     0.3518       1.07         33        640: 65% ━━━━━━━╸──── 100/154 2.2it/s 46.2s<24.3sCorrupt JPEG data: 240 extraneous bytes before marker 0xd9
     85/120      6.57G     0.6402     0.3528      1.071         31        640: 100% ━━━━━━━━━━━━ 154/154 2.2it/s 1:10
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 29% ━━━───────── 22/77 3.5it/s 6.0s<15.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 30% ━━━╸──────── 23/77 3.5it/s 6.2s<15.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 3.4it/s 22.5s
                   all       2462       2541      0.992      0.976      0.974      0.729
EarlyStopping: Training stopped early as no improvement observed in last 15 epochs. Best results observed at epoch 70, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

85 epochs completed in 2.270 hours.
Optimizer stripped from /home/sagemaker-user/CNN_A3/outputs/exp01_detection_baseline/run_20260422_132232/weights/last.pt, 52.0MB
Optimizer stripped from /home/sagemaker-user/CNN_A3/outputs/exp01_detection_baseline/run_20260422_132232/weights/best.pt, 52.0MB

Validating /home/sagemaker-user/CNN_A3/outputs/exp01_detection_baseline/run_20260422_132232/weights/best.pt...
Ultralytics 8.4.41 🚀 Python-3.12.9 torch-2.2.2+cu118 CUDA:0 (Tesla T4, 14912MiB)
Model summary (fused): 93 layers, 25,840,339 parameters, 0 gradients, 78.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 77/77 2.6it/s 29.4s
                   all       2462       2541      0.988      0.976      0.974      0.734
Speed: 0.2ms preprocess, 6.8ms inference, 0.0ms loss, 0.9ms postprocess per image
Results saved to /home/sagemaker-user/CNN_A3/outputs/exp01_detection_baseline/run_20260422_132232
MLflow: results logged to runs/mlflow
MLflow: disable with 'yolo settings mlflow=False'

================================================================================
TRAINING COMPLETE
================================================================================
Best model saved to: outputs/exp01_detection_baseline/run_20260422_132232/model/best_model.pt
Training log saved to: outputs/exp01_detection_baseline/run_20260422_132232/logs/training_log.csv
2026-04-22 15:39:47 - exp01_detection_baseline - INFO - Training completed successfully!
2026-04-22 15:39:47 - exp01_detection_baseline - INFO - 
[Step 4/5] Evaluating model on test set...
2026-04-22 15:39:47 - exp01_detection_baseline - INFO - Reloading best model weights from: outputs/exp01_detection_baseline/run_20260422_132232/model/best_model.pt
2026-04-22 15:39:47 - exp01_detection_baseline - INFO - Best model loaded successfully
================================================================================
DETECTION MODEL EVALUATION
================================================================================
Ultralytics 8.4.41 🚀 Python-3.12.9 torch-2.2.2+cu118 CUDA:0 (Tesla T4, 14912MiB)
Model summary (fused): 93 layers, 25,840,339 parameters, 0 gradients, 78.7 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1397.7±709.2 MB/s, size: 55.9 KB)
val: Scanning /home/sagemaker-user/CNN_A3/data/processed/detection/labels/val.cache... 2462 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 2462/2462 688.4Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 154/154 3.5it/s 43.5s
                   all       2462       2541      0.981      0.981      0.993      0.744
Speed: 0.6ms preprocess, 14.2ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to /home/sagemaker-user/CNN_A3/runs/detect/val

Evaluation Results:
  mAP@0.5: 0.9931
  mAP@0.5:0.95: 0.7438
  Precision: 0.9807
  Recall: 0.9810
  F1-Score: 0.9809

Metrics saved to: outputs/exp01_detection_baseline/run_20260422_132232/logs/evaluation_metrics.json
Report saved to: outputs/exp01_detection_baseline/run_20260422_132232/logs/experiment_report.md
2026-04-22 15:40:35 - exp01_detection_baseline - INFO - 
================================================================================
2026-04-22 15:40:35 - exp01_detection_baseline - INFO - EXPERIMENT COMPLETED SUCCESSFULLY
2026-04-22 15:40:35 - exp01_detection_baseline - INFO - ================================================================================
2026-04-22 15:40:35 - exp01_detection_baseline - INFO - Results saved to: outputs/exp01_detection_baseline/run_20260422_132232
2026-04-22 15:40:35 - exp01_detection_baseline - INFO - Metrics: {'mAP50': 0.9930897172570865, 'mAP50_95': 0.743801346448008, 'precision': 0.9807216631431066, 'recall': 0.9809968954095153, 'f1_score': 0.9808592549685465}