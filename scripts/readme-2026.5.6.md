  GPU/CPU 现状

  现在就是用 GPU 跑的。两个脚本里 --device 默认值都是 "0"（即 cuda:0）。Ultralytics 的约定：
  - device="0" → GPU 0
  - device="cpu" → CPU
  - device="" 或不传 → 自动选（有 GPU 用 GPU）

  只要本机 PyTorch 装的是 CUDA 版且 RTX 5060 Ti 被识别，训练/推理就在 GPU 上。验证一句：
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

  新脚本我把默认值改成 "0" 同时支持 cuda / cpu / 0,1 等写法，并在启动时打印实际 device 名称，避免再产生疑问。

  关于"YOLO 26"

  Ultralytics 目前（2025-01）发布到 YOLO11，没有"YOLO26"。我猜你是想试更大的现代模型，新脚本支持下面这些权重一键切换：

  ┌───────────────┬────────────────────┬─────────────────────────────────┐
  │     选项      │        权重        │              备注               │
  ├───────────────┼────────────────────┼─────────────────────────────────┤
  │ yolov8s/m/l/x │ yolov8{s,m,l,x}.pt │ 经典                            │
  ├───────────────┼────────────────────┼─────────────────────────────────┤
  │ yolo11s/m/l/x │ yolo11{s,m,l,x}.pt │ 最新 YOLO，同精度更小更快       │
  ├───────────────┼────────────────────┼─────────────────────────────────┤
  │ rtdetr-l      │ rtdetr-l.pt        │ DETR 系列，ultralytics 原生支持 │
  ├───────────────┼────────────────────┼─────────────────────────────────┤
  │ rtdetr-x      │ rtdetr-x.pt        │ 更大的 RT-DETR                  │
  └───────────────┴────────────────────┴─────────────────────────────────┘

  RT-DETR 是 DETR 家族里唯一适合实时的（Baidu 出品，无 NMS）。原版 DETR/Deformable-DETR 不适合实时（>100ms/帧）。

  下面创建新脚本：

● 写好了。两个新脚本：

  - scripts/train_v2.py — --arch 支持 yolov8{n,s,m,l,x} / yolo11{n,s,m,l,x} / rtdetr-l / rtdetr-x，自动选 YOLO 或 RTDETR 加载器。RT-DETR 自动关闭 mosaic/copy_paste/mixup（DETR
  系列对这些敏感）。启动会打印实际 CUDA 设备名。
  - scripts/infer_video_v2.py — 复用 infer_video.py 里的 FrameAnnotator / open_writer / resolve_model_classes / status_label/color，以及 utils_ppe 里的 TrackVoter /
  associate_ppe，没有重复实现任何核心逻辑。

  推荐组合（16GB 显存）

  实时性优先 → yolo11s 或 yolo11m（imgsz=1280 batch≈16/8）
  精度优先 → yolo11l 或 rtdetr-l（batch 调小到 4~8）

  例如：
  # 训练 yolo11s
  python scripts/train_v2.py --arch yolo11s --imgsz 1280 --epochs 150 --batch -1

  # 训练 RT-DETR-L（batch 显式给小一点更稳）
  python scripts/train_v2.py --arch rtdetr-l --imgsz 1280 --epochs 100 --batch 8

  # 推理
  python scripts/infer_video_v2.py \
    --arch yolo11s \
    --weights runs/ppe/yolo11s_imgsz1280/weights/best.pt \
    --source 'test-video/test-video-1倍焦距切换.mp4' \
    --no-show --half

  加 --half 可以让 RTX 5060 Ti 上的推理 FPS 再涨 30%~50%（FP16 张量核心）。

  显存预估（imgsz=1280）

  ┌──────────┬────────────┬──────────────────┐
  │   模型   │ 训练 batch │ 推理 FPS（粗估） │
  ├──────────┼────────────┼──────────────────┤
  │ yolov8n  │ 32+        │ 50~70            │
  ├──────────┼────────────┼──────────────────┤
  │ yolo11s  │ 16         │ 40~55            │
  ├──────────┼────────────┼──────────────────┤
  │ yolo11m  │ 8          │ 25~35            │
  ├──────────┼────────────┼──────────────────┤
  │ yolo11l  │ 4~6        │ 15~22            │
  ├──────────┼────────────┼──────────────────┤
  │ rtdetr-l │ 6~8        │ 18~28            │
  └──────────┴────────────┴──────────────────┘

  实测以 train_v2.py 启动后 ultralytics AutoBatch (

● --batch -1) 给的数为准。

● Write(scripts/train_v2.py)
  ⎿  Wrote 142 lines to scripts/train_v2.py
       1 """Train larger detector variants (YOLOv8 s/m/l/x, YOLO11, RT-DETR) on wrj_ppe.
       2
       3 Reuses dataset / augmentation choices from train.py but exposes a wider model
       4 zoo via --arch.  Default device is GPU 0; pass --device cpu to force CPU.
       5
       6 Usage:
       7     python scripts/train_v2.py --arch yolo11s --imgsz 1280 --epochs 150
       8     python scripts/train_v2.py --arch rtdetr-l --imgsz 1280 --epochs 100 --batch 8
       9 """
      10 from __future__ import annotations
     … +132 lines (ctrl+o to expand)

● Write(scripts/infer_video_v2.py)
  ⎿  Wrote 263 lines to scripts/infer_video_v2.py
       1 """Real-time PPE inference with selectable backbone (YOLOv8/YOLO11/RT-DETR).
       2
       3 This is a thin wrapper around the same pipeline used in infer_video.py:
       4 the heavy lifting (TrackVoter, associate_ppe, FrameAnnotator, FFmpegWriter)
       5 is reused unchanged.  Differences vs infer_video.py:
       6
       7 - --arch flag picks YOLO / RT-DETR loader automatically based on the weights.
       8 - Defaults --device to '0' and prints the resolved CUDA device.
       9 - Falls back to detect-then-associate without ByteTrack when the model is
      10   RT-DETR (ultralytics RTDETR.track is supported, but tracker config may
     … +253 lines (ctrl+o to expand)

● 两个新脚本已创建。

  文件
  - scripts/train_v2.py — 支持 yolov8/yolo11/rtdetr 全系列，--device 默认 0，启动打印 cuda:0 -> RTX 5060 Ti
  - scripts/infer_video_v2.py — 复用 infer_video.py 的 FrameAnnotator/open_writer/resolve_model_classes/status_* 和 utils_ppe 的 TrackVoter/associate_ppe，无重复实现

  关于 GPU/CPU：你之前一直就在用 GPU（原脚本 --device 默认 "0"）。新脚本同样默认 GPU，并在启动时打印实际设备名，避免疑问。先验证：
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

  关于"YOLO26"：截至当前 Ultralytics 最新发布是 YOLO11，没有 YOLO26。新脚本里推荐试 yolo11s/m 或 rtdetr-l（DETR 家族里唯一实时的，Baidu 出品，无 NMS）。

  典型用法
  # 训练 yolo11s（推荐，精度/速度平衡）
  python scripts/train_v2.py --arch yolo11s --imgsz 1280 --epochs 150

  # 训练 RT-DETR-L（注意 batch 给小一点）
  python scripts/train_v2.py --arch rtdetr-l --imgsz 1280 --epochs 100 --batch 8

  # 推理
  python scripts/infer_video_v2.py --arch yolo11s \
    --weights runs/ppe/yolo11s_imgsz1280/weights/best.pt \
    --source 'test-video/test-video-1倍焦距切换.mp4' --no-show --half

  加 --half (FP16) 在 RTX 5060 Ti 上推理 FPS 大约 +30%~50%。