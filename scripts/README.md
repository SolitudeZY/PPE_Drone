# PPE Drone — 阶段性 README（训练 / 验证 / 推理 流水线）

本文件汇总目前 `scripts/` 下各脚本的用途、典型命令和推理脚本超参数说明，作为当前 6 类数据集（`datasets/20260506`，类别：`person, helmet, vest, head, crane, excavator`）阶段的参考。

---

## 1. 流水线概览

```
  数据准备                 训练                   评估                   推理
──────────       ──────────────────       ────────────       ─────────────────────
fix_data_yaml ─► train.py / train_v2 ─► val.py ─► infer_video.py / infer_video_v2.py
                                                            │
                                                            ├─► output.mp4
                                                            ├─► alerts.jsonl
                                                            └─► alert_snapshots/
                                                            │
                                                            └─► extract_hard_negatives.py（回灌训练）
```

- `train.py` 是早期 4 类（person/helmet/vest/head）流水线，仍可用于旧数据集 `wrj_ppe`。
- `train_v2.py` 面向 6 类新数据集 `datasets/20260506`，支持 YOLOv8 / YOLO11 / RT-DETR 多种 backbone。
- 推理同理：`infer_video.py` 是基础版；`infer_video_v2.py` 增加了 `--arch` 切换 + 设备打印 + crane/excavator 装备类绘制。
- PPE 合规判定（戴帽/穿反光衣）只用 person/helmet/head/vest 四类；crane / excavator 仅做态势可视化，不进入告警逻辑。

### 文件清单

| 文件 | 作用 |
|------|------|
| `fix_data_yaml.py` | 把数据集自带 Windows 绝对路径的 `data.yaml` 转成本机可用的 `data_local.yaml` |
| `train.py` | YOLOv8 n/s/m 训练（旧 4 类数据集 `wrj_ppe`） |
| `train_v2.py` | YOLOv8 / YOLO11 / RT-DETR 训练（新 6 类数据集 `20260506`） |
| `val.py` | 在 val/test split 上评估权重，输出每类 mAP50 / mAP50-95 |
| `infer_video.py` | 视频推理基础版（YOLO，4 类） |
| `infer_video_v2.py` | 视频推理 v2：可选 arch、装备类可视化、告警快照 |
| `utils_ppe.py` | `TrackVoter`（滑动窗口投票 + 粘滞告警）+ `associate_ppe`（几何包含判定） |
| `extract_hard_negatives.py` | 从 `alerts.jsonl` 抽取告警帧用于难负样本回灌训练 |

---

## 2. 数据准备

### `fix_data_yaml.py`

旧数据集 `wrj_ppe/data.yaml` 内含形如 `E:/...` 的 Windows 绝对路径，ultralytics 读不到图片。该脚本生成同目录的 `data_local.yaml`，把 `path` 指向本机绝对路径。

```bash
python scripts/fix_data_yaml.py
```

> 6 类新数据集 `datasets/20260506/data_local.yaml` 已经手工写好，无需运行此脚本。

### `extract_hard_negatives.py`

推理时常见误报来源（红白色锥桶被识别成反光衣 / 远景人物虚检），最有效的办法是把误报帧加入训练集做纯背景负样本（YOLO 把空 `.txt` 视为纯负图）。

```bash
python scripts/extract_hard_negatives.py \
    --alerts runs/ppe_v2/infer_v2/alerts.jsonl \
    --video  'test-video/test-video-1倍焦距切换.mp4' \
    --out    runs/ppe_v2/hard_negatives \
    --min-gap 30 \
    --max-per-alert 200
```

输出：
- `out/images/<stem>_f{frame}_{alert}.png`
- `out/labels/<stem>_f{frame}_{alert}.txt`（空文件）

人工再筛选：删除其中包含真正违规的帧（避免污染负样本），剩下的 image/label 对挪到 `datasets/.../images/train/` 与 `labels/train/` 后重训。

关键参数：
- `--min-gap N`：同一 track + 同一告警类型，相邻导出帧至少间隔 N 帧（去重）。
- `--max-per-alert K`：每种告警 tag 全局最多导出 K 帧。
- `--alert-types no_vest no_helmet`：只导出指定告警类型。

---

## 3. 训练

### `train.py`（旧 4 类）

```bash
python scripts/train.py --model s --imgsz 1280 --epochs 150 --batch -1
```

- `--model {n,s,m}`：YOLOv8 nano/small/medium。
- `--imgsz 1280`：保大图 → 远景小目标更易检出。
- `--batch -1`：AutoBatch。
- 内置增强：`mosaic=1.0, close_mosaic=15, copy_paste=0.3, mixup=0.10, scale=0.5`。

### `train_v2.py`（6 类，推荐）

```bash
# YOLO11m，6 类
python scripts/train_v2.py --arch yolo11m --imgsz 1280 --epochs 150 --batch -1

# RT-DETR-L（建议关掉强增强）
python scripts/train_v2.py --arch rtdetr-l --imgsz 1280 --epochs 100 --batch 8 --no-strong-aug
```

关键参数：
- `--arch`：可选 `yolov8{n,s,m,l,x}` / `yolo11{n,s,m,l,x}` / `rtdetr-{l,x}`。
- `--data`：默认 `datasets/20260506/data_local.yaml`。
- `--device 0` / `0,1` / `cpu`。
- `--cache {ram,disk,False}`：内存大可 ram；磁盘 IO 慢可 disk。
- `--no-strong-aug`：关闭 mosaic/copy_paste/mixup（RT-DETR 推荐关）。
- `--patience 30`：连续 30 epoch 无提升早停。

输出目录：`runs/ppe_v2/<arch>_imgsz<size>/weights/best.pt`。

---

## 4. 验证

### `val.py`

```bash
python scripts/val.py \
    --weights runs/ppe_v2/yolo11m_imgsz1280/weights/best.pt \
    --split val \
    --imgsz 1280
```

- `--split {val,test}`：选评估子集。
- `--conf 0.001` / `--iou 0.6`：评估惯例，低 conf 取全 PR 曲线。
- 终端打印每类 mAP50 与整体 mAP50 / mAP50-95，曲线图存 `runs/ppe_v2/val_<split>/`。

---

## 5. 推理（重点）

### 5.1 `infer_video_v2.py` 典型命令

```bash
python scripts/infer_video_v2.py \
    --arch yolo11m \
    --weights runs/ppe_v2/yolo11m_imgsz1280/weights/best.pt \
    --source 'test-video/test-video-1倍焦距切换.mp4' \
    --imgsz 1280 \
    --half --no-show
```

输出（默认 `runs/ppe/infer_v2/`）：
- `output.mp4`：渲染好告警框的可视化视频（FFmpeg libx264）。
- `alerts.jsonl`：每条告警一行 JSON `{frame, time_sec, track_id, alert, bbox_xyxy}`。
- `alert_snapshots/f{frame:06d}_id{tid}_{tag}.jpg`：告警触发瞬间的整帧快照。

### 5.2 流程内部要点

1. `model.track(...)` → ultralytics 内置 ByteTrack（`bytetrack.yaml`），给每个检测分配 `track_id`。
2. 按类拆框：person / helmet / head / vest /（可选 crane、excavator）。
3. `associate_ppe(person, helmets, heads, vests)`：用「helmet 在 person 框上 0.0–0.4 段」「vest 在 person 框中 0.15–0.75 段」的几何包含判定，返回 `has_helmet / has_vest`。
4. `TrackVoter.update(tid, status)`：在每个 `track_id` 上做窗口投票（默认窗口 15 帧、阈值 10 帧），输出平滑后的 `sm_h / sm_v` + 粘滞告警 tag（`no_helmet` / `no_vest`）。
5. 颜色：绿=合规、红=违规、灰=远景未判定。粘滞机制：触发红框后只有恢复期满才解除（避免帧间闪烁）。
6. 装备类（crane / excavator）只画框 + 中文标签，不进入第 3–5 步。

### 5.3 超参数详解

#### 检测阈值（控制各类「算数」的下限）

| 参数 | 默认 | 含义 / 调参建议 |
|------|-----:|-----------------|
| `--conf` | `0.4` | ultralytics 全局 conf 下限。比这个还低的检测连进入循环都不会。建议保持 0.3–0.45。 |
| `--helmet-conf` | `0.45` | helmet 单类二次过滤。安全帽误报多→调高到 0.5；远景漏检多→调到 0.35。 |
| `--vest-conf` | `0.30` | vest 二次过滤。反光衣常因色块小、强光抖动 conf 不稳，所以默认偏低。 |
| `--head-conf` | `0.30` | head（裸头）二次过滤。head 是 helmet 的反例，低 conf 头部误判会让粘滞红框失控。 |
| `--equip-conf` | `0.40` | crane / excavator 二次过滤。仅影响装备框的可视化，与告警无关。 |
| `--iou` | `0.5` | NMS IoU 阈值，传给 ultralytics。 |

#### 关联（associate_ppe）

| 参数 | 默认 | 含义 |
|------|-----:|------|
| `--contain-thr` | `0.5` | 判定 helmet/vest「属于该 person」的最小包含比例。person 框很紧时下调至 0.35；俯拍人头与背景近时上调到 0.6。 |
| `--vest-negative-min-px` | `80` | person 短边小于此像素时跳过 vest 反例打分（远景反光衣信号弱）。 |
| `--min-person-px` | `24` | person 短边小于此像素直接标灰为「远景」，不参与 PPE 判定，也不入投票。值越小召回越激进、误报也越多。 |

#### 时序投票（TrackVoter）

| 参数 | 默认 | 含义 |
|------|-----:|------|
| `--vote-window` | `15` | 每个 track 维护的滑动窗口长度（帧）。25 fps 下 15 帧≈0.6 s。 |
| `--vote-thresh` | `10` | 窗口内多少帧判违规才触发告警。`10/15 ≈ 67%` 多数表决；提高 → 更稳但延迟增大；降低 → 灵敏但抖动。 |

粘滞规则：一旦该 track 被判 `no_helmet`，`alerted_helmet` 置位，`smoothed_h` 在恢复期前一直保持 False（红框不闪）。`no_vest` 同理。

#### 渲染 / 性能 / IO

| 参数 | 默认 | 含义 |
|------|-----:|------|
| `--arch` | `yolo11s` | 给 v2 选 loader：`yolov*` / `yolo11*` → `YOLO`；`rtdetr*` → `RTDETR`。仅作识别，权重路径用 `--weights`。 |
| `--weights` | 必填 | 训练产物 `best.pt` 路径。 |
| `--source` | 必填 | 视频文件路径或 USB 摄像头编号（数字字符串如 `"0"`）。 |
| `--imgsz` | `1280` | 推理输入尺寸。需与训练 imgsz 一致；想再快可降到 960。 |
| `--device` | `"0"` | CUDA 设备号 / `cpu`。多卡 `0,1`。 |
| `--half` | off | FP16 推理。RTX 5060 Ti 走 Tensor Core，速度 +30%~50%、显存减半，精度损失通常 mAP <0.5%。**只在推理用**。 |
| `--tracker` | `bytetrack.yaml` | ultralytics 跟踪器配置，可换 `botsort.yaml`。 |
| `--max-frames` | `0` | 0 = 处理全片；>0 用于快速 smoke test。 |
| `--no-show` | off | 不弹 OpenCV 预览窗口（无 GUI 的服务器/远程必开）。 |
| `--draw-ppe-boxes / --no-draw-ppe-boxes` | on | 是否画 helmet/head/vest 小框（人物大框始终画）。复杂场景可关闭以减干扰。 |
| `--no-equipment` | off | 关掉 crane/excavator 类的检出与绘制。 |
| `--font-size` | `18` | 中文标签字号（PIL 渲染，使用 Noto CJK）。 |
| `--save-dir` | `runs/ppe/infer_v2` | 输出根目录，包含 `output.mp4`、`alerts.jsonl`、`alert_snapshots/`。 |

### 5.4 调参经验速查

- 误报反光衣多（红白锥桶）→ `--vest-conf 0.4`，并跑 `extract_hard_negatives.py` 回灌负样本。
- 远景虚警多 → 增大 `--min-person-px` 到 32–40。
- 告警闪烁 / 漏报短暂违规 → 调大 `--vote-window` 同时按比例调 `--vote-thresh`（保持 ≈67%）。
- 推理太慢 → `--half`、降 `--imgsz` 到 960、关 `--no-show`。
- 头戴帽但仍判 no_helmet → 多半是 helmet conf 抖动，下调 `--helmet-conf`，或检查 `--contain-thr`。

---

## 6.后续添加内容：
  1. 新建 scripts/augment_offline.py                                                                                                                                                   
   
  离线增强脚本，针对 OOD 测试视频域：                                                                                                                                                  
                                                                                                                                                                                     
  变换内容：
  - 色彩/光照：RandomBrightnessContrast、HueSaturationValue、CLAHE、RandomGamma
  - 模糊/伪影：MotionBlur/GaussianBlur/Defocus（OneOf）、ImageCompression(质量 55–90)、GaussNoise
  - 户外环境：RandomFog/RandomShadow（OneOf，p=0.15）
  - 几何：HorizontalFlip、Affine(scale±15%, translate±5%, rotate±5°, shear±3°)
  - bbox 用 format='yolo'，min_visibility=0.35 自动同步

  筛选逻辑：
  - 只处理 label 里含 person (class 0) 的训练图
  - 每张原图生成 1 张 _aug 文件，写回 images/train/ 和 labels/train/                                                                                                                   
  - 增强后若 person bbox 全部消失则重试（最多 5 次），仍失败就丢弃
  - 默认跳过已存在的 _aug 文件，加 --overwrite 强制重做                                                                                                                                
  - 提供 --dry-run 先看统计                                                                                                                                                          

  2. 修改 scripts/train_v2.py（仅 YOLO 分支强增强参数）

  ┌───────┬───────┬──────┐
  │ 参数  │ 旧值  │ 新值 │
  ├───────┼───────┼──────┤
  │ scale │   0.5 │  0.7 │
  ├───────┼───────┼──────┤
  │ hsv_h │ 0.015 │ 0.02 │
  ├───────┼───────┼──────┤
  │ hsv_s │   0.7 │  0.8 │
  ├───────┼───────┼──────┤
  │ hsv_v │   0.4 │  0.5 │
  └───────┴───────┴──────┘

  更大的 scale 抖动 → 训练时见到更多远近变化，对小 person 召回友好；更大的 HSV 抖动 → 提升对测试视频不同光照/白平衡的鲁棒性。RT-DETR 分支保持不变。

  使用流程

  # 1. 先 dry-run 看会扩多少张
  python scripts/augment_offline.py --dry-run

  # 2. 正式生成
  python scripts/augment_offline.py

  # 3. 用新增强后的训练集重训
  python scripts/train_v2.py --arch yolo11s --imgsz 1280 --epochs 150
  # 或 --arch yolo11m 升级模型容量

  注意事项

  - 增强后训练集图片会接近翻倍（具体看含 person 的图占多少），训练时间随之上升。
  - 跑训练前别忘了清掉 datasets/20260506/labels/*.cache，否则 ultralytics 不会扫到新 _aug 文件：
  rm -f datasets/20260506/labels/train.cache datasets/20260506/labels/val.cache
  - 若觉得效果还不够，可以再跑一次 --overwrite 用不同 --seed 生成新的扰动结果（或者改成 2 张/图：自己复制生成逻辑）。
  - val 集不要做离线增强，会污染评估。脚本只动 images/train / labels/train，是安全的。
