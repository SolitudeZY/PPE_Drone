# PPE Drone —— 无人机航拍工地 PPE 检测

基于无人机航拍视频的施工现场个人防护装备（PPE）检测系统。识别目标包括：施工人员、安全帽、反光背心、以及场景中的吊机、挖掘机等大型机械。

当前阶段定位：**模型侧迭代 + 推理链路稳定化**。数据集已从最初的 `wrj_ppe` 扩充到 `datasets/20260506`（6 类），训练/验证/推理三段式脚本均已成型，正在针对实拍视频的 OOD（分布外）问题做数据增强与训练策略调整。

---

## 1. 目录结构

```
ppe_drone/
├── scripts/                    # 所有训练 / 验证 / 推理 / 数据处理脚本
│   ├── train.py                # 旧版 4 类训练入口
│   ├── train_v2.py             # 当前主训练入口（6 类，支持 YOLO / RT-DETR 切换）
│   ├── val.py                  # 独立验证脚本
│   ├── infer_video.py          # 旧版视频推理
│   ├── infer_video_v2.py       # 当前主推理入口（含 TrackVoter 时序投票）
│   ├── augment_offline.py      # 离线数据增强（albumentations，仅对含 person 的图）
│   ├── prepare_4class.py       # 6 类 → 4 类标签转换
│   ├── fix_data_yaml.py        # data.yaml 路径修复
│   ├── extract_hard_negatives.py  # 难负样本挖掘
│   ├── utils_ppe.py            # 推理侧通用工具（关联、投票、渲染）
│   ├── requirements.txt
│   └── README.md               # 脚本详细说明与超参手册
├── datasets/
│   ├── wrj_ppe/                # 初版数据集
│   └── 20260506/               # 当前主数据集（6 类）
│       ├── images/{train,val,test}
│       ├── labels/{train,val,test}
│       ├── labels_4class/      # 转成 4 类后的标签
│       ├── data.yaml           # 相对路径版
│       ├── data_local.yaml     # 绝对路径版（训练用）
│       └── data_4class.yaml
├── runs/                       # Ultralytics 训练/验证/推理输出
│   ├── ppe/                    # 旧版 4 类结果
│   └── ppe_v2/                 # 当前 6 类结果（yolo11n/s、rtdetr-l 等多组对照）
├── test-video/                 # 实拍测试视频（不入库）
├── other/                      # 临时素材
├── *.pt                        # Ultralytics 预训练权重（yolo11n/s/m、rtdetr-l 等，不入库）
└── runs_train.log              # 早期训练日志
```

---

## 2. 数据集现状

当前主数据集：`datasets/20260506/`

- **类别数**：6
- **类别顺序**：`person, helmet, vest, head, crane, excavator`（id 0–5）
- **划分**：train / val / test，标签为 YOLO 归一化格式（xc yc w h）
- **与 wrj_ppe 对比**：在原有基础上新增了部分航拍图像，小目标 person 数量从 228 → 497（小/中/大分箱）

配置文件：
- `data_local.yaml`：绝对路径，训练脚本默认读取
- `data.yaml`：相对路径版本
- `data_4class.yaml`：配合 `labels_4class/` 使用，映射为 `person / helmet / vest / head` 4 类

---

## 3. 工作流程

```
┌────────────┐   ┌──────────────┐   ┌────────────┐   ┌────────────┐
│  数据集整理 │ → │ 离线增强(可选) │ → │   训练     │ → │   验证     │
└────────────┘   └──────────────┘   └────────────┘   └────────────┘
                                                           │
                                                           ▼
                                                    ┌────────────┐
                                                    │  视频推理   │
                                                    └────────────┘
```

典型一次迭代：

```bash
# （可选）离线增强：仅对含 person 的图做 2x 扩充
python scripts/augment_offline.py --dry-run        # 先看会扩多少张
python scripts/augment_offline.py                  # 实际落盘 _aug.{jpg,txt}
rm -f datasets/20260506/labels/train.cache         # 清缓存，让 ultralytics 重新扫描

# 训练
python scripts/train_v2.py --arch yolo11s --imgsz 1280 --epochs 150

# 验证
python scripts/val.py

# 视频推理
python scripts/infer_video_v2.py \
  --weights runs/ppe_v2/yolo11s_imgsz1280/weights/best.pt \
  --source  test-video/test-video-1倍焦距切换.mp4
```

脚本级的参数说明、超参手册（尤其是推理侧的阈值、关联、时序投票配置）集中放在 `scripts/README.md`。

---

## 4. 当前进展（截至 2026-05-09）

### 已完成
- 6 类数据集切换、训练/验证/推理链路打通
- `train_v2.py` 支持 YOLO 系列与 RT-DETR 一键切换，同时调优了 YOLO 分支的强增强配置（scale 0.7、hsv_s 0.8、hsv_v 0.5 等）
- `infer_video_v2.py` 引入 TrackVoter 时序投票与多阈值控制（person / equip / 粘性告警）
- 离线增强脚本 `augment_offline.py`：基于 albumentations，仅处理含 person 的标签，2x 扩充
- 多组对照训练：yolo11n / yolo11s / rtdetr-l，分别在 1280 / 1600 / 1920 输入尺度上跑过

### 进行中 / 待办
- **OOD 问题排查**：实拍视频上远景灰框（person 小目标）置信度被压到 0.4 阈值以下。结论是分布外问题，非类别错位或分布稀释（val person mAP50=0.97）
  - 短期缓解：推理时 `--conf` 临时下调到 0.15–0.25
  - 中期方案：离线增强 + 训练时强增强（已落盘，待重训验证）
  - 长期方案：将测试视频抽帧纳入训练集
- `infer_video_v2.py` 存在 `--equip-conf` 与 `--no-equipment` argparse 重复声明的 bug（lines 95–99 vs 100–104），**待人工修复**
- 模型升级评估：yolo11s → yolo11m 对小目标召回的收益

---

## 5. 环境 / 依赖

- Python 3.10+
- 主要依赖见 `scripts/requirements.txt`：ultralytics、albumentations、opencv-python 等
- 建议使用独立虚拟环境，训练需要 CUDA GPU

```bash
pip install -r scripts/requirements.txt
```

---

## 6. 备注

- `runs/`、`datasets/`、`test-video/`、`*.pt` 等大文件 / 产物目录均已在 `.gitignore` 中忽略
- 脚本级的详细文档（参数表、调参经验）见 `scripts/README.md`
