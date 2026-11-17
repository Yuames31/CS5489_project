# CS5489 AIGC Detection — 提交版（Kaggle public 0.82）

这份代码就是**第一次生成 `submission.csv` 并提交到 Kaggle 拿 0.82 的那一套**。
文件夹里只有 11 个文件 + 1 个模型目录，没有别的东西。

---

## 文件地图（就这些）

| 文件 | 是什么 | 要动它吗 |
|---|---|---|
| `rle.py` | RLE 编解码：**列优先 + 1-based**，以及"只解码窗口"的函数 | 不动 |
| `data.py` | 训练用的数据集：512² 原生分辨率裁片 + 增强 | 不动 |
| `model.py` | 模型：ResNet34-UNet（ASPP + SCSE + 深监督 + 全分辨率细节支路） | 不动 |
| `losses.py` | 损失：BCE(自适应 pos_weight) + Dice + Lovász | 不动 |
| `metrics.py` | IoU / Dice / P / R 统计 | 不动 |
| `inference.py` | 推理：滑窗 + Hann 融合 + 翻转 TTA | 不动 |
| `util.py` | 控制台编码小工具 | 不动 |
| **`train.py`** | **训练入口** | 只想改**文件末尾 `PRESET_ARGS` 一行** |
| **`infer.py`** | **生成提交**（`ID,Predicted`） | 不用改，右键就能跑 |
| `README.md` | 就是这份说明 | — |
| `requirements.txt` | 依赖说明（只有 torch/torchvision/numpy/pandas/Pillow） | — |
| `submission.csv` | 你交到 Kaggle 的那份提交（501 行 = 表头 + 500 张） | 留档，别改 |
| `runs/exp/model.pth` | **产出那份提交的权重**（107 MB） | ⚠️ 千万别删 |
| `runs/exp/model_meta.json` | 推理要用的配置（arch / 归一化 / 阈值 0.7） | 别删 |
| `runs/exp/history.csv` | 30 个 epoch 的训练记录（画曲线用） | 报告要用 |

两份关键文件的对应关系：**`runs/exp/model.pth` + `model_meta.json` → `infer.py` → `submission.csv`**。

---

## 怎么跑（都在 PyCharm 里右键）

1. **生成提交**：右键 `infer.py` → Run。
   自动使用 `runs/exp/model.pth`、阈值 0.7、4 路翻转 TTA，输出 `submission.csv`
   （500 张约 15~25 分钟）。
   > 想再白拿约 0.6 分：改 `infer.py` 末尾
   > `PRESET_ARGS = ["--fallback"]`（救回"整张图都不报"的那些图）。默认是关的，
   > 关着才是当初提交的那份行为。

2. **重新训练**：右键 `train.py` → Run。
   默认配置 = 当初那次训练（crop 512 / batch 8 / 30 epoch / lr 3e-4）。
   > 末尾 `PRESET_ARGS` 里的 `--out-dir` **必须指向新文件夹**，否则会覆盖
   > `runs/exp/model.pth`。

---

## 关键数字（写报告直接引用）

| 项 | 数值 |
|---|---|
| 本地验证集（500 张，逐图平均 IoU，阈值 0.7） | **0.8120** |
| 同一模型的 global IoU / Dice | 0.8553 / 0.8604 |
| Kaggle public | **0.82** |
| 中位 IoU（说明误差是长尾：少数图完全失败） | 0.931 |
| 按 mask 面积分桶 IoU | <1%: 0.69 · 1-5%: 0.78 · 5-20%: 0.86 · 20-50%: 0.92 · >50%: 0.86 |
| 旧版 baseline（`aigc/`，256² 输入） | 0.31 |

---

## 被删掉的东西在哪

删掉的是报告工具和评测输出（`evaluate.py` / `visualize.py` / `plot_history.py` /
`tests/` / `docs/` / `runs/exp/eval/` / `runs/exp/figs_worst/`），
它们全部打包在项目根目录的：

```
Coures Project\_backup_before_cleanup.zip     (2.6 MB, 16 个文件)
```

写报告需要时告诉我，我 1 分钟解开给你。确认不再需要就直接删掉那个 zip。
