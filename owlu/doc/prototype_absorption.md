# Module 3: Prototype Absorption Module

> 把已经被闭环体系批准的新标签，真正吸收到 LTC-MPE 的表示空间里。

## 模块定位

Prototype Absorption 是闭环的**执行终端**。它接收 Writer 晋升的标签信息和 `LabelBank` 的语义数据，将新知识注入 LTCE 模型的 Embedding (E) 和 Prototype (P) 矩阵，并重新标定决策阈值。更新后的模型反馈给 Discovery，完成闭环。

## 目录结构

```
owlu/absorption/
├── __init__.py    # PrototypeAbsorption — Facade 统一入口
├── metrics.py     # 向量数学、评分、推断、阈值校准
├── fast_sync.py   # 语义刷新（标签数不变）
└── slow_sync.py   # 标签扩维 L→L' + 增量微调
```

---

## 1. metrics.py — 共享数学与评估工具

### 向量运算

| 函数 | 签名 | 说明 |
|------|------|------|
| `normalize` | `(vec) -> Vector` | L2 归一化 |
| `cosine_similarity` | `(left, right) -> float` | 余弦相似度 |
| `blend_and_normalize` | `(base, update, eta) -> Vector` | `(1-η)*base + η*update` 后归一化 |
| `mean_vector` | `(vectors) -> Vector` | 多向量取均值 |

### 文本编码器

| 函数 | 签名 | 说明 |
|------|------|------|
| `default_text_encoder` | `(text, dim) -> Vector` | 确定性轻量编码器。基于 SHA-256 哈希的稀疏投影，用于测试和无 BERT 场景的兜底 |

**编码过程**：
```
text → 小写分词 → 对每个 token:
    SHA-256(token) → bucket = hash[:4] % dim
                   → sign = +1/-1 (hash[4] 奇偶)
    vec[bucket] += sign
→ L2 归一化
```

### 评分与推断

| 函数 | 签名 | 说明 |
|------|------|------|
| `score_document` | `(embedding, prototypes: Matrix) -> Vector` | 文档 embedding 与所有 prototype 的 cosine 相似度 |
| `infer_topk` | `(embedding, model_state, top_k=3) -> list[(label_id, score)]` | Top-k 标签推断 |
| `infer_above_threshold` | `(embedding, model_state) -> list[str]` | 高于阈值的所有标签 |

### 阈值校准

| 函数 | 签名 | 说明 |
|------|------|------|
| `recalibrate_threshold` | `(prototypes, label_ids, validation_set, current_threshold) -> float` | Macro-F1 Grid Search |

**校准过程**：
```
grid = [0.10, 0.12, 0.14, ..., 0.90]  (共 41 个候选阈值)

for each threshold in grid:
    for each label:
        计算 TP, FP, FN → per-label F1
    macro_f1 = mean(per-label F1)

best_threshold = argmax(macro_f1)
平局时优先选更接近 current_threshold 的值
```

---

## 2. fast_sync.py — 语义刷新 (标签数不变)

### 函数：`fast_sync`

**触发时机**：Writer 为已有标签添加了新别名（merge 操作）后。

**输入**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `model_state` | `dict` | 包含 `label_ids`, `E`, `P`, `threshold` |
| `label_bank` | `LabelBank` | 含最新别名的标签库 |
| `eta_e` | `float` (0.2) | Embedding 混合率 |
| `eta_p` | `float` (0.1) | Prototype 混合率 |
| `validation_set` | `list[ValidationSample]` | 用于阈值重标定 |
| `text_encoder` | `Callable` | 文本→向量编码器（默认 SHA-256 哈希） |

**算法流程**：

```
for each label_id in model_state["label_ids"]:
    1. texts = label_bank.get_label_aliases(label_id)    # 获取所有别名
    2. alias_vecs = [encoder(t, dim) for t in texts]     # 编码每个别名
    3. alias_mean = normalize(mean_vector(alias_vecs))    # 别名均值
    4. E'[y] = blend(E[y], alias_mean, η_e=0.2)          # 混合 embedding
    5. P'[y] = blend(P[y], E'[y],      η_p=0.1)          # 混合 prototype

threshold' = recalibrate_threshold(P', labels, val_set, old_threshold)
```

**输出**：更新后的 `model_state`，新增 `sync_report`：
```json
{
  "sync_type": "fast",
  "eta_e": 0.2,
  "eta_p": 0.1,
  "old_threshold": 0.44,
  "new_threshold": 0.42,
  "avg_embedding_prototype_alignment": 0.987,
  "num_labels": 54,
  "dim": 128
}
```

---

## 3. slow_sync.py — 标签扩维 L → L'

### 函数：`slow_sync`

**触发时机**：Writer 晋升了新标签（promote 操作）后。

**输入**（额外参数）：
| 参数 | 类型 | 说明 |
|------|------|------|
| `training_samples` | `list[ValidationSample]` | 用于增量微调的训练数据 |
| `new_lr` | `float` (0.05) | 新标签的学习率 |
| `old_lr` | `float` (0.01) | 旧标签的学习率 |

**算法流程**：

```
Step 1 — 发现新标签
    new_ids = [lid for lid in label_bank.promoted_labels if lid not in model_state]
    如果没有新标签 → 回退到 fast_sync

Step 2 — 初始化新标签向量
    for lid in new_ids:
        texts = label_bank.aliases(lid)
        init_vec = normalize(mean(encoder(t) for t in texts))
        E.append(init_vec)
        P.append(init_vec)
        label_ids.append(lid)

Step 3 — 增量微调 (_incremental_finetune)
    for each label:
        pos_vecs = [sample.embedding for sample if label in sample.true_labels]
        neg_vecs = [其余]
        lr = new_lr (新标签) 或 old_lr (旧标签)
        E[y] = blend(E[y], mean(pos_vecs), η=lr)
        P[y] = blend(P[y], E[y],           η=lr*0.5)

Step 4 — 阈值重标定
    threshold' = recalibrate_threshold(P', labels', val_set, old_threshold)
```

**输出**：扩维后的 `model_state`，`sync_report` 含：
```json
{
  "sync_type": "slow",
  "old_num_labels": 54,
  "new_num_labels": 56,
  "added_labels": ["cs.NLP", "stat.DL"],
  "old_threshold": 0.44,
  "new_threshold": 0.40,
  "avg_embedding_prototype_alignment": 0.971,
  "dim": 128
}
```

**回退行为**：当 `label_bank.promoted_labels` 中没有尚未加入模型的新标签时，自动回退到 `fast_sync`。

---

## 4. Facade：`PrototypeAbsorption`

**位置**：`absorption/__init__.py`

```python
class PrototypeAbsorption:
    def __init__(self, label_bank: LabelBank, text_encoder=None)

    def fast_absorb(model_state, *, eta_e=0.2, eta_p=0.1,
                    validation_set=None) -> dict
        # 语义刷新，标签数不变

    def slow_absorb(model_state, *, validation_set=None,
                    training_samples=None, new_lr=0.05,
                    old_lr=0.01, eta_e=0.2, eta_p=0.1) -> dict
        # 标签扩维 + 增量微调
```

### 典型调用

```python
from owlu.absorption import PrototypeAbsorption
from owlu.writer.label_bank import LabelBank

label_bank = writer.bank  # 从 OntologyWriter 获取

absorber = PrototypeAbsorption(label_bank)

# 场景 A: Writer 仅添加了别名 → fast absorb
updated = absorber.fast_absorb(model_state, validation_set=val_samples)

# 场景 B: Writer 晋升了新标签 → slow absorb
updated = absorber.slow_absorb(
    model_state,
    validation_set=val_samples,
    training_samples=train_samples,
)

# 更新后的模型状态反馈给 Discovery
# updated["E"], updated["P"], updated["threshold"]
```

---

## 输入 / 输出契约

### 输入

| 来源 | 数据 | 说明 |
|------|------|------|
| Writer | `LabelBank` 实例 | 含别名、描述、promoted_labels |
| LTCE 模型 | `model_state` dict | `{label_ids, E, P, threshold}` |
| 训练数据 | `list[ValidationSample]` | slow_sync 增量微调用 |
| 验证数据 | `list[ValidationSample]` | 阈值重标定用 |

### 输出

| 消费方 | 数据 | 说明 |
|--------|------|------|
| Discovery (Gate) | 更新后的 `model_state` | 新的 E/P/threshold，影响下一轮门控判定 |
| 审计日志 | `sync_report` | 嵌套在 model_state 中的同步报告 |

---

## model_state 字段参考

```python
model_state = {
    "label_ids":          list[str],      # 标签 ID 列表
    "E":                  Matrix,         # [num_labels × dim] Embedding 矩阵
    "P":                  Matrix,         # [num_labels × dim] Prototype 矩阵
    "threshold":          float,          # 决策阈值
    "label_aliases":      dict[str, list[str]],  # (fast_sync 输出)
    "label_descriptions": dict[str, str],         # (fast_sync 输出)
    "sync_report":        dict,                   # 同步审计报告
}
```
