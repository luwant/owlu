# Module 3: Prototype Absorption Module

> 把已经被闭环体系批准的新标签，真正吸收到 LTC-MPE 的表示空间里。

## 模块定位

Prototype Absorption 是闭环的**执行终端**。它接收 Writer 晋升的标签信息和 `LabelBank` 的语义数据，将新知识注入 LTCE 模型的 Embedding (E) 和 Prototype (P) 矩阵，并重新标定决策阈值。更新后的模型反馈给 Discovery，完成闭环。

## 目录结构

```
owlu/absorption/
├── __init__.py    # PrototypeAbsorption — Facade 统一入口
├── metrics.py     # 向量数学、评分、推断、阈值校准（纯 Python + Torch 双层）
├── fast_sync.py   # 语义刷新（标签数不变）— dict 模式 + model 模式
└── slow_sync.py   # 标签扩维 L→L' + 增量微调（纯 dict 模式，暂未对接 LTCE 模型）
```

---

## 1. metrics.py — 共享数学与评估工具

包含两层实现：
1. **纯 Python (list-based)** — 用于单元测试和轻量使用
2. **Torch 加速** — 用于真实 LTCEModel 注册缓冲区的操作

### 向量运算（纯 Python）

| 函数 | 签名 | 说明 |
|------|------|------|
| `normalize` | `(vec) -> Vector` | L2 归一化 |
| `cosine_similarity` | `(left, right) -> float` | 余弦相似度 |
| `blend_and_normalize` | `(base, update, eta) -> Vector` | `(1-η)*base + η*update` 后归一化 |
| `mean_vector` | `(vectors) -> Vector` | 多向量取均值 |

### Torch 加速工具

| 函数 | 签名 | 说明 |
|------|------|------|
| `blend_and_normalize_torch` | `(base: Tensor, update: Tensor, eta) -> Tensor` | Torch 原生混合 + L2 归一化，支持 1-D / 2-D |
| `recalibrate_model_threshold` | `(model, dataloader, device, current_threshold) -> float` | 基于模型前向传播的 Macro-F1 Grid Search 阈值校准（sigmoid 空间） |

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

### 评分与推断（纯 Python）

| 函数 | 签名 | 说明 |
|------|------|------|
| `score_document` | `(embedding, prototypes: Matrix) -> Vector` | 文档 embedding 与所有 prototype 的 cosine 相似度 |
| `infer_topk` | `(embedding, model_state, top_k=3) -> list[(label_id, score)]` | Top-k 标签推断 |
| `infer_above_threshold` | `(embedding, model_state) -> list[str]` | 高于阈值的所有标签 |

### 阈值校准

**纯 Python 版 `recalibrate_threshold`** — 基于 cosine 相似度的 Macro-F1 Grid Search（给 dict 模式用）:

```
grid = [0.10, 0.12, 0.14, ..., 0.90]  (共 41 个候选阈值)

for each threshold in grid:
    for each label:
        计算 TP, FP, FN → per-label F1
    macro_f1 = mean(per-label F1)

best_threshold = argmax(macro_f1)
平局时优先选更接近 current_threshold 的值
```

**Torch 版 `recalibrate_model_threshold`** — 基于 `sigmoid(logits)` 的 Macro-F1 Grid Search（给 model 模式用）:

```
1. model.eval()
2. 对验证集执行前向传播，收集所有 logits 和 labels
3. probs = sigmoid(logits)
4. grid search 同上，但用 probs >= threshold 作为预测
```

---

## 2. fast_sync.py — 语义刷新 (标签数不变)

提供两个入口点：

### 2.1 `fast_sync` — dict 模式（纯 Python）

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

**输出**：更新后的 `model_state`，新增 `sync_report`。

### 2.2 `fast_sync_model` — model 模式（Torch 原生，已验证 ✅）

**触发时机**：同上，但直接操作 LTCEModel 注册缓冲区。

**输入**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `LTCEModel` | 已加载的 LTCE 模型（含 `label_embeddings` / `label_prototypes`） |
| `label_bank` | `LabelBank` | 含最新别名的标签库 |
| `label_ids` | `list[str]` | 有序标签名列表，对应 E/P 矩阵行 |
| `eta_e` | `float` (0.2) | Embedding 混合率 |
| `eta_p` | `float` (0.1) | Prototype 混合率 |
| `validation_loader` | `DataLoader` | 用于 sigmoid 空间的阈值重标定 |
| `current_threshold` | `float` (0.45) | 当前决策阈值 |
| `text_encoder` | `Callable` | 文本→向量编码器（推荐 `BertEncoder.as_text_encoder()`） |
| `device` | `torch.device` | 从模型参数推断（可选覆盖） |

**算法流程**：

```
with torch.no_grad():
    for idx, label_id in enumerate(label_ids):
        1. texts = label_bank.get_label_aliases(label_id)
        2. alias_vecs → torch tensor, mean → L2 normalize
        3. E[idx] = blend_and_normalize_torch(E[idx], alias_mean, η_e)
        4. P[idx] = blend_and_normalize_torch(P[idx], E[idx],     η_p)

# 阈值重标定（sigmoid 空间，使用 model forward pass）
if validation_loader:
    threshold' = recalibrate_model_threshold(model, loader, device, old_threshold)
```

**输出**：`dict` 含 `threshold`, `label_aliases`, `sync_report`。E/P 已原地更新。

**sync_report 示例**：
```json
{
  "sync_type": "fast",
  "eta_e": 0.2,
  "eta_p": 0.1,
  "old_threshold": 0.45,
  "new_threshold": 0.72,
  "avg_embedding_prototype_alignment": 0.0996,
  "num_labels": 54,
  "dim": 768
}
```

### 实验验证结果（AAPD, seed=22, best.pt）

在 10 个标签上注入 2-4 个语义别名，使用 `BertEncoder` 编码：

| 指标 | Baseline@0.45 | E/P 后@0.45 | E/P + 阈值重标定@0.72 |
|------|:---:|:---:|:---:|
| **Macro-F1** | 0.5843 | 0.5443 (-0.040) | **0.5953 (+0.011)** |
| **Micro-F1** | 0.7424 | 0.6768 (-0.066) | 0.7371 (-0.005) |
| **P@1** | 0.858 | — | 0.851 (-0.007) |

**关键结论**：
1. E/P 混合会改变 logit 分布，**必须配合阈值重标定**（传入 `validation_loader`）
2. 重标定后 Macro-F1 提升 +1.1%（尾部标签受益于别名语义），Micro-F1 仅损失 -0.5%
3. 向量漂移极小：E cosine ≈ 0.9999，P cosine ≈ 0.994

---

## 3. slow_sync.py — 标签扩维 L → L'（暂未对接 LTCE 模型）

> *当前仅纯 Python dict 模式实现。LTCE 增量训练需要的 MLP classifier 层扩展和 loss 反传尚未实现。*

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

    # --- dict 模式 (纯 Python) ---
    def fast_absorb(model_state, *, eta_e=0.2, eta_p=0.1,
                    validation_set=None) -> dict
        # 语义刷新，标签数不变

    def slow_absorb(model_state, *, validation_set=None,
                    training_samples=None, new_lr=0.05,
                    old_lr=0.01, eta_e=0.2, eta_p=0.1) -> dict
        # 标签扩维 + 增量微调

    # --- model 模式 (Torch，直接操作 LTCEModel) ---
    def fast_absorb_model(model, label_ids, *, eta_e=0.2, eta_p=0.1,
                          validation_loader=None, current_threshold=0.45,
                          device=None) -> dict
        # 原地更新 model.label_embeddings / label_prototypes
```

### 典型调用（model 模式 — 推荐用于真实 LTCE 模型）

```python
from owlu.absorption import PrototypeAbsorption
from owlu.common.encoder import BertEncoder
from owlu.writer.label_bank import LabelBank

label_bank = writer.bank  # 从 OntologyWriter 获取
encoder = BertEncoder(model_path="bert/bert-base-uncased", device="cuda")

absorber = PrototypeAbsorption(label_bank, text_encoder=encoder.as_text_encoder())

# Writer 添加了别名 → fast absorb (model 模式)
result = absorber.fast_absorb_model(
    model,
    label_ids=label_names,       # 有序标签名列表
    validation_loader=val_loader, # 必须传，用于阈值重标定
    current_threshold=0.45,
)

new_threshold = result["threshold"]
# model.label_embeddings 和 model.label_prototypes 已原地更新
```

### 典型调用（dict 模式 — 用于测试或无 torch 场景）

```python
absorber = PrototypeAbsorption(label_bank)

# 场景 A: Writer 仅添加了别名 → fast absorb
updated = absorber.fast_absorb(model_state, validation_set=val_samples)

# 场景 B: Writer 晋升了新标签 → slow absorb
updated = absorber.slow_absorb(
    model_state,
    validation_set=val_samples,
    training_samples=train_samples,
)

# updated["E"], updated["P"], updated["threshold"]
```

---

## 输入 / 输出契约

### 输入

| 来源 | 数据 | 说明 |
|------|------|------|
| Writer | `LabelBank` 实例 | 含别名、描述、promoted_labels |
| LTCE 模型 | `LTCEModel` 实例（model 模式） | `label_embeddings` / `label_prototypes` 注册缓冲区 |
| LTCE 模型 | `model_state` dict（dict 模式） | `{label_ids, E, P, threshold}` |
| 验证数据 | `DataLoader`（model 模式）或 `list[ValidationSample]`（dict 模式） | 阈值重标定用 |
| 训练数据 | `list[ValidationSample]` | slow_sync 增量微调用 |

### 输出

| 消费方 | 数据 | 说明 |
|--------|------|------|
| Discovery (Gate) | 更新后的模型（model 模式原地更新）或 `model_state`（dict 模式） | 新的 E/P/threshold，影响下一轮门控判定 |
| 审计日志 | `sync_report` | 嵌套在返回字典中的同步报告 |

---

## model_state 字段参考（dict 模式）

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

## LTCE 模型字段参考（model 模式）

| 属性 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `label_embeddings` | 注册缓冲区 | `(num_labels, 768)` | 标签嵌入 E，原地更新 |
| `label_prototypes` | 注册缓冲区 | `(num_labels, 768)` | 标签原型 P，原地更新 |
| 决策阈值 | `float` | 标量 | 由 `fast_absorb_model` 返回的 `result["threshold"]` |
