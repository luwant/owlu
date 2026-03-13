# Module 3: Prototype Absorption Module

> 把已经被闭环体系批准的新标签，真正吸收到 LTCE 表示空间和分类头里。

## 模块定位

Prototype Absorption 是闭环的执行终端。它接收 Writer 晋升后的 `LabelBank`，对 LTCE 模型做两类更新：

1. `fast_sync`：标签数不变，只刷新已有标签的语义表示。
2. `slow_sync`：标签数从 `L` 扩到 `L'`，同时扩展 classifier 输出维度并做增量微调。

更新后的 `label_embeddings`、`label_prototypes`、`classifier[-1]` 和决策阈值会反馈给 Discovery，形成闭环。

## 目录结构

```text
owlu/absorption/
├── __init__.py      # PrototypeAbsorption facade
├── metrics.py       # 向量运算、推断、阈值重标定
├── fast_sync.py     # fast_sync / fast_sync_model
├── slow_sync.py     # slow_sync / slow_sync_model
└── ltce_bridge.py   # 真实 LTCE 运行时加载与 expanded loader 构造
```

---

## 1. metrics.py

`metrics.py` 同时服务 dict 模式和 model 模式：

- 纯 Python 工具：
  `normalize`、`cosine_similarity`、`blend_and_normalize`、`mean_vector`
- 推断与校准：
  `score_document`、`infer_topk`、`infer_above_threshold`、`recalibrate_threshold`
- Torch 工具：
  `blend_and_normalize_torch`、`recalibrate_model_threshold`
- 兜底文本编码器：
  `default_text_encoder(text, dim)`

`recalibrate_model_threshold` 会直接跑 LTCE `forward()`，基于 `sigmoid(logits)` 做 Macro-F1 grid search。

---

## 2. fast_sync.py

### 2.1 `fast_sync`

dict 模式，输入是：

- `model_state = {label_ids, E, P, threshold}`
- `label_bank`
- `validation_set`
- `text_encoder`

行为：

1. 对每个已有标签收集别名。
2. 用别名均值向量混合 `E[y]`。
3. 再用新的 `E[y]` 混合 `P[y]`。
4. 用验证集重新搜索阈值。

### 2.2 `fast_sync_model`

Torch 原生模式，直接原地修改：

- `model.label_embeddings`
- `model.label_prototypes`

如果传入 `validation_loader`，会用真实 LTCE 前向结果做阈值重标定。

这个路径已经在 AAPD alias 注入实验中验证过。

---

## 3. slow_sync.py

## 3.1 `slow_sync`

dict 模式原型实现，适合无 Torch 或单元测试场景。

行为：

1. 发现 `label_bank.promoted_labels` 中尚未进入模型的新标签。
2. 用标签别名文本初始化新标签向量。
3. 扩展 `label_ids / E / P`。
4. 用 `training_samples` 做一次轻量更新。
5. 用验证集重标定阈值。

若没有新标签，会自动回退到 `fast_sync`。

## 3.2 `slow_sync_model`

这条路径现在已经对接真实 LTCE 模型，前提是模型满足 `Label-gen/src/ltce/models/ltce.py` 的真实契约。

### 真实模型契约

`slow_sync_model` 假定模型具备：

- `model.label_embeddings: Tensor[L, H]`
- `model.label_prototypes: Tensor[L, H]`
- `model.classifier[-1]: Linear(H, L)`
- `model.num_labels`
- `model.forward(input_ids, attention_mask, token_type_ids, sentence_map, labels)`
- `model.update_prototypes(label_representations, labels)` 可选但推荐

这些字段与 `Label-gen/src/ltce/models/ltce.py` 中的 `LTCEModel` 一致。

### 核心行为

`slow_sync_model` 的流程是：

1. 找出新标签 `new_label_ids`
2. 用标签别名文本初始化新标签语义向量
3. 扩容：
   - `label_embeddings: (L, H) -> (L', H)`
   - `label_prototypes: (L, H) -> (L', H)`
   - `classifier[-1]: Linear(H, L) -> Linear(H, L')`
   - `num_labels: L -> L'`
4. 新标签输出行优先用语义向量初始化，而不是纯随机 Xavier
5. 旧标签先执行一次 alias 语义混合
6. 若提供 `training_loader`，按 LTCE 真实 batch 契约做增量微调
7. 若提供 `validation_loader`，在 expanded label space 上重标定阈值

### 增量微调策略

相比旧版草稿实现，当前 model 模式增加了这些真实训练细节：

- classifier 新行使用 row-wise 学习率缩放
- 可调用 `model.update_prototypes(...)` 做 prototype EMA
- 对旧标签 classifier 行加入 anchor regularization，抑制灾难性遗忘
- 严格校验 `training_loader` / `validation_loader` 的 `labels.shape[1] == L'`

### `training_loader` / `validation_loader` 的要求

它们必须产出与 LTCE 原训练一致的 batch：

```python
{
    "input_ids": Tensor[B, S],
    "attention_mask": Tensor[B, S],
    "token_type_ids": Tensor[B, S] | None,
    "sentence_map": Tensor[B, S] | None,
    "labels": Tensor[B, L'],
}
```

这里最关键的是 `labels` 的列数必须已经扩成新标签空间 `L'`。

如果只有旧的 `(B, L)` 标签矩阵，`slow_sync_model` 会直接报错，而不会静默训练错误维度。

### 注意

`slow_sync_model` 只负责“吸收”与“增量训练”。

它并不会自动凭空生成新标签正例。要真实训练新标签，你仍然需要：

- 新标签的正例文本样本，或
- 对已有 LTCE 训练/验证文档的新增标签映射

否则它只能完成“扩维 + 语义初始化”，不能学到可靠的新标签分类边界。

---

## 4. ltce_bridge.py

`ltce_bridge.py` 用来把 OWLU 和 `Label-gen` 下的真实 LTCE 工程接起来。

## 4.1 `load_ltce_artifacts`

```python
load_ltce_artifacts(
    config_path,
    checkpoint_path=None,
    label_gen_root=None,
    device=None,
    num_workers=0,
) -> LtceArtifacts
```

功能：

1. 从 `Label-gen` 动态导入真实 LTCE 模块
2. 读取 `configs/*.yaml`
3. 把相对路径改写为 `Label-gen` 根目录下的绝对路径
4. 构造：
   - `LtceDatasetBuilder`
   - tokenizer
   - train / validation / test dataloader
   - `LTCEModel`
5. 加载 label embeddings
6. 可选加载 checkpoint

返回的 `LtceArtifacts` 包含：

- `config`
- `dataset_builder`
- `tokenizer`
- `collator`
- `model`
- `device`
- `label_ids`
- `train_loader`
- `validation_loader`
- `test_loader`

## 4.2 `build_ltce_incremental_loaders`

```python
build_ltce_incremental_loaders(
    runtime,
    label_bank,
    label_ids,
    promoted_samples=None,
    train_doc_label_updates=None,
    validation_doc_label_updates=None,
    test_doc_label_updates=None,
    include_base_train=True,
    include_base_validation=True,
    include_base_test=False,
    num_workers=0,
) -> LtceIncrementalLoaders
```

功能：

1. 根据 `label_bank.promoted_labels` 计算新标签集合
2. 把标签空间从 `L` 扩成 `L'`
3. 对原 LTCE train / val / test 数据集的标签向量补零列
4. 可选把某些旧文档补充上新标签
5. 可选追加 `LtceTextSample` 形式的新标签文本样本
6. 生成新的 expanded dataloader

### `LtceTextSample`

```python
LtceTextSample(
    doc_id: str,
    text: str,
    true_labels: set[str],
    split: Literal["train", "val", "test"] = "train",
)
```

这个结构用于把新标签样本显式注入真实 LTCE 训练管线。

---

## 5. Facade: PrototypeAbsorption

`PrototypeAbsorption` 现在包含两类入口。

### 5.1 吸收入口

```python
absorber.fast_absorb(...)
absorber.slow_absorb(...)
absorber.fast_absorb_model(...)
absorber.slow_absorb_model(...)
```

其中 `slow_absorb_model(...)` 已经透传：

- `new_lr`
- `old_lr`
- `finetune_epochs`
- `update_prototypes`
- `classifier_anchor_weight`

### 5.2 LTCE bridge 入口

```python
PrototypeAbsorption.load_ltce_artifacts(...)
absorber.build_ltce_incremental_loaders(...)
```

这样可以保持使用方式统一：

1. 先用 facade 加载真实 LTCE 运行时
2. 再用 facade 基于 `LabelBank` 构造 expanded loader
3. 最后直接调用 `slow_absorb_model(...)`

---

## 6. 推荐真实对接流程

### 6.1 加载 LTCE

```python
from owlu import LabelBank, PrototypeAbsorption

bank = LabelBank()
absorber = PrototypeAbsorption(bank)

runtime = PrototypeAbsorption.load_ltce_artifacts(
    config_path="configs/ltce_aapd_full.yaml",
    checkpoint_path="outputs/aapd_ablation/aapd_full_seed22/best.pt",
    label_gen_root="e:/lwt/workspace/Label-gen",
    device="cuda",
)
```

### 6.2 注入新标签样本并构造 expanded loader

```python
from owlu import LtceTextSample

bank.register_label("cs.NLP", "computational linguistics", aliases=["natural language processing"])
bank.promoted_labels["cs.NLP"] = object()

expanded = absorber.build_ltce_incremental_loaders(
    runtime,
    runtime.label_ids,
    promoted_samples=[
        LtceTextSample(
            doc_id="demo_0",
            text="this paper studies natural language processing",
            true_labels={"cs.NLP"},
            split="train",
        ),
    ],
)
```

### 6.3 slow sync 到真实 LTCE

```python
base_label_ids = list(runtime.label_ids)

result = absorber.slow_absorb_model(
    runtime.model,
    label_ids=base_label_ids,
    training_loader=expanded.train_loader,
    validation_loader=expanded.validation_loader,
    current_threshold=0.45,
    new_lr=5e-5,
    old_lr=1e-5,
    finetune_epochs=3,
    update_prototypes=True,
    classifier_anchor_weight=0.05,
    device=runtime.device,
)
```

这里要先复制一份原始 `label_ids`，因为 `slow_absorb_model` 会原地 append 新标签名。

---

## 7. 输入 / 输出契约

### 输入

| 来源 | 数据 | 说明 |
|------|------|------|
| Writer | `LabelBank` | 含 aliases / descriptions / promoted_labels |
| LTCE | `LTCEModel` | 真实模型实例 |
| LTCE | `LtceArtifacts` | 真实 config + tokenizer + dataloaders + model |
| LTCE | `LtceTextSample` | 新标签训练或验证文本 |
| LTCE | expanded `DataLoader` | `labels.shape[1] == L'` |
| Dict mode | `model_state` | `{label_ids, E, P, threshold}` |

### 输出

| 输出 | 说明 |
|------|------|
| `threshold` | 新阈值 |
| `added_labels` | 新增标签 ID |
| `label_ids` | 更新后的有序标签列表 |
| `sync_report` | 审计信息 |
| `model` | model 模式下原地更新后的 LTCE 模型 |

---

## 8. 当前验证范围

目前验证到的范围是：

- `fast_sync_model`：AAPD alias 注入实验已验证
- `slow_sync_model`：已完成真实 LTCE 契约对接
- `ltce_bridge.py`：已能从 `Label-gen` 加载 AAPD config / checkpoint / dataloader
- `build_ltce_incremental_loaders`：已能把标签维度从 `54` 扩到 `55` 并生成带新标签列的 batch
- 单元测试：覆盖 bridge 扩维与 slow sync 关键路径

尚未在仓库内固化的是完整 AAPD slow-sync 指标实验脚本与最终指标表。
