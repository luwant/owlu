# Module 3: Prototype Absorption

> Absorption 负责把 Writer 批准后的标签知识真正写回 LTCE 模型，其中 slow sync 现在已经可以直接消费 Writer 导出的文本样本。

## 模块定位

Prototype Absorption 是闭环的执行端，接收：

- Writer 产出的 `LabelBank`
- Writer 导出的 `LtceTextSample`
- `Label-gen` 的真实 LTCE 运行时

并完成两类同步：

1. `fast_sync`
   - 标签数不变，只刷新旧标签语义表示
2. `slow_sync`
   - 标签数从 `L` 扩到 `L'`
   - 扩展分类头
   - 用新标签文本样本做增量训练

---

## 当前目录

```text
owlu/absorption/
|-- __init__.py
|-- metrics.py
|-- fast_sync.py
|-- slow_sync.py
`-- ltce_bridge.py
```

---

## 1. fast_sync

`fast_sync` / `fast_sync_model` 解决的是“标签没变，但 alias 和语义描述变了”的场景：

- 基于 label alias 文本更新 `label_embeddings`
- 同步刷新 `label_prototypes`
- 结合验证集重新标定阈值

这一条路径已经完成 AAPD alias 注入验证。

---

## 2. slow_sync

### 2.1 解决的问题

当 Writer 产生新的 promoted label 时，LTCE 原模型并不知道这些标签。`slow_sync_model` 负责：

1. 找出 `label_bank.promoted_labels` 中尚未进入模型的新标签
2. 初始化新标签的 embedding / prototype
3. 将分类头从 `L` 维扩到 `L'` 维
4. 在 expanded label space 上做增量训练
5. 重新标定阈值

### 2.2 这次真正补齐的输入

此前 slow sync 已具备“扩维和训练”的代码骨架，但缺少一条稳定的上游样本供给链路。现在这条链路已经固定为：

```text
Discovery -> Writer.ingest_with_document -> SQLite label_examples
         -> Writer.promote -> Writer.export_ltce_samples
         -> build_ltce_incremental_loaders -> slow_sync_model
```

也就是说，slow sync 现在不再依赖外部临时脚本手工拼装新标签样本。

### 2.3 模型契约

`slow_sync_model` 目前对真实 LTCE 模型的要求是：

- `model.label_embeddings: Tensor[L, H]`
- `model.label_prototypes: Tensor[L, H]`
- `model.classifier[-1]: Linear(H, L)`
- `model.num_labels`
- `model.forward(...)`
- 推荐支持 `model.update_prototypes(...)`

同时，训练和验证 loader 必须输出：

```python
{
    "input_ids": Tensor[B, S],
    "attention_mask": Tensor[B, S],
    "token_type_ids": Tensor[B, S] | None,
    "sentence_map": Tensor[B, S] | None,
    "labels": Tensor[B, L'],
}
```

其中最关键的是：`labels.shape[1]` 必须已经扩成 `L'`。如果仍是旧的 `(B, L)`，实现会显式报错，而不是静默错训。

---

## 3. LTCE Bridge

`ltce_bridge.py` 的职责是把 OWLU 与 `Label-gen` 工程接起来。

### 3.1 `load_ltce_artifacts`

负责加载：

- LTCE config
- tokenizer
- dataset builder
- train / validation / test loader
- LTCEModel
- label ids
- 可选 checkpoint

### 3.2 `build_ltce_incremental_loaders`

负责：

1. 根据 `label_bank.promoted_labels` 计算新增标签
2. 将 base label space 从 `L` 扩到 `L'`
3. 对 base 数据集标签向量补零扩维
4. 接收 Writer 导出的 `LtceTextSample`
5. 生成 expanded dataloader

Writer 导出的样本格式是：

```python
LtceTextSample(
    doc_id="demo_0",
    text="graph neural networks for traffic forecasting",
    true_labels={"cs.GNN"},
    split="train",
)
```

---

## 4. 推荐联调用法

```python
from owlu import OntologyWriter, PrototypeAbsorption

writer = OntologyWriter(db_path="owlu_state.db")

# 1. 在线写入 discovery 结果和原始文档
writer.ingest_with_document(result, document_text=doc_text)

# 2. 晋升后直接导出 LTCE 样本
samples = writer.export_ltce_samples(min_positive_examples=2)

# 3. 构造 expanded loader
absorber = PrototypeAbsorption(writer.bank)
runtime = PrototypeAbsorption.load_ltce_artifacts(...)
expanded = absorber.build_ltce_incremental_loaders(
    runtime,
    runtime.label_ids,
    promoted_samples=samples,
)

# 4. 执行 slow sync
result = absorber.slow_absorb_model(
    runtime.model,
    label_ids=list(runtime.label_ids),
    training_loader=expanded.train_loader,
    validation_loader=expanded.validation_loader,
    device=runtime.device,
)
```

---

## 5. 当前验证范围

本地已验证：

- `build_ltce_incremental_loaders` 能扩张标签空间
- `slow_sync_model` 会拒绝维度错误的 loader
- `slow_sync_model` 能在 LTCE 风格 dummy model 上完成扩维与训练
- `test_discovery_writer_real.py` 已实际跑通真实 API + AAPD + Writer 持久化链路

测试结果：

```text
python -m pytest owlu/tests/test_absorption_ltce.py -q
3 passed in 1.88s

python -m pytest owlu/tests -q
39 passed, 1 warning in 65.53s
```

---

## 6. 当前判断

从实现层面看，Absorption 这一段已经不再缺关键接口，尤其是 slow sync 的上游样本供给已固定化。当前剩下的主要不是工程连通性问题，而是实验规模问题：要评估新标签是否真正提升 LTCE 效果，需要持续积累足够多、足够干净的新标签正例文本。
