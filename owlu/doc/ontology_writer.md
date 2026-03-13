# Module 2: Ontology-Constrained Writer

> Writer 是 OWLU 闭环中的决策和沉淀层：它不仅决定一个候选短语是否晋升为新标签，还要把后续 slow sync 需要的文档监督样本保存下来。

## 模块定位

Writer 接收 Discovery 输出的 `MatchResult`，完成以下工作：

1. 维护跨文档候选簇
2. 做四门控晋升判断
3. 在晋升前执行本体约束校验
4. 持久化标签、聚类、文档和样本证据
5. 导出可直接用于 LTCE slow sync 的 `LtceTextSample`

当前目录结构：

```text
owlu/writer/
|-- __init__.py      # OntologyWriter facade
|-- label_bank.py    # LabelBank: 聚类、状态流转、晋升
|-- constraints.py   # OntologyConstraintChecker: 约束校验
`-- persistence.py   # LabelBankStore: SQLite 持久化与样本导出
```

---

## 1. LabelBank

### 1.1 维护的状态

```text
labels               -> 已注册标签元数据
proto_label_clusters -> 所有未正式写回 ontology 的簇
hold_pool            -> 证据不足，继续积累
candidate_labels     -> 满足四门控，可晋升
promoted_labels      -> 已晋升的新标签
```

### 1.2 这次新增的聚类能力

此前聚类更接近“按归一化短语精确合并”。现在 `LabelBank` 支持可选的语义聚类：

- `dense_encoder`
  - 外部注入短语编码函数
- `cluster_merge_threshold`
  - 新短语与已有簇质心的最小相似度阈值
- `cluster_merge_margin`
  - 第一名和第二名相似度差的安全边际
- `centroid_embedding`
  - 保存在 `ProtoLabelCluster` 中的簇质心

新短语进入时，Writer 会：

1. 归一化短语
2. 编码为 dense embedding
3. 与已有簇质心做 cosine similarity
4. 若最佳簇满足阈值且领先幅度足够，则并入该簇
5. 否则分配新的 `cluster_xxxxxx`

这解决了下列情况会被拆成多个簇的问题：

- `graph neural network`
- `graph neural networks`
- `graph-based neural network`

### 1.3 代表短语选择

簇内 `representative_phrase` 不再只看出现频次，也会兼顾与当前簇质心的贴合度：

```python
score = phrase_count + cosine(phrase_embedding, centroid_embedding)
```

这样在多个表面变体频次相近时，代表短语更稳定。

### 1.4 四门控晋升

候选簇从 `hold` 升到 `candidate` 需要同时满足：

| 条件 | 参数 | 默认值 |
|------|------|------|
| 频次足够 | `min_freq` | 3 |
| 来源文档足够分散 | `min_source_docs` | 2 |
| 多次采样一致性足够 | `min_agreement` | 0.5 |
| 与最近现有标签足够远 | `min_semantic_distance` | 0.3 |

伪代码：

```python
def should_promote(cluster) -> bool:
    return (
        cluster.freq >= min_freq
        and cluster.source_doc_count >= min_source_docs
        and cluster.agreement >= min_agreement
        and (cluster.nearest_label_distance or 1.0) >= min_semantic_distance
    )
```

---

## 2. MatchResult 路由

`process_match_result(result)` 的行为如下：

| Discovery 输出 | Writer 动作 | 返回值 |
|------|------|------|
| `merge_pre` + `target_label` | 为已有标签补 alias | `merge` |
| `novel_pre` | 聚类并尝试进入 candidate | `candidate` 或 `hold` |
| `hold_pre` | 聚类但固定放入 hold | `hold` |
| 其他 | 丢弃 | `discard` |

注意点：

- 每次处理前会重置 `phrase.cluster_id`
- 如果后续进入簇，`cluster_id` 会被回填到 `CandidatePhrase`
- 这个回填值会在文档证据落库时继续使用

---

## 3. Ontology 约束校验

`OntologyConstraintChecker` 在晋升阶段校验：

- `naming_format`
  - 新标签 id 是否符合命名规则
- `parent_existence`
  - 父类是否存在
- `duplicate`
  - 是否与现有标签重复

AAPD 模式示例：

```python
checker = OntologyConstraintChecker.for_aapd()
```

---

## 4. SQLite 持久化

`LabelBankStore` 现在不仅保存 LabelBank 状态，也保存 slow sync 训练样本。

### 4.1 保存的实体

| 表 | 内容 |
|------|------|
| `labels` / `label_aliases` | 已注册标签和别名 |
| `clusters` | 簇状态、代表短语、质心、最近标签距离、promoted label 映射 |
| `cluster_phrases` | 簇内短语计数 |
| `cluster_source_docs` | 簇关联文档 |
| `documents` | 原始文档文本 |
| `label_examples` | 文档与标签/簇之间的证据关系 |
| `metadata` | 运行阈值、簇计数器等元信息 |

### 4.2 这次新增的持久化能力

- `centroid_json`
  - 保存簇质心，重启后还能继续语义聚类
- `promoted_label_id`
  - 显式记录 promoted 簇对应的新 label id
- `record_match_result(...)`
  - 持久化文档和样本证据
- `approve_cluster_examples(...)`
  - 当簇晋升后，把关联样本批量转成该标签的已批准正例
- `count_label_examples(...)`
  - 统计一个标签已有多少正例文档
- `get_slow_sync_ready_labels(...)`
  - 找出样本数足够的标签
- `export_ltce_samples(...)`
  - 导出 `LtceTextSample`

---

## 5. OntologyWriter facade

`OntologyWriter` 是业务上应该直接使用的入口。

### 5.1 基础写入接口

```python
writer.ingest(result)
writer.ingest_batch(results)
writer.register_existing_label(label_id, canonical_text, aliases=None, description=None)
writer.promote(cluster_id, new_label_id, skip_constraints=False)
writer.auto_promote_all(skip_constraints=False)
```

### 5.2 这次新增的闭环接口

```python
writer.ingest_with_document(
    result,
    document_text=doc_text,
    source_type="discovery",
    split="train",
)

writer.count_label_examples("cs.GNN")
writer.get_slow_sync_ready_labels(min_positive_examples=3)
writer.export_ltce_samples(min_positive_examples=3)
```

推荐使用方式已经从单纯的 `ingest(...)` 变成：

1. Discovery 产出 `MatchResult`
2. Writer 用 `ingest_with_document(...)` 同时写入聚类状态和文档证据
3. 候选簇晋升后，Writer 直接导出 `LtceTextSample`
4. Absorption 用这些样本构造 expanded loader 并执行 slow sync

### 5.3 示例

```python
from owlu.writer import OntologyWriter
from owlu.writer.constraints import OntologyConstraintChecker

checker = OntologyConstraintChecker.for_aapd()
writer = OntologyWriter(
    constraint_checker=checker,
    db_path="owlu_state.db",
)

writer.register_existing_label("cs.AI", "Artificial Intelligence")

action = writer.ingest_with_document(
    result,
    document_text=document_text,
)

if action == "candidate":
    for cluster_id in writer.get_promotion_candidates():
        writer.promote(cluster_id, "cs.NewLabel", skip_constraints=True)

samples = writer.export_ltce_samples(min_positive_examples=2)
```

---

## 6. 与 Absorption 的接口

Writer 当前对 Absorption 提供两类输出：

1. `promoted_labels`
   - 告诉 slow sync 需要新增哪些标签
2. `LtceTextSample`
   - 告诉 slow sync 这些新标签有哪些正例文本

这两个输出缺一不可：

- 没有 `promoted_labels`，模型不会扩标签维度
- 没有 `LtceTextSample`，模型虽然能扩维，但没有新标签监督样本可学

---

## 7. 当前验证结果

Writer 相关测试已覆盖：

- 归一化与路由
- 四门控晋升
- 语义聚类与代表短语选择
- SQLite round-trip
- promoted cluster 显式映射恢复
- 文档样本落库、审批、计数、导出
- 已有标签 merge 样本导出

本地结果：

```text
python -m pytest owlu/tests/test_writer.py -q
35 passed in 0.18s
```

全量仓库结果：

```text
python -m pytest owlu/tests -q
39 passed, 1 warning in 65.53s
```

---

## 8. 当前判断

到这一步，Writer 已经不是单纯的 ontology 更新器，而是 slow sync 的数据生产层。就实现完整性而言，这轮功能已经走通；后续工作的重点会从“补代码路径”转向“扩大真实正例样本规模”和“评估 slow sync 指标收益”。
