# Module 2: Ontology-Constrained Writer

> 把候选标签写入标签体系（以 AAPD 为例），在写入前进行本体约束检查。

## 模块定位

Ontology-Constrained Writer 是闭环体系的**决策中枢**。接收 Discovery 产出的 `MatchResult`，完成跨文档证据积累、簇聚合晋升，并在晋升前执行本体约束校验，确保新标签符合目标分类体系的结构规范。

## 目录结构

```
owlu/writer/
├── __init__.py      # OntologyWriter — Facade 统一入口
├── label_bank.py    # LabelBank — 跨文档积累、簇管理、晋升
└── constraints.py   # OntologyConstraintChecker — 本体约束检查
```

---

## 1. label_bank.py — 跨文档积累与晋升

### 类：`LabelBank`

**职责**：维护所有标签元数据和候选簇的完整生命周期状态。

#### 状态层次

```
labels                 ← 已注册标签 {label_id → LabelInfo}
proto_label_clusters   ← 所有簇（含 hold / candidate / promoted）
  ├── hold_pool        ← 证据不足，等待积累
  ├── candidate_labels ← 达到晋升阈值，待审核
  └── promoted_labels  ← 已晋升为正式标签
```

#### 核心方法

| 方法 | 签名 | 说明 |
|------|------|------|
| `register_label` | `(label_id, canonical_text, aliases?, description?)` | 注册/更新已有标签（用于初始化标签库） |
| `add_alias` | `(label_id, phrase, description?)` | 为已有标签添加别名短语 |
| `process_match_result` | `(result: MatchResult) -> str` | 路由入口：merge → 加别名；novel → 加候选；hold → 加 hold |
| `add_candidate` | `(phrase, ...) -> "candidate" \| "hold"` | 写入候选池，满足四重门控则升为 candidate |
| `add_hold` | `(phrase, ...) -> "hold"` | 积累证据但不触发晋升判定 |
| `promote_cluster` | `(cluster_id, new_label_id)` | 将 candidate 簇晋升为正式标签 |
| `build_review_packet` | `(cluster_id) -> dict` | 生成人工审核包（代表短语、来源、证据、距离） |

#### 簇聚合逻辑 (`_upsert_cluster`)

```
CandidatePhrase 到达
    │
    ▼
cluster = proto_label_clusters[cluster_id]  # 按短语归一化文本做 key
    │
    ├── freq += 1
    ├── agreement_sum += phrase.agreement
    ├── source_docs.add(doc_id)
    ├── phrases[normalized] += 1
    └── representative_phrase = max(phrases, key=票数)
```

#### 短语归一化 (`_normalize_phrase`)

```python
text → lowercase → 去除标点 → 合并空白
```

例如 `"Deep-Learning"` → `"deep learning"`，`"N.L.P."` → `"nlp"`。

标点去除确保词法变体（连字符、缩写点等）在归一化阶段被消除，不会产生冗余簇。

#### 晋升条件（四重门控）

簇从 `hold` 晋升为 `candidate` 需**同时**满足以下四个条件，组织在两个学术框架下：

**框架 1：跨文档众包质量控制 (Dawid & Skene, 1979)**

将每篇文档视为一个独立标注者（annotator），LLM 对文档的短语提取视为标注行为（annotation），则候选簇的晋升等价于一个众包标注质量判定问题——只有当**足够多的独立标注者**以**足够高的一致性**提取了同一短语时，该短语才被认为是可靠的标签候选。

| 条件 | 参数 | 默认值 | 众包类比 | 学术来源 |
|------|------|--------|----------|----------|
| 标注量充足 | `min_freq` | 3 | ≥ k 条标注 | 最小支持度 (Agrawal & Srikant, 1994)；X-MLClass §5.3 使用 ≥15 次 |
| 标注者独立 | `min_source_docs` | 2 | ≥ n 个独立标注者 | 多标注者独立观测假设 (Dawid & Skene, 1979) |
| 标注一致 | `min_agreement` | 0.5 | 标注者间一致性 | 标注者一致性质量控制 (Dawid & Skene, 1979) |

`min_agreement` 利用 Discovery 阶段 `multi_sample_aggregate` 已计算的一致性分数 `agreement ∈ [0, 1]`，在簇内累积为 `agreement_sum / agreement_count`，使晋升决策同时考虑**数量**和**质量**。

**框架 2：标签空间新颖性约束 (Li et al., 2024)**

| 条件 | 参数 | 默认值 | 学术来源 |
|------|------|--------|----------|
| 语义新颖 | `min_semantic_distance` | 0.3 | X-MLClass §5.3 语义距离过滤 |

要求 `nearest_label_distance ≥ 0.3`（即与最近已有标签的 cosine similarity < 0.7），确保只有语义不可归并的候选才被晋升。该距离由 Discovery 阶段的 `SemanticMatcher` 计算并通过 `MatchResult.similarity` 传入，Writer 内不重复计算嵌入。

#### 晋升判定伪代码

```python
def should_promote(cluster) -> bool:
    return (
        cluster.freq >= min_freq
        and cluster.source_doc_count >= min_source_docs
        and cluster.agreement >= min_agreement
        and (cluster.nearest_label_distance or 1.0) >= min_semantic_distance
    )
```

#### MatchResult 路由表

| MatchResult.action | LabelBank 操作 | 返回值 |
|--------------------|----------------|--------|
| `"merge_pre"` + target_label | `add_alias(target_label, phrase)` | `"merge"` |
| `"novel_pre"` | `add_candidate(phrase)` | `"candidate"` 或 `"hold"` |
| `"hold_pre"` | `add_hold(phrase)` | `"hold"` |
| 其他 | 忽略 | `"discard"` |

---

## 2. constraints.py — 本体约束检查

### 类：`OntologyConstraintChecker`

**职责**：在标签晋升前验证新标签是否符合目标本体结构。

#### 约束规则

| 规则 | 检查内容 | 触发条件 |
|------|----------|----------|
| `naming_format` | 新标签 ID 必须匹配正则模式 | 配置了 `naming_pattern` |
| `parent_existence` | 父类别必须在已知集合中 | 配置了 `known_parents` 且标签含 `.` 分隔符 |
| `duplicate` | 不得与已有标签（大小写不敏感）重复 | 始终生效 |

#### 方法

| 方法 | 签名 | 说明 |
|------|------|------|
| `check` | `(new_label_id, representative_phrase, existing_label_ids) -> ConstraintViolation \| None` | 返回 None 表示通过，否则返回违规详情 |
| `for_aapd` | `(parent_categories?) -> OntologyConstraintChecker` | 工厂方法：预配置 arXiv 分类体系约束 |

#### AAPD 预配置

```python
checker = OntologyConstraintChecker.for_aapd()
# naming_pattern = r"^[a-z\-]+\.[A-Z][A-Za-z0-9\-]+$"
# known_parents = {"cs", "stat", "math", "physics", "econ", ...}
```

这意味着：
- `cs.NLP` ✓ — 格式正确，父类 `cs` 存在
- `deeplearning` ✗ — 格式不匹配（缺少 `.` 分隔符）
- `xx.AI` ✗ — 父类 `xx` 不在已知集合中

---

## 3. Facade：`OntologyWriter`

**位置**：`writer/__init__.py`

将 LabelBank + ConstraintChecker 封装为统一写入接口：

```python
class OntologyWriter:
    def ingest(result: MatchResult) -> str                     # 单条写入
    def ingest_batch(results: list[MatchResult]) -> list[str]  # 批量写入
    def register_existing_label(label_id, canonical_text, ...)  # 初始化已有标签
    def get_promotion_candidates() -> dict[str, ProtoLabelCluster]  # 查看待晋升簇
    def promote(cluster_id, new_label_id, skip_constraints?) -> bool  # 晋升（含校验）
    def auto_promote_all(skip_constraints?) -> list[str]       # 自动晋升所有候选
    def get_promoted_labels() -> dict[str, ProtoLabelCluster]  # 查看已晋升标签
    def get_label_inventory() -> dict[str, str]                # 导出标签清单给 Discovery
```

### 典型调用

```python
from owlu.writer import OntologyWriter
from owlu.writer.constraints import OntologyConstraintChecker

# 使用 AAPD 约束
checker = OntologyConstraintChecker.for_aapd()
writer = OntologyWriter(constraint_checker=checker)

# 初始化已有标签
for label_id, name in existing_labels.items():
    writer.register_existing_label(label_id, name)

# 接收 Discovery 产出
for result in discovery_results:
    action = writer.ingest(result)
    print(f"{result.phrase.text} → {action}")

# 查看待晋升候选
candidates = writer.get_promotion_candidates()
for cid, cluster in candidates.items():
    packet = writer.build_review_packet(cid)
    print(packet)

# 晋升（含本体约束校验）
ok = writer.promote("cluster_xyz", "cs.NLP")
# ok=True → 晋升成功；ok=False → 约束不通过

# 导出更新后的标签清单，回传给 Discovery
discovery.update_label_inventory(writer.get_label_inventory())
```

---

## 输入 / 输出契约

### 输入
- 来自 Discovery：`MatchResult`（通过 `ingest()` 接收）

### 输出
- 给 Absorption：`promoted_labels` 字典（通过 `get_promoted_labels()` 查询）
- 给 Discovery：更新后的 `label_inventory`（通过 `get_label_inventory()` 导出）
- 给 Absorption：`LabelBank` 实例（直接传入 `PrototypeAbsorption` 构造函数）

### 关键数据结构

| 结构 | 位置 | 说明 |
|------|------|------|
| `LabelInfo` | `common/types.py` | `{label_id, aliases: set, description}` |
| `ProtoLabelCluster` | `common/types.py` | `{cluster_id, phrases, freq, source_docs, state, agreement, ...}` |
| `ConstraintViolation` | `writer/constraints.py` | `{rule, message}` — 约束不通过时返回 |

---

## 状态机

```
                    ┌──────────┐
   novel_pre ──────▶│   hold   │◀─── hold_pre
                    └────┬─────┘
                         │ freq ≥ 3 AND docs ≥ 2
                         │ AND agreement ≥ 0.5
                         │ AND semantic_dist ≥ 0.3
                         ▼
                    ┌──────────┐
                    │candidate │
                    └────┬─────┘
                         │ promote() + constraints check
                         ▼
                    ┌──────────┐
   merge_pre ─────▶│ promoted │──▶ Absorption (fast/slow sync)
   (加别名)        └──────────┘
```

---

## 与 X-MLClass 的关系

本模块的晋升机制参考了 X-MLClass (Li et al., 2024) 的标签空间构建策略，关键差异如下（具体对应关系见「晋升条件」节）：

| 维度 | X-MLClass | 本模块 |
|---|---|---|
| 处理模式 | 批处理（一次性处理 $\mathcal{D}_{sub}$） | 流式（逐文档积累） |
| 频率阈值 | 高（≥15 次） | 低（≥3 次），额外要求来源多样性 |
| 语义距离阈值 γ | 自适应（取中位数） | 固定（0.3） |
| 去重方式 | Sentence-Transformer cosine（重模型） | 归一化 + 精确去重（轻量） |
| 质量信号 | 无（仅频率） | LLM 多次采样一致性 |

### 参考文献

- Agrawal, R. & Srikant, R. (1994). Fast algorithms for mining association rules. *Proc. 20th VLDB Conference*, 487–499.
- Dawid, A. P. & Skene, A. M. (1979). Maximum likelihood estimation of observer error-rates using the EM algorithm. *Journal of the Royal Statistical Society: Series C*, 28(1), 20–28.
- Li, X., Jiang, J., Dharmani, R., Srinivasa, J., Liu, G., & Shang, J. (2024). Open-world Multi-label Text Classification with Extremely Weak Supervision. arXiv:2407.05609.
