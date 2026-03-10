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
| `add_candidate` | `(phrase, ...) -> "candidate" \| "hold"` | 写入候选池。满足 freq ≥ min_freq AND source_docs ≥ min_source_docs 则升为 candidate |
| `add_hold` | `(phrase, ...) -> "hold"` | 强制写入 hold 池 |
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

#### 晋升条件

| 条件 | 默认值 | 含义 |
|------|--------|------|
| `min_freq` | 3 | 该短语（簇）至少被提取 3 次 |
| `min_source_docs` | 2 | 来自至少 2 篇不同文档 |

两条同时满足 → 状态变为 `candidate`，可被 promote。

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
                         │ freq ≥ 3 AND source_docs ≥ 2
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
