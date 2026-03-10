# Module 1: Candidate Discovery Module

> 从"模型目前解释不好的样本"里，发现值得进入闭环体系审核的候选标签。

## 模块定位

Candidate Discovery 是 OWLU 闭环的**入口**。它监听 LTCE 模型的输出置信度，对低置信（潜在 OOD）文档触发 LLM 短语抽取，随后通过轻量语义匹配给出初步判定（merge / novel / hold），产出 `MatchResult` 送入下游 Writer。

## 目录结构

```
owlu/discovery/
├── __init__.py          # CandidateDiscovery — Facade 统一入口
├── gate.py              # LtceGate — LTCE 置信度门控
├── phrase_generator.py  # LLMPhraseGenerator — DeepSeek 短语抽取
└── matcher.py           # SemanticMatcher — BOW 语义匹配 & 初判
```

---

## 1. gate.py — LTCE 置信度门控

### 类：`LtceGate`

**职责**：根据 LTCE 模型的 raw logits，决定是否为该文档调用 LLM。

**设计理由**：  
多标签 sigmoid 分类器对 OOD 样本会对**所有标签**输出接近零的 sigmoid 激活。传统 OOD 指标（熵、cosine、阈值以上标签数）在这种场景下表现反常。因此采用 `raw_max_prob`（T=1.0 下最大 sigmoid 值）作为识别信号。

#### 核心方法

| 方法 | 签名 | 说明 |
|------|------|------|
| `raw_max_prob` | `(logits: Sequence[float]) -> float` | 静态方法。对 raw logits 逐元素 sigmoid 后取 max。 |
| `calibrate` | `(validation_logits: Sequence[Sequence[float]]) -> float` | 从验证集 logits 自适应校准阈值：取第 `adaptive_percentile` 百分位，clamp 到 `recognition_floor`。 |
| `evaluate` | `(logits: Sequence[float]) -> GateDecision` | 单文档判定：`raw_max_prob < threshold` → `should_invoke_llm=True`。 |
| `batch_evaluate` | `(batch_logits) -> list[GateDecision]` | 批量判定。 |
| `filter_for_llm` | `(doc_ids, batch_logits) -> list[str]` | 返回需要触发 LLM 的文档 ID 列表。 |

#### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `recognition_floor` | 0.10 | 自适应阈值下限，防止坍缩到零 |
| `adaptive_percentile` | 0.05 | 取验证集 raw_max_prob 排序后的第 5% 位 |
| `fixed_threshold` | None | 指定后跳过校准，直接使用 |

#### 数据流

```
LTCE logits ──▶ raw_max_prob() ──▶ < threshold? ──▶ GateDecision
                                        │
                          calibrate() ◀─┘ (验证集)
```

---

## 2. phrase_generator.py — LLM 短语抽取

### 类：`LLMPhraseGenerator`

**职责**：调用 DeepSeek 兼容 API，从文档文本中提取候选名词短语。

#### 核心方法

| 方法 | 签名 | 说明 |
|------|------|------|
| `generate` | `(text, doc_id) -> list[CandidatePhrase]` | 单次 LLM 调用，pass_id=1，agreement=1.0。 |
| `should_trigger_uncertain` | `(top1_score, top2_score) -> bool` | 判定是否需要多采样：top1 < 0.45 或 margin < 0.15。 |
| `multi_sample_aggregate` | `(text, doc_id, k=3) -> list[CandidatePhrase]` | k 次 LLM 调用 + 一致性投票。agreement = votes/k。 |
| `generate_uncertain_batch` | `(texts, doc_ids, scores) -> list[CandidatePhrase]` | 批量不确定性触发入口。 |

#### 内部流程

```
text ──▶ _request_once()
             │
             ▼
        LLM JSON response
             │
        _extract_json_payload()   # 容错：纯 JSON / fenced markdown
             │
        _build_candidates()       # 去重、截断到 max_phrases
             │
             ▼
        list[CandidatePhrase]
```

**多采样聚合** (`multi_sample_aggregate`):
1. 发起 k 次独立 LLM 调用
2. 对所有返回短语按 `phrase.lower()` 聚合投票
3. 取票数最高的 top-`max_phrases` 个短语
4. `agreement = votes[phrase] / k`

#### 异常处理

- `LLMOutputError`：JSON 解析失败、响应为空、`phrases` 字段类型异常
- `EnvironmentError`：`DEEPSEEK_API_KEY` 未设置

---

## 3. matcher.py — 语义匹配 & 初判

### 类：`SemanticMatcher`

**职责**：将 LLM 产出的 `CandidatePhrase` 与现有标签库做轻量语义匹配，输出 `MatchResult`。

**设计选择**：使用 BOW (Bag-of-Words) cosine 相似度，**零外部 NLP 依赖**（不需要 spaCy、NLTK 等）。

#### 核心方法

| 方法 | 签名 | 说明 |
|------|------|------|
| `normalize` | `(phrase) -> str` | 小写 → 正则分词 → 去停用词 → 朴素词形还原 → 截断到 8 token。 |
| `match` | `(phrase, labels: {id: text}) -> MatchResult` | 对标签库逐一计算 cosine，取最高分，调用 `preliminary_decide`。 |
| `preliminary_decide` | `(s_max, agreement) -> str` | 三路判定：merge_pre / novel_pre / hold_pre。 |

#### 决策规则

```
if s_max ≥ 0.80 AND agreement ≥ 0.67 → "merge_pre"   (合并到已有标签)
elif s_max < 0.52                    → "novel_pre"    (可能是新标签)
else                                 → "hold_pre"     (暂不决策，积累更多证据)
```

---

## 4. Facade：`CandidateDiscovery`

**位置**：`discovery/__init__.py`

将 Gate + Generator + Matcher 封装为统一接口：

```python
class CandidateDiscovery:
    def calibrate_gate(validation_logits) -> float
    def discover(doc_id, text, logits) -> list[MatchResult]
    def discover_uncertain(doc_id, text, logits, top1, top2) -> list[MatchResult]
    def batch_discover(doc_ids, texts, batch_logits) -> dict[str, list[MatchResult]]
    def update_label_inventory(label_inventory) -> None
```

### 典型调用

```python
from owlu.discovery import CandidateDiscovery
from owlu.common.types import OWLUConfig

config = OWLUConfig.from_yaml("owlu/configs/owlu.yaml")
discovery = CandidateDiscovery(config, label_inventory={"cs.AI": "artificial intelligence", ...})

# 校准门控
discovery.calibrate_gate(validation_logits)

# 单文档发现
results = discovery.discover(doc_id="doc_001", text="...", logits=[...])
# results: list[MatchResult]  → 送入 OntologyWriter.ingest()
```

---

## 输出契约

本模块的唯一输出类型是 `list[MatchResult]`，每个 `MatchResult` 包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `phrase` | `CandidatePhrase` | 原始候选短语（含 agreement、evidence 等） |
| `action` | `str` | `"merge_pre"` / `"novel_pre"` / `"hold_pre"` |
| `target_label` | `str \| None` | merge 时指向的现有标签 ID |
| `similarity` | `float` | 与最近标签的 cosine 相似度 |
| `decision_reason` | `str` | 人类可读的决策理由 |
| `normalized_phrase` | `str` | 归一化后的短语文本 |

下游 **Ontology-Constrained Writer** 通过 `OntologyWriter.ingest(result)` 消费这些输出。
