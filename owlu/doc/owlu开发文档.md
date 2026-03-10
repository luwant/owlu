# OWLU × LTC-MPE 开发文档（兼容修订版）

## 0. 文档定位

本文档为 OWLU（Open World Label Updating Module）与 LTC-MPE 的兼容版开发方案。

目标是以最小工程改动完成开放标签创新，同时保持与现有 LTC-MPE 训练与推理主链路兼容。

本版强调三点：

1. 创新点聚焦标签体系生成与长尾发现，不引入重型开放世界训练框架。
2. LLM 固定使用 DeepSeek API（OpenAI 兼容调用方式）。
3. 明确快慢同步边界：快同步不改标签维度，慢同步采用周期扩容与增量微调。

---

## 1. 总体设计目标

### 1.1 研究问题

外部公开文本持续变化会导致两类失配：

- 已有标签的表达老化，覆盖不足；
- 新概念无法进入标签空间，模型仅在旧标签集合上工作。

OWLU 的职责是从外部文本中持续提取标签信号，完成标签语义更新与候选新标签维护，并周期性同步到 LTC-MPE。

### 1.2 总体流程

```text
外部公开文本流
   ↓
[Step 1] DeepSeek 候选短语归纳（首轮）
   ↓
[Step 2] 低置信样本二次归纳 + 多采样一致性门控
   ↓
[Step 3] 标签匹配与标签库更新（merge/candidate/hold）
   ↓
[Step 4] 与 LTC-MPE 周期同步（快同步 / 慢同步）
```

### 1.3 与 LTC-MPE 的兼容边界

- 快同步：仅更新标签文本语义，不改变 `num_labels`，直接兼容现有 LTC-MPE。
- 慢同步：新增正式标签时采用周期扩容 `L -> L'` 后增量微调，不做在线热扩容声明。
- 不改变 LTC-MPE 主干计算图和现有训练入口。

---

## 2. 三个创新点

### 2.1 创新点 A：X-MLClass 式标签体系生成循环

从“单次短语抽取”升级为“抽取-聚合-回访”的循环：

- 首轮抽取候选短语；
- 对低置信文档回访，补挖长尾新表达；
- 形成 `proto-label clusters`，作为候选标签簇输入审核链路。

低置信触发条件：

- `top1_score < 0.45`，或
- `top1_score - top2_score < 0.15`。

### 2.2 创新点 B：多采样一致性门控

替换“多样本分布对齐门控”，采用更可实现的多采样一致性机制：

- 对低置信样本进行 `k=3` 次低温采样；
- 聚合后计算一致性 `agreement`（主短语投票占比）；
- 与语义相似度、频次联合决策。

决策规则：

- `merge`：`s_max >= 0.80` 且 `agreement >= 0.67`
- `candidate`：`s_max < 0.52` 且 `freq >= 3` 且 `source_docs >= 2`
- 其余进入 `hold_pool`（观察池）

### 2.3 创新点 C：候选簇轻量晋升

候选不直接升正式标签，先形成可审核簇包：

- 代表短语；
- 支撑文档；
- 与最近已有标签的距离；
- 来源频次与一致性统计。

人工仅做 yes/no 决策；通过后自动完成标签扩容初始化并进入慢同步。

---

## 3. 核心步骤设计

### 3.1 Step 1：DeepSeek 候选短语归纳（首轮）

输入：

- 文本 `x`
- `doc_id`

输出：

- `summary`
- `phrases`
- `evidence`

Prompt 目标：

- 只做候选短语归纳，不做最终分类；
- 输出严格 JSON。

### 3.2 Step 2：低置信回访与多采样一致性

- 仅对低置信文档触发二次归纳；
- 每条文档进行 `k=3` 次采样；
- 聚合同义项并计算 `agreement`；
- 产出二次候选并标记 `pass_id=2`。

### 3.3 Step 3：匹配与标签库更新

语义匹配：

\[
s(c, y) = \cos(e(c), e(y)), \quad s_{max}(c)=\max_{y\in\mathcal{Y}} s(c,y)
\]

动作：

- `merge`：加入目标标签 `aliases` 或补充 `description`
- `candidate`：加入候选簇池
- `hold`：进入观察池，等待下一周期

### 3.4 Step 4：与 LTC-MPE 周期同步

快同步（语义级）：

1. 刷新标签文本描述；
2. 重新编码标签向量；
3. 校准推理阈值。

慢同步（类别级）：

1. 周期末确认晋升簇，扩展标签集合 `L -> L'`；
2. 重新构建标签维度相关参数；
3. 使用旧样本回放 + 新增样本做一次增量微调；
4. 更新验证阈值后发布。

说明：慢同步为“周期扩容+增量微调”，不定义为在线热扩容。

---

## 4. DeepSeek API 约束（固定）

- Provider：DeepSeek
- Base URL：`https://api.deepseek.com`
- Model：`deepseek-chat`
- 鉴权：环境变量 `DEEPSEEK_API_KEY`
- 输出：强制 JSON（请求需显式声明 JSON 输出）

安全约束：

- 不在配置文件中写明文 API Key；
- 失败重试与超时由调用层统一控制。

---

## 5. 接口与类型（修订）

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

@dataclass
class CandidatePhrase:
    text: str
    raw_text: str
    source_doc_id: str
    timestamp: datetime
    summary: str | None = None
    evidence: list[str] | None = None
    agreement: float = 0.0
    pass_id: int = 1
    source_count: int = 1
    cluster_id: str | None = None

@dataclass
class MatchResult:
    phrase: CandidatePhrase
    action: Literal["merge", "candidate", "hold", "discard"]
    target_label: Optional[str] = None
    similarity: float = 0.0
    decision_reason: str = ""
    support_docs: int = 0
    agreement: float = 0.0

class LLMPhraseGenerator:
    def generate(self, text: str, doc_id: str) -> list[CandidatePhrase]:
        ...

    def generate_uncertain_batch(self, texts: list[str], doc_ids: list[str]) -> list[CandidatePhrase]:
        ...

    def multi_sample_aggregate(self, text: str, doc_id: str, k: int = 3) -> list[CandidatePhrase]:
        ...

class LabelBank:
    proto_label_clusters: dict
    hold_pool: dict

    def add_alias(self, label_id: str, phrase: str) -> None:
        ...

    def add_candidate(self, phrase: CandidatePhrase) -> None:
        ...

    def add_hold(self, phrase: CandidatePhrase) -> None:
        ...

    def promote_cluster(self, cluster_id: str, new_label_id: str) -> None:
        ...
```

---

## 6. 关键参数（默认）

```yaml
tau_merge: 0.80
tau_new: 0.52
min_freq: 3
min_source_docs: 2
agreement_threshold: 0.67
uncertain_top1_threshold: 0.45
uncertain_margin_threshold: 0.15
multi_sample_k: 3
llm_model: deepseek-chat
llm_base_url: https://api.deepseek.com
```

---

## 7. 测试计划

### 7.1 兼容性测试

- 快同步后推理链路可运行，维度无报错；
- 慢同步后 `L'` 训练与评估可运行；
- 标签扩容后历史标签性能可接受。

### 7.2 有效性测试

- 标签覆盖率；
- Micro-F1 / Macro-F1；
- 长尾标签 F1；
- 候选审核通过率。

### 7.3 成本测试

- 平均 token 消耗；
- 二次采样触发比例；
- 平均文档处理延迟。

### 7.4 消融实验

- 去掉一致性门控；
- 去掉不确定性回访；
- 去掉候选簇晋升。

---

## 8. 假设与边界

- 保持轻量实现，不引入完整生成分布建模训练框架；
- “分布偏置相关论文”仅用于问题动机与误差分析，不宣称复现其完整方法；
- 默认覆盖原文档，不另建副本。

---

## 9. 参考文献

1. Li, X., Jiang, J., et al. 2024. Open-world Multi-label Text Classification with Extremely Weak Supervision. EMNLP 2024.
2. X-MLClass 项目仓库: https://github.com/Kaylee0501/X-MLClass
3. Reimers, N. & Gurevych, I. 2019. Sentence-BERT. EMNLP-IJCNLP 2019.
4. Chen, S.-A., et al. 2025. Preserving Zero-shot Capability in Supervised Fine-tuning for Multi-label Text Classification. Findings of NAACL 2025.
5. LLMs Do Multi-Label Classification Differently. EMNLP 2025.
