# 任务1：OWLU 抽取与预判门控开发

## 1. 任务目标

完成 OWLU 前半链路，覆盖：

- 配置与运行骨架
- DeepSeek 首轮候选短语抽取
- 短语规范化与语义匹配
- 低置信回访 + 多采样一致性门控（单文档级）

该任务交付后，只输出单文档级 `preliminary decision`，不做跨文档 `candidate` 最终判定。

---

## 2. 开发范围

### 2.1 代码范围

- `owlu/llm_phrase_generator.py`
- `owlu/semantic_matcher.py`
- `owlu/configs/owlu.yaml`

### 2.2 关键规则

- DeepSeek 固定参数：
  - `base_url=https://api.deepseek.com`
  - `model=deepseek-chat`
  - `api_key` 固定写入 `owlu/configs/owlu.yaml`
  - 若配置未提供，则回退 `DEEPSEEK_API_KEY` 环境变量
  - JSON 输出强约束
- 预判规则（Task 1 仅单文档）：
  - `merge_pre`: `s_max >= 0.80` 且 `agreement >= 0.67`
  - `novel_pre`: `s_max < 0.52`
  - `hold_pre`: 其余情况
- 低置信触发：
  - `top1_score < 0.45` 或 `top1-top2 < 0.15`
- 多采样默认：`k=3`
- 说明：`freq >= 3` 与 `source_docs >= 2` 依赖跨文档状态，放在 Task 2 的 LabelBank 完成。

### 2.3 接口要求

- `LLMPhraseGenerator.generate()`
- `LLMPhraseGenerator.generate_uncertain_batch(texts, doc_ids, scores)`
- `LLMPhraseGenerator.multi_sample_aggregate()`
- `LLMPhraseGenerator.should_trigger_uncertain(top1_score, top2_score)`
- `SemanticMatcher.normalize()`
- `SemanticMatcher.match()`
- `SemanticMatcher.preliminary_decide()`（返回 `merge_pre/novel_pre/hold_pre`）
- `OWLUConfig.from_yaml()`（从 `owlu/configs/owlu.yaml` 加载参数）
- `get_api_key()`（当配置缺失时回退读取 `DEEPSEEK_API_KEY`）

---

## 3. 测试验证

### 3.1 必做测试

- `test_config_load`
- `test_env_key_required`
- `test_generate_json_contract`
- `test_generate_handles_invalid_json`
- `test_normalize_pipeline`
- `test_match_merge_threshold`
- `test_preliminary_decision_without_global_state`
- `test_uncertain_trigger`
- `test_multi_sample_agreement`
- `test_gate_rule_with_agreement`

可选真实 API 测试（integration）：

- `test_live_generate_smoke`（文件：`owlu/tests/test_task1_live_api.py`）

### 3.2 通过标准（DoD）

- 单元测试全部通过。
- 低置信文档会触发二次采样，普通文档不会误触发。
- 输出对象包含 `agreement/pass_id/source_count` 等新字段。
- 不依赖 LabelBank 即可运行，且不会输出最终 `candidate`。
- 日志可追踪 `doc_id` 与判定原因。
- 本任务测试默认使用 mock LLM 客户端，不依赖外网调用。
- 若执行真实 API 测试，需显式设置 `OWLU_RUN_LIVE=1`。

---

## 4. 交付物

- 可运行的抽取与门控模块代码
- 对应测试代码与测试报告
- 更新后的 `owlu.yaml` 配置样例
- 测试入口：`python -m pytest -q owlu/tests/test_task1.py`
- 真实 API 入口：`$env:OWLU_RUN_LIVE='1'; python -m pytest -q owlu/tests/test_task1_live_api.py`
