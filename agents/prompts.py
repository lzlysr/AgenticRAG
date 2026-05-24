"""Prompt 分级管理：按模型能力加载不同 prompt 和配置参数，支持中英文切换"""
import re
from config import AGENT_LLM_MODEL


# ── Planner Prompts ──────────────────────────────────────────────────────────

_PLANNER_PROMPT_SMALL = """You are a query decomposition planner for a multi-hop QA system.

Break the complex question into sequential sub-queries that can each be answered by searching a knowledge base.

Question: {query}

{feedback_section}

Available tools for each sub-query:
{tools_section}

Respond in JSON array format:
[
  {{"id": 1, "sub_query": "...", "tool": "<tool_name>", "depends_on": []}},
  {{"id": 2, "sub_query": "...", "tool": "<tool_name>", "depends_on": [1]}},
  ...
]

IMPORTANT RULES:
- Keep the plan MINIMAL. Each step should search for ONE specific entity or fact. Only add steps that are truly necessary.
- Prefer semantic_search as default tool. Only use keyword_search when searching for very specific technical names or codes. Use graph_search when the question involves tracing entity relationships (e.g., comparing entities or following a chain like "the director of the film starring X").
- For critical steps where high recall matters, you can specify multiple tools as a list (e.g., "tool": ["semantic_search", "keyword_search"]) to fuse results from multiple retrieval methods.
- Do NOT create redundant steps that search for the same thing with different tools.
- Do NOT add verification or confirmation steps - just search for the needed facts."""

_PLANNER_PROMPT_LARGE = """You are a query decomposition planner for a multi-hop QA system.

Break the complex question into sequential sub-queries that can each be answered by searching a knowledge base.

Question: {query}

{feedback_section}

Available tools for each sub-query:
{tools_section}

Respond in JSON array format:
[
  {{"id": 1, "sub_query": "...", "tool": "<tool_name>", "depends_on": []}},
  {{"id": 2, "sub_query": "...", "tool": "<tool_name>", "depends_on": [1]}},
  ...
]

IMPORTANT RULES:
- Keep the plan MINIMAL. Each step should search for ONE specific entity or fact. Only add steps that are truly necessary.
- Choose the best tool for each sub-query: use keyword_search for exact names/codes, semantic_search for conceptual or descriptive queries, graph_search for tracing entity relationships across documents (e.g., comparing two entities, or following a chain like "the director of the film starring X").
- For critical steps where high recall matters, you can specify multiple tools as a list (e.g., "tool": ["semantic_search", "keyword_search"]) to fuse results from multiple retrieval methods.
- Do NOT create redundant steps that search for the same thing with different tools.
- Do NOT add verification or confirmation steps - just search for the needed facts."""

# ── Verifier Prompts ─────────────────────────────────────────────────────────

_VERIFIER_PROMPT_SMALL = """You are an evidence sufficiency verifier for a multi-hop QA system.

Original Question: {query}

Collected Evidence:
{evidence_text}

Evaluate whether the collected evidence is sufficient to answer the original question.

IMPORTANT: Be LENIENT. If the evidence contains information about the key entities mentioned in the question, judge it as "sufficient". Only judge "insufficient" if critical entities are completely missing from ALL evidence.

Respond in JSON:
{{
  "verdict": "sufficient" or "insufficient",
  "reasoning": "brief assessment",
  "feedback": "if insufficient, what specific entity/fact is completely missing?"
}}"""

_VERIFIER_PROMPT_LARGE = """You are an evidence sufficiency verifier for a multi-hop QA system.

Original Question: {query}

Collected Evidence:
{evidence_text}

Evaluate whether the collected evidence is sufficient to answer the original question.

Check that:
1. Evidence covers ALL key entities mentioned in the question.
2. The specific facts needed to answer (e.g., nationality, date, attribute) are present.
3. For comparison questions, evidence exists for EACH entity being compared.

Respond in JSON:
{{
  "verdict": "sufficient" or "insufficient",
  "reasoning": "brief assessment",
  "feedback": "if insufficient, what specific entity/fact is missing?"
}}"""

# ── Synthesizer Prompt (相同) ────────────────────────────────────────────────

_SYNTHESIZER_PROMPT = """You are an answer synthesizer for a QA benchmark. Based on the collected evidence, answer the question.

Question: {query}

Evidence:
{evidence_text}

CRITICAL RULES:
- Output ONLY the answer itself, nothing else
- The answer should be a short entity, name, number, yes/no, or brief phrase
- Do NOT explain your reasoning
- Do NOT add qualifications, caveats, or "based on the evidence" phrases
- Do NOT repeat the question
- Examples of good answers: "Paris", "yes", "42", "Albert Einstein", "the blue one"

Answer:"""

# ── Simple RAG Prompt (相同) ─────────────────────────────────────────────────

_SIMPLE_RAG_PROMPT = """Answer the question based on the following context. Output ONLY the answer as a short phrase or entity. No explanations.

Context:
{context}

Question: {query}

Answer:"""

# ── Router Prompt (相同) ─────────────────────────────────────────────────────

_ROUTER_PROMPT = """You are a query complexity classifier. Analyze the question and determine if it requires:
- "simple": Can be answered with a single retrieval step (single fact lookup)
- "multi_hop": Requires multiple retrieval steps, comparison, or inference across documents

Question: {query}

Respond in JSON: {{"query_type": "simple" or "multi_hop", "reasoning": "brief explanation"}}"""

# ── Replan Feedback (相同) ───────────────────────────────────────────────────

_REPLAN_FEEDBACK = """Previous plan was insufficient. Verifier feedback:
{feedback}

Already tried searches (tool + query):
{evidence_summary}

Create an improved plan that addresses the gaps. IMPORTANT:
- Do NOT repeat the same tool + query combinations listed above.
- Try a DIFFERENT tool (e.g., switch from semantic_search to keyword_search or graph_search) or rephrase the query with different keywords.
- Focus on the specific missing entity/fact mentioned in the feedback."""

# ── 中文 Prompt ──────────────────────────────────────────────────────────────

_PLANNER_PROMPT_SMALL_ZH = """你是一个多跳问答系统的查询分解规划器。

将复杂问题拆解为可通过知识库检索逐一回答的子查询序列。

问题：{query}

{feedback_section}

每个子查询可用的工具：
{tools_section}

以 JSON 数组格式回复：
[
  {{"id": 1, "sub_query": "...", "tool": "<工具名>", "depends_on": []}},
  {{"id": 2, "sub_query": "...", "tool": "<工具名>", "depends_on": [1]}},
  ...
]

重要规则：
- 计划尽量精简。每步只检索一个具体实体或事实，只添加真正必要的步骤。
- 默认使用 semantic_search。仅在搜索非常具体的专有名词或代码时使用 keyword_search。当问题涉及实体关系追踪（如比较两个实体、或"出演X的电影的导演"这类链式推理）时使用 graph_search。
- 对召回率要求高的关键步骤，可以指定多个工具（如 "tool": ["semantic_search", "keyword_search"]）来融合多路检索结果。
- 不要创建用不同工具搜索相同内容的冗余步骤。
- 不要添加验证或确认步骤——只搜索所需的事实。"""

_PLANNER_PROMPT_LARGE_ZH = """你是一个多跳问答系统的查询分解规划器。

将复杂问题拆解为可通过知识库检索逐一回答的子查询序列。

问题：{query}

{feedback_section}

每个子查询可用的工具：
{tools_section}

以 JSON 数组格式回复：
[
  {{"id": 1, "sub_query": "...", "tool": "<工具名>", "depends_on": []}},
  {{"id": 2, "sub_query": "...", "tool": "<工具名>", "depends_on": [1]}},
  ...
]

重要规则：
- 计划尽量精简。每步只检索一个具体实体或事实，只添加真正必要的步骤。
- 为每个子查询选择最佳工具：精确名称/代码用 keyword_search，概念性或描述性查询用 semantic_search，跨文档实体关系追踪用 graph_search（如比较两个实体、或"出演X的电影的导演"这类链式推理）。
- 对召回率要求高的关键步骤，可以指定多个工具（如 "tool": ["semantic_search", "keyword_search"]）来融合多路检索结果。
- 不要创建用不同工具搜索相同内容的冗余步骤。
- 不要添加验证或确认步骤——只搜索所需的事实。"""

_VERIFIER_PROMPT_SMALL_ZH = """你是一个多跳问答系统的证据充分性验证器。

原始问题：{query}

已收集的证据：
{evidence_text}

评估已收集的证据是否足以回答原始问题。

重要：请宽松判定。如果证据包含问题中提到的关键实体的相关信息，就判定为"sufficient"。仅当关键实体在所有证据中完全缺失时才判定为"insufficient"。

以 JSON 格式回复：
{{
  "verdict": "sufficient" 或 "insufficient",
  "reasoning": "简要评估",
  "feedback": "如果 insufficient，具体缺少什么实体/事实？"
}}"""

_VERIFIER_PROMPT_LARGE_ZH = """你是一个多跳问答系统的证据充分性验证器。

原始问题：{query}

已收集的证据：
{evidence_text}

评估已收集的证据是否足以回答原始问题。

检查以下几点：
1. 证据是否覆盖了问题中提到的所有关键实体。
2. 回答所需的具体事实（如国籍、日期、属性）是否存在。
3. 对于比较类问题，是否每个被比较的实体都有对应证据。

以 JSON 格式回复：
{{
  "verdict": "sufficient" 或 "insufficient",
  "reasoning": "简要评估",
  "feedback": "如果 insufficient，具体缺少什么实体/事实？"
}}"""

_SYNTHESIZER_PROMPT_ZH = """你是一个问答基准测试的答案合成器。根据收集到的证据回答问题。

问题：{query}

证据：
{evidence_text}

关键规则：
- 只输出答案本身，不要输出其他任何内容
- 答案应该是简短的实体、名称、数字、是/否或简短短语
- 不要解释推理过程
- 不要添加限定词、注意事项或"根据证据"等短语
- 不要重复问题
- 好的答案示例："巴黎"、"是"、"42"、"爱因斯坦"、"蓝色的那个"

答案："""

_SIMPLE_RAG_PROMPT_ZH = """根据以下上下文回答问题。只输出简短的短语或实体作为答案，不要解释。

上下文：
{context}

问题：{query}

答案："""

_ROUTER_PROMPT_ZH = """你是一个查询复杂度分类器。分析问题并判断它需要：
- "simple"：单次检索即可回答（单一事实查找）
- "multi_hop"：需要多次检索、比较或跨文档推理

问题：{query}

以 JSON 格式回复：{{"query_type": "simple" 或 "multi_hop", "reasoning": "简要说明"}}"""

_REPLAN_FEEDBACK_ZH = """上一轮计划不充分。验证器反馈：
{feedback}

已尝试的检索（工具 + 查询）：
{evidence_summary}

请制定改进的计划来弥补不足。重要：
- 不要重复上面列出的相同工具+查询组合。
- 尝试不同的工具（如从 semantic_search 切换到 keyword_search 或 graph_search）或用不同的关键词重新表述查询。
- 聚焦于反馈中提到的具体缺失实体/事实。"""

# ── Profile 定义 ─────────────────────────────────────────────────────────────

PROMPT_PROFILES = {
    "small": {
        "planner": _PLANNER_PROMPT_SMALL,
        "verifier": _VERIFIER_PROMPT_SMALL,
        "synthesizer": _SYNTHESIZER_PROMPT,
        "simple_rag": _SIMPLE_RAG_PROMPT,
        "router": _ROUTER_PROMPT,
        "replan_feedback": _REPLAN_FEEDBACK,
        "max_iterations": 3,
        "max_retrieval_calls": 10,
    },
    "large": {
        "planner": _PLANNER_PROMPT_LARGE,
        "verifier": _VERIFIER_PROMPT_LARGE,
        "synthesizer": _SYNTHESIZER_PROMPT,
        "simple_rag": _SIMPLE_RAG_PROMPT,
        "router": _ROUTER_PROMPT,
        "replan_feedback": _REPLAN_FEEDBACK,
        "max_iterations": 5,
        "max_retrieval_calls": 15,
    },
    "small_zh": {
        "planner": _PLANNER_PROMPT_SMALL_ZH,
        "verifier": _VERIFIER_PROMPT_SMALL_ZH,
        "synthesizer": _SYNTHESIZER_PROMPT_ZH,
        "simple_rag": _SIMPLE_RAG_PROMPT_ZH,
        "router": _ROUTER_PROMPT_ZH,
        "replan_feedback": _REPLAN_FEEDBACK_ZH,
        "max_iterations": 3,
        "max_retrieval_calls": 10,
    },
    "large_zh": {
        "planner": _PLANNER_PROMPT_LARGE_ZH,
        "verifier": _VERIFIER_PROMPT_LARGE_ZH,
        "synthesizer": _SYNTHESIZER_PROMPT_ZH,
        "simple_rag": _SIMPLE_RAG_PROMPT_ZH,
        "router": _ROUTER_PROMPT_ZH,
        "replan_feedback": _REPLAN_FEEDBACK_ZH,
        "max_iterations": 5,
        "max_retrieval_calls": 15,
    },
}

# ── 模型 → Profile 映射 ─────────────────────────────────────────────────────

_SMALL_PATTERN = re.compile(r"(?:^|[^0-9])(?:7|8|14)[Bb]", re.IGNORECASE)


def get_profile(model_name: str | None = None) -> dict:
    """根据模型名称和语言返回对应的 prompt profile。

    规则：模型名包含 7B/8B/14B → small，其余 → large。
    语言：config.PROMPT_LANG = "zh" 时使用中文 prompt。
    """
    if model_name is None:
        model_name = AGENT_LLM_MODEL
    size = "small" if _SMALL_PATTERN.search(model_name) else "large"
    import config
    lang_suffix = "_zh" if getattr(config, "PROMPT_LANG", "en") == "zh" else ""
    return PROMPT_PROFILES[size + lang_suffix]
