LLaMA-Factory/data/lf_react.jsonl 和 data/financial_eval/sft_zh_llamafactory/lf_react_zh.jsonl 是一样的

dataset_info.json 两个注册其实也一样
旧注册:
financial_agent_react
  -> LLaMA-Factory/data/lf_react.jsonl
  -> 当前也是 797 条 clean 中文 ReAct 数据

新注册:
financial_agent_zh_react
  -> data/financial_eval/sft_zh_llamafactory/lf_react_zh.jsonl
  -> 797 条 clean 中文 ReAct 数据