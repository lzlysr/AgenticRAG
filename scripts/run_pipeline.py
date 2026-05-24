"""端到端：query → agent graph → answer"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_single(query: str, verbose: bool = True) -> dict:
    from agents.graph import run_query
    from llm.client import stats

    stats.reset()
    state = run_query(query)

    if verbose:
        print(f"\nQuery: {query}")
        print(f"Answer: {state.get('final_answer', '')}")
        print(f"Query Type: {state.get('query_type', '')}")
        print(f"Iterations: {state.get('iteration_count', 0)}")
        print(f"Tool Calls: {state.get('total_tool_calls', 0)}")
        print(f"LLM Stats: {stats.snapshot()}")
        print(f"\nExecution Trace:")
        for t in state.get("trace", []):
            print(f"  [{t.get('node', '?')}] {json.dumps({k: v for k, v in t.items() if k != 'node' and k != 'plan'}, ensure_ascii=False)}")

    return state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="Were Scott Derrickson and Ed Wood of the same nationality?")
    args = parser.parse_args()
    run_single(args.query)
