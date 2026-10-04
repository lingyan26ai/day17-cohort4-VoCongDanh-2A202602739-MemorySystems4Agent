import json
import tempfile
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config

@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int

def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))

def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 0.0
    normalized = unicodedata.normalize("NFC", answer).casefold()
    return sum(unicodedata.normalize("NFC", fact).casefold() in normalized for fact in expected) / len(expected)

def heuristic_quality(answer: str, expected: list[str]) -> float:
    # ponytail: recall-based proxy, not an independent LLM/human quality assessment.
    return recall_points(answer, expected) * (1.0 if len(answer) <= 1200 else 0.8)

def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    users = {c["user_id"] for c in conversations}
    initial = sum(agent.memory_file_size(u) for u in users) if hasattr(agent, "memory_file_size") else 0
    threads = []
    recalls, qualities = [], []
    for index, conversation in enumerate(conversations):
        user = conversation["user_id"]
        thread = f"conversation-{index}-{conversation['id']}"
        threads.append(thread)
        for message in conversation["turns"]:
            agent.reply(user, thread, message)
        # Evaluate immediately, before later conversations can reveal future corrections.
        for q_index, question in enumerate(conversation["recall_questions"]):
            recall_thread = f"recall-{index}-{q_index}"
            threads.append(recall_thread)
            answer = agent.reply(user, recall_thread, question["question"])["response"]
            recalls.append(recall_points(answer, question["expected_contains"]))
            qualities.append(heuristic_quality(answer, question["expected_contains"]))
    final = sum(agent.memory_file_size(u) for u in users) if hasattr(agent, "memory_file_size") else 0
    return BenchmarkRow(agent_name, sum(agent.token_usage(t) for t in threads),
                        sum(agent.prompt_token_usage(t) for t in threads),
                        sum(recalls) / len(recalls) if recalls else 0,
                        sum(qualities) / len(qualities) if qualities else 0,
                        final - initial, sum(agent.compaction_count(t) for t in threads))

def format_rows(rows: list[BenchmarkRow]) -> str:
    lines = ["| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row.agent_name} | {row.agent_tokens_only} | {row.prompt_tokens_processed} | {row.recall_score:.1%} | {row.response_quality:.1%} | {row.memory_growth_bytes} | {row.compactions} |")
    return "\n".join(lines)

def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    print("Offline mode: token estimates; response quality is a recall/length heuristic. Includes recall turns.")
    for title, filename in (("Standard Benchmark", "conversations.json"), ("Long-Context Stress Benchmark", "advanced_long_context.json")):
        conversations = load_conversations(config.data_dir / filename)
        # Fresh isolated state on every run; never remove an existing user profile.
        with tempfile.TemporaryDirectory(prefix="benchmark-", dir=config.state_dir) as directory:
            suite_config = replace(config, state_dir=Path(directory))
            rows = [run_agent_benchmark("Baseline", BaselineAgent(suite_config, force_offline=True), conversations, suite_config),
                    run_agent_benchmark("Advanced", AdvancedAgent(suite_config, force_offline=True), conversations, suite_config)]
        print("\n" + title)
        print(format_rows(rows))

if __name__ == "__main__":
    main()
