from dataclasses import replace
from pathlib import Path
import pytest
from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import load_conversations, recall_points, run_agent_benchmark
from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore, extract_profile_updates
from model_provider import normalize_provider

def make_config(tmp_path: Path):
    config = load_config(tmp_path)
    return replace(config, compact_threshold_tokens=200, compact_keep_messages=2)

def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.file_size("alice") == 0
    path = store.write_text("alice", "# User\nTên: An\nTên: An\n")
    assert path.name == "User.md"
    assert store.read_text("alice").count("An") == 2
    assert store.edit_text("alice", "An", "Bình")
    assert store.read_text("alice").count("An") == 1
    assert not store.edit_text("alice", "missing", "x")
    assert store.file_size("alice") == len(store.read_text("alice").encode("utf-8"))
    assert store.path_for("../../outside").resolve().is_relative_to(store.root_dir.resolve())
    assert store.path_for("a/b") != store.path_for("a_b")

def test_compact_trigger(tmp_path: Path) -> None:
    memory = CompactMemoryManager(100, 2)
    memory.append("one", "user", "Mình tên là An. " + "Ngữ cảnh dài. " * 70)
    memory.append("one", "assistant", "Đã ghi nhận.")
    memory.append("one", "user", "Tin mới.")
    context = memory.context("one")
    assert memory.compaction_count("one") == 1
    assert "An" in context["summary"]
    assert [m["content"] for m in context["messages"]] == ["Đã ghi nhận.", "Tin mới."]
    assert memory.context("other")["messages"] == []

def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline, advanced = BaselineAgent(config, True), AdvancedAgent(config, True)
    for agent in (baseline, advanced):
        agent.reply("alice", "first", "Mình tên là An. Mình ở Huế.")
        assert "An" in agent.reply("alice", "first", "Mình tên gì?")["response"]
    assert "An" not in baseline.reply("alice", "second", "Mình tên gì?")["response"]
    restarted = AdvancedAgent(config, True)
    answer = restarted.reply("alice", "second", "Mình tên gì và ở đâu?")["response"]
    assert "An" in answer and "Huế" in answer
    assert "An" not in restarted.reply("bob", "third", "Mình tên gì?")["response"]

def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline, advanced = BaselineAgent(config, True), AdvancedAgent(config, True)
    for _ in range(20):
        for agent in (baseline, advanced):
            agent.reply("alice", "long", "Mình đang đọc tài liệu. " * 100)
    assert advanced.compaction_count("long") > 1
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")

def test_correction_noise_and_questions(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), True)
    messages = [
        "Mình tên là An. Mình ở Đà Nẵng và đang làm backend engineer.",
        "Giờ mình đang ở Huế chứ không còn ở Đà Nẵng. Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.",
        "Hà Nội chỉ là nơi mình đi họp. Chuyển sang product manager chỉ là câu đùa.",
        "Nếu nhắc lại nghề nghiệp, đừng nói backend engineer nữa nhé.",
    ]
    for message in messages:
        agent.reply("alice", "first", message)
    before = agent.profile_store.read_text("alice")
    agent.reply("alice", "new", "Mình tên gì và hiện ở đâu?")
    agent.reply("alice", "new", "Nhắc lại giúp mình: tên và mình nuôi con gì.")
    agent.reply("alice", "new", "Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.")
    agent.reply("alice", "new", "Hôm nay mình làm việc ở quán cà phê gần sông.")
    assert before == agent.profile_store.read_text("alice")
    facts = agent.profile_store.facts("alice")
    assert facts["location"] == "Huế"
    assert facts["profession"] == "MLOps engineer"
    assert "backend engineer" not in before and "Hà Nội" not in before
    assert extract_profile_updates("Bạn có biết DũngCT không?") == {}
    correction = extract_profile_updates("Lúc đầu mình nói hiện ở Huế, nhưng thực ra từ tuần này mình đang làm việc ở Đà Nẵng vài tháng.")
    assert correction["location"] == "Đà Nẵng"

def test_benchmark_datasets(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), compact_threshold_tokens=1200, compact_keep_messages=4)
    data = Path(__file__).resolve().parent.parent / "data"
    for filename in ("conversations.json", "advanced_long_context.json"):
        conversations = load_conversations(data / filename)
        suite = replace(config, state_dir=tmp_path / filename)
        baseline = run_agent_benchmark("Baseline", BaselineAgent(suite, True), conversations, suite)
        advanced = run_agent_benchmark("Advanced", AdvancedAgent(suite, True), conversations, suite)
        assert baseline.recall_score == 0
        assert advanced.recall_score > baseline.recall_score
        assert advanced.memory_growth_bytes > 0
        if filename == "advanced_long_context.json":
            assert advanced.compactions > 1
            assert advanced.prompt_tokens_processed < baseline.prompt_tokens_processed

def test_scoring_and_provider_validation():
    assert recall_points("An thích trà", ["An", "cà phê"]) == 0.5
    assert recall_points("An thích trà", ["An", "trà"]) == 1
    assert normalize_provider(" Anthorpic ") == "anthropic"
    for provider in ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter"):
        assert normalize_provider(provider) == provider
    with pytest.raises(ValueError):
        normalize_provider("invalid")
