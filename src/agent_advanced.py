from dataclasses import dataclass
from typing import Any
from agent_baseline import SYSTEM_PROMPT
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates, offline_response
from model_provider import build_chat_model

@dataclass
class AgentContext:
    user_id: str
    memory_path: str

class AdvancedAgent:
    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(self.config.compact_threshold_tokens, self.config.compact_keep_messages)
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        context = self.compact_memory.context(thread_id)
        if self.langchain_agent is None:
            answer = self._offline_response(user_id, thread_id, message)
        else:
            system = SYSTEM_PROMPT + "\n" + self.profile_store.read_text(user_id) + "\n" + context["summary"]
            answer = self.langchain_agent.invoke([("system", system)] + [(m["role"], m["content"]) for m in context["messages"]]).text
        self.compact_memory.append(thread_id, "assistant", answer)
        tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + tokens
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens
        return {"response": answer, "token_usage": tokens, "prompt_tokens": prompt_tokens}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        context = self.compact_memory.context(thread_id)
        return (estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(self.profile_store.read_text(user_id))
                + estimate_tokens(context["summary"]) + sum(estimate_tokens(m["content"]) for m in context["messages"]))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        context = self.compact_memory.context(thread_id)
        facts = extract_profile_updates(context["summary"])
        for item in context["messages"]:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))
        # The persistent profile carries the latest corrections across threads.
        facts.update(self.profile_store.facts(user_id))
        return offline_response(facts, message)

    def _maybe_build_langchain_agent(self):
        if self.force_offline or (not self.config.model.api_key and self.config.model.provider != "ollama"):
            return None
        return build_chat_model(self.config.model)
