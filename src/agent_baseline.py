from dataclasses import dataclass, field
from typing import Any
from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates, offline_response
from model_provider import build_chat_model

SYSTEM_PROMPT = "Trả lời bằng tiếng Việt, dùng thông tin trong ngữ cảnh; nếu chưa biết thì nói rõ."

@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0

class BaselineAgent:
    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = estimate_tokens(SYSTEM_PROMPT) + sum(estimate_tokens(m["content"]) for m in session.messages)
        facts = {}
        for item in session.messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))
        if self.langchain_agent is None:
            answer = offline_response(facts, message)
        else:
            answer = self.langchain_agent.invoke([("system", SYSTEM_PROMPT)] + [(m["role"], m["content"]) for m in session.messages]).text
        session.messages.append({"role": "assistant", "content": answer})
        tokens = estimate_tokens(answer)
        session.token_usage += tokens
        session.prompt_tokens_processed += prompt_tokens
        return {"response": answer, "token_usage": tokens, "prompt_tokens": prompt_tokens}

    def _maybe_build_langchain_agent(self):
        if self.force_offline or (not self.config.model.api_key and self.config.model.provider != "ollama"):
            return None
        return build_chat_model(self.config.model)
