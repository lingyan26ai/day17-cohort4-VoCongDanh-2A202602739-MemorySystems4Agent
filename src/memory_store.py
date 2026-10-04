import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

def estimate_tokens(text: str) -> int:
    # ponytail: character heuristic; a provider tokenizer is needed for billing accuracy.
    return math.ceil(len(text.strip()) / 4)

@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        if not user_id.strip():
            raise ValueError("user_id cannot be empty")
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)[:64]
        digest = hashlib.sha256(user_id.encode()).hexdigest()[:12]
        return self.root_dir / f"{slug}-{digest}" / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else "# User profile\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        text = self.read_text(user_id)
        if not search_text or search_text not in text or search_text == replacement:
            return False
        self.write_text(user_id, text.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return dict(re.findall(r"^- (\w+): (.+)$", self.read_text(user_id), re.MULTILINE))

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        facts = self.facts(user_id)
        if facts.get(key) == value:
            return
        facts[key] = value
        self.write_text(user_id, "# User profile\n\n" + "".join(f"- {k}: {v}\n" for k, v in facts.items()))

def extract_profile_updates(message: str) -> dict[str, str]:
    # ponytail: explicit Vietnamese assertions only; wider language needs an extractor model.
    facts = {}
    for sentence in re.findall(r"[^.!?\n]+[.!?]?", message):
        text = sentence.strip()
        # In a correction, the assertion after 'nhưng' replaces the earlier claim.
        text = re.split(r",?\s+nhưng\s+(?:thực ra\s+)?", text, flags=re.I)[-1]
        text = re.split(r"\s+dù trước đó\b", text, flags=re.I)[0]
        if not text or text.endswith("?") or re.search(r"\b(nếu|đừng|lúc đầu|trước đó|câu đùa)\b", text, re.I):
            continue
        if re.search(r"nhắc lại|nhớ lại|là gì|con gì|tên gì|ở đâu", text, re.I):
            continue
        text = text.rstrip(".!")
        patterns = {
            "name": r"(?:mình|tôi)\s+(?:tên là|tên)\s+([^,;:]+)",
            "location": r"(?:mình\s+(?:(?:vẫn|đang|hiện|hiện tại)\s+)*(?:ở|làm việc ở)|hiện ở|nơi ở hiện tại là)\s+([^,;:]+)",
            "profession": r"(?:đang làm|giờ chuyển sang|mình làm|nghề nghiệp hiện tại vẫn là|nghề nghiệp thì vẫn là|nghề\s+(?!nghiệp))\s*([\w -]+?engineer)\b",
            "drink": r"đồ uống yêu thích\s+(?:là|của mình là)\s+([^,;:]+)",
            "food": r"món ăn yêu thích\s+(?:là|của mình là)\s+([^,;:]+)",
            "pet": r"mình nuôi\s+([^,;:]+)",
        }
        for key, pattern in patterns.items():
            matches = list(re.finditer(pattern, text, re.I))
            if matches:
                value = matches[-1].group(1).strip()
                if key == "location":
                    value = re.split(r"\s+(?:và|chứ|trong|vài|để|mỗi|dù|cho)\b", value, maxsplit=1)[0]
                    # Ignore temporary venues (e.g. a cafe), keep explicit place names.
                    if not value or not value[0].isupper():
                        continue
                if key == "name":
                    value = re.split(r"\s+(?:nghề|hiện|và)\b", value, maxsplit=1)[0]
                facts[key] = value
        if re.search(r"(?:mình thích|mình quan tâm|mình đang quan tâm|dài hạn:)", text, re.I):
            interests = [word for word in ("Python", "AI", "MLOps") if re.search(rf"\b{word}\b", text, re.I)]
            if interests:
                facts["interests"] = ", ".join(interests)
        if re.search(r"(?:mình muốn|mình thích|hãy trả lời|style trả lời|cách giải thích)", text, re.I) and re.search(r"ngắn|bullet", text, re.I):
            style = ["ngắn gọn"]
            if "bullet" in text.lower():
                style.append("3 bullet" if re.search(r"3\s+bullet", text, re.I) else "bullet")
            if "ví dụ" in text.lower():
                style.append("có ví dụ thực tế / thực chiến")
            if "trade-off" in text.lower():
                style.append("so sánh trade-off")
            facts["style"] = ", ".join(style)
    return facts

def offline_response(facts: dict[str, str], message: str) -> str:
    if "?" in message or re.search(r"nhắc lại|tóm tắt|nhớ lại", message, re.I):
        if not facts:
            return "Mình chưa có thông tin về bạn trong thread này."
        labels = {"name": "Tên", "location": "Nơi ở", "profession": "Nghề nghiệp", "drink": "Đồ uống", "food": "Món ăn", "pet": "Thú cưng", "interests": "Quan tâm", "style": "Style"}
        items = [f"{labels.get(k, k)}: {v}" for k, v in facts.items()]
        if "3 bullet" in facts.get("style", ""):
            return "\n".join("- " + "; ".join(items[i::3]) for i in range(3) if items[i::3])
        return "; ".join(items) + "."
    return "Đã ghi nhận thông tin bạn chia sẻ."

def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    facts = {}
    snippets = []
    for message in messages:
        if message["role"] == "user":
            facts.update(extract_profile_updates(message["content"]))
            snippets.append(message["content"][:160])
    return ("; ".join(f"{key}: {value}" for key, value in facts.items()) + "\n" + "\n".join(snippets[-max_items:])).strip()

@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self):
        if self.threshold_tokens <= 0 or self.keep_messages < 1:
            raise ValueError("Invalid compact settings")

    def append(self, thread_id: str, role: str, content: str) -> None:
        context = self.context(thread_id)
        messages = context["messages"]
        messages.append({"role": role, "content": content})
        tokens = estimate_tokens(context["summary"]) + sum(estimate_tokens(m["content"]) for m in messages)
        if tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            summary = summarize_messages(messages[:-self.keep_messages])
            budget = max(160, self.threshold_tokens * 2)
            context["summary"] = (context["summary"] + "\n" + summary)[-budget:]
            context["messages"] = messages[-self.keep_messages:]
            context["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return self.context(thread_id)["compactions"]
