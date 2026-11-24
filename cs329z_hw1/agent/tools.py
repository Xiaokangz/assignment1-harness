"""The Cardinal toolset. The email archive is opened the first time a tool needs it."""

from __future__ import annotations

import threading

from cs329z_hw1.cardinal import CARDINAL_TOOLS, cardinal_spec
from cs329z_hw1.types import AgentConfig, ToolSpec

_SNIPPET_CHARS = 400
_archive_lock = threading.Lock()
_archive_cache: dict = {}

_DESCRIPTIONS = {
    "search_emails": (
        "Search the email archive. query is the search text. "
        "k is how many matching emails to return."
    ),
    "read_email": "Read one archived email. email_id is its id, such as em-00001.",
    "email_qa": "Answer one question from the email archive. question is the question.",
    "daily_digest": "Write the morning summary for one UTC day. date is YYYY-MM-DD.",
    "search_docs": (
        "Search the company documents. query is the search text. "
        "k is how many documents to return."
    ),
    "read_doc": (
        "Read a window of one company document. doc_id identifies the document. "
        "start_line is the first line to return, counting from 1."
    ),
    "run_terminal": "Run one shell command in the workspace. cmd is the command line.",
    "remember": "Store one fact for later conversations. fact is the sentence to store.",
    "recall": "Find facts stored earlier. query describes what to look for.",
    "ask_user": "Ask the person a question and return their answer. question is what to ask.",
}


def build_tools(tools: list[ToolSpec] | None, config: AgentConfig) -> list[ToolSpec]:
    """The tools for one session.

    ``None`` selects the whole Cardinal set. An explicit list is kept as
    given, with ask_user added when a user is configured and remember plus
    recall added when a memory directory is configured.
    """
    if tools is None:
        return [_spec(name, config) for name in CARDINAL_TOOLS]
    chosen = list(tools)
    names = {tool.name for tool in chosen}
    if config.user is not None and "ask_user" not in names:
        chosen.append(_spec("ask_user", config))
        names.add("ask_user")
    if config.memory_dir is not None:
        for name in ("remember", "recall"):
            if name not in names:
                chosen.append(_spec(name, config))
                names.add(name)
    return chosen


def _spec(name: str, config: AgentConfig) -> ToolSpec:
    return cardinal_spec(name, _DESCRIPTIONS[name], _FUNCTIONS[name](config))


def _archive():
    """Emails, an id lookup, and the BM25 index. Built once per process."""
    cached = _archive_cache.get("index")
    if cached is not None:
        return _archive_cache["emails"], _archive_cache["by_id"], cached
    with _archive_lock:
        cached = _archive_cache.get("index")
        if cached is None:
            from cs329z_hw1 import data
            from cs329z_hw1.pipelines.bm25 import build_index

            emails = data.load_emails()
            _archive_cache["emails"] = emails
            _archive_cache["by_id"] = {email["id"]: email for email in emails}
            _archive_cache["index"] = build_index([data.email_text(email) for email in emails])
        return _archive_cache["emails"], _archive_cache["by_id"], _archive_cache["index"]


def _search_emails(config: AgentConfig):
    def search_emails(query: str, k: int = 5) -> str:
        from cs329z_hw1.pipelines.bm25 import search

        emails, _by_id, index = _archive()
        hits = search(index, query, k)
        if not hits:
            return "No matching emails."
        blocks = []
        for doc_id, _score in hits:
            email = emails[doc_id]
            snippet = " ".join((email.get("body") or "").split())
            if len(snippet) > _SNIPPET_CHARS:
                snippet = snippet[:_SNIPPET_CHARS] + "..."
            blocks.append(
                f"{email['id']}  {email.get('date', '')}  from {email.get('from', '')}\n"
                f"Subject: {email.get('subject') or '(no subject)'}\n"
                f"{snippet}"
            )
        return "\n\n".join(blocks)

    return search_emails


def _read_email(config: AgentConfig):
    def read_email(email_id: str) -> str:
        _emails, by_id, _index = _archive()
        email = by_id.get(email_id)
        if email is None:
            return f"No email with id {email_id}."
        recipients = email.get("to") or []
        to_line = ", ".join(str(addr) for addr in recipients)
        return (
            f"Id: {email['id']}\n"
            f"From: {email.get('from', '')}\n"
            f"To: {to_line}\n"
            f"Date: {email.get('date', '')}\n"
            f"Subject: {email.get('subject') or '(no subject)'}\n\n"
            f"{email.get('body') or ''}"
        )

    return read_email


def _email_qa(config: AgentConfig):
    def email_qa(question: str) -> str:
        from cs329z_hw1.pipelines.bm25 import search
        from cs329z_hw1.pipelines.email_qa import answer_question

        emails, _by_id, index = _archive()

        def find(query: str, k: int = 5):
            return [emails[doc_id] for doc_id, _score in search(index, query, k)]

        result = answer_question(question, find, config.aux_lm)
        answer = result["answer"] if result["answer"] is not None else "No answer found."
        support = ", ".join(result["support"]) or "(none)"
        return f"{answer}\nSupport: {support}"

    return email_qa


def _daily_digest(config: AgentConfig):
    def daily_digest(date: str) -> str:
        from cs329z_hw1 import data
        from cs329z_hw1.pipelines.digest import write_digest

        return write_digest(data.emails_on(date), config.aux_lm)

    return daily_digest


def _search_docs(config: AgentConfig):
    def search_docs(query: str, k: int = 5) -> str:
        return "Document search is not available yet."

    return search_docs


def _read_doc(config: AgentConfig):
    def read_doc(doc_id: str, start_line: int = 1) -> str:
        return "Document reading is not available yet."

    return read_doc


def _run_terminal(config: AgentConfig):
    def run_terminal(cmd: str) -> str:
        return "The terminal is not available yet."

    return run_terminal


def _remember(config: AgentConfig):
    def remember(fact: str) -> str:
        return "Memory storage is not available yet."

    return remember


def _recall(config: AgentConfig):
    def recall(query: str) -> str:
        return "Memory lookup is not available yet."

    return recall


def _ask_user(config: AgentConfig):
    def ask_user(question: str) -> str:
        user = config.user
        if user is None:
            return "No person is available to answer."
        return str(user.ask(question))

    return ask_user


_FUNCTIONS = {
    "search_emails": _search_emails,
    "read_email": _read_email,
    "email_qa": _email_qa,
    "daily_digest": _daily_digest,
    "search_docs": _search_docs,
    "read_doc": _read_doc,
    "run_terminal": _run_terminal,
    "remember": _remember,
    "recall": _recall,
    "ask_user": _ask_user,
}
