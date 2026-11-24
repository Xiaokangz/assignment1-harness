"""Morning digest: classify the day's mail, then one call writes the summary."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from cs329z_hw1.pipelines.priority import classify_priority
from cs329z_hw1.types import Email, LMCallable

_EMPTY_DAY = "No messages arrived."
_NOTHING_TO_REPORT = "No messages need attention this morning."
_CLASSIFY_WORKERS = 8
_BODY_CHARS = 800

_SYSTEM = """You write the morning email digest for one busy employee.

The emails below are already classified. Use only those emails.

- Give every urgent email its own sentence. Copy the company, project, or site name from that email's subject exactly as written, including capitalization. A description that leaves the name out does not count.
- A normal email that asks the recipient to act today or on the next working day also gets its own sentence, with the name from its subject copied exactly.
- Summarize the remaining normal emails in a few sentences. Skip a normal email rather than inventing a name for it.
- Use only names that appear in the emails below. Do not add companies, products, events, or places from anywhere else.
- Write at most 180 words. A word is a whitespace-separated token, the same as len(text.split()). If you would go over, shorten the normal sentences first. Never drop an urgent name.
- Plain prose only. No heading, no bullet list, no label such as "urgent" or "normal".
"""


def write_digest(emails: list[Email], lm: LMCallable) -> str:
    """Return the morning summary for one day's emails."""
    if not emails:
        return _EMPTY_DAY

    labeled = _classify(emails, lm)
    kept = [(email, category) for email, category in labeled if category != "ignore"]
    if not kept:
        return _NOTHING_TO_REPORT

    chosen = _one_per_thread(kept)
    chosen.sort(key=lambda item: (0 if item[1] == "urgent" else 1, item[0].get("date") or ""))
    try:
        reply = lm(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _user_message(chosen)},
            ]
        )
    except Exception:
        return _NOTHING_TO_REPORT
    text = reply.strip() if isinstance(reply, str) else str(reply).strip()
    return text or _NOTHING_TO_REPORT


def _classify(emails: list[Email], lm: LMCallable) -> list[tuple[Email, str]]:
    def label(email: Email) -> tuple[Email, str]:
        category = classify_priority(email, lm).get("category")
        if category == "urgent":
            return email, "urgent"
        if category == "ignore":
            return email, "ignore"
        return email, "normal"

    with ThreadPoolExecutor(max_workers=_CLASSIFY_WORKERS) as pool:
        return list(pool.map(label, emails))


def _one_per_thread(labeled: list[tuple[Email, str]]) -> list[tuple[Email, str]]:
    chosen: dict[str, tuple[Email, str]] = {}
    for item in labeled:
        email, category = item
        key = str(email.get("thread_id") or email.get("id") or "")
        current = chosen.get(key)
        if current is None or _prefer(item, current):
            chosen[key] = item
    return list(chosen.values())


def _prefer(new: tuple[Email, str], old: tuple[Email, str]) -> bool:
    """True when new should replace old as the thread's representative."""
    new_email, new_category = new
    old_email, old_category = old
    new_rank = 0 if new_category == "urgent" else 1
    old_rank = 0 if old_category == "urgent" else 1
    if new_rank != old_rank:
        return new_rank < old_rank
    return (new_email.get("date") or "") > (old_email.get("date") or "")


def _user_message(labeled: list[tuple[Email, str]]) -> str:
    blocks = []
    for email, category in labeled:
        body = " ".join((email.get("body") or "").split())
        if len(body) > _BODY_CHARS:
            body = body[:_BODY_CHARS] + "..."
        subject = email.get("subject") or "(no subject)"
        blocks.append(
            f"[{category}]\n"
            f"From: {email.get('from') or ''}\n"
            f"Date: {email.get('date') or ''}\n"
            f"Subject: {subject}\n"
            f"Body: {body}"
        )
    return "Write the digest for these emails.\n\n" + "\n\n".join(blocks)
