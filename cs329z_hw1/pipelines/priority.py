"""One-call email priority classification."""

from __future__ import annotations

import json
import re
from datetime import datetime

from cs329z_hw1.types import Email, LMCallable

LABELS = ("urgent", "normal", "ignore")

_LABEL_WORD = re.compile(r"\b(urgent|normal|ignore)\b", re.IGNORECASE)
_CATEGORY_LINE = re.compile(
    r"^(?:category|label)\s*[:=]\s*(urgent|normal|ignore)\b",
    re.IGNORECASE | re.MULTILINE,
)
_REASON_LINE = re.compile(r"^reason\s*[:=]\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_JSON_CATEGORY = re.compile(
    r'"(?:category|label)"\s*:\s*"(urgent|normal|ignore)"',
    re.IGNORECASE,
)
_JSON_REASON = re.compile(r'"(?:reason|why)"\s*:\s*"((?:\\.|[^"\\])*)"', re.IGNORECASE)

_SYSTEM = """You classify one email for its recipient. Judge from the point of view of the person in the To field, at the time the email was sent. The question is what that person should do with the message, not how important the topic is to the company.

A working day is Monday through Friday, counted from the email's date.

Labels:
- urgent: BOTH are true. (1) The email asks the recipient to do something, or reports a live problem the recipient is responsible for (an outage, a failed payment, a customer or manager escalating an issue). (2) There is time pressure: the action is needed the same day, the next working day, or by a stated deadline no more than two working days away.
- normal: Legitimate work or personal mail written for this recipient or a small working group, with no deadline in the next two working days. This covers status updates and FYIs, ordinary questions, documents sent for review without a near deadline, meeting scheduling for later dates, replies in an ongoing discussion, and personal notes from friends or family.
- ignore: Mail the recipient can leave unread at no cost: advertising and spam from vendors, newsletters and news digests, announcements sent to large employee lists (events, training offerings, facilities notices), and automated notifications that do not require the recipient to do anything.

Decision rules:
- Wording does not decide the label. An email with "URGENT" in the subject that asks nothing of the recipient is not urgent. A calm email that needs a signature by 3:00 PM today is urgent.
- A request with a deadline more than two working days away, or with no deadline, is normal.
- Mass mailings and advertising are ignore even when they say "act now" or "important", because nothing is asked of this recipient in particular.
- An automated notice that asks this specific recipient to act (for example, an expense report waiting for their approval) is normal, or urgent if it states a deadline within two working days.
- A short personal or social message from someone the recipient knows is normal.

Reply with exactly two lines and nothing else:
category: urgent
reason: one sentence

The category line must be exactly one of urgent, normal, or ignore, in lower case.
The reason must be one sentence and must not contain the words urgent, normal, or ignore.
"""

_NO_REASON = "The model reply did not include a reason."


def classify_priority(email: Email, lm: LMCallable) -> dict:
    """Classify one email. Returns category and a non-empty reason."""
    try:
        reply = lm(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _user_message(email)},
            ]
        )
    except Exception as exc:
        return {"category": "unknown", "reason": f"The model call failed: {exc}"}
    category, reason = _parse(reply if isinstance(reply, str) else str(reply))
    return {"category": category, "reason": reason}


def _user_message(email: Email) -> str:
    recipients = email.get("to") or []
    if isinstance(recipients, list):
        to_line = ", ".join(str(addr) for addr in recipients)
    else:
        to_line = str(recipients)
    date = str(email.get("date") or "")
    weekday = _weekday(date)
    when = f"{date} ({weekday})" if weekday else date
    subject = email.get("subject") or "(no subject)"
    body = email.get("body") or ""
    return (
        "Classify this email.\n\n"
        f"From: {email.get('from') or ''}\n"
        f"To: {to_line}\n"
        f"Date: {when}\n"
        f"Subject: {subject}\n"
        f"\n{body}"
    )


def _weekday(date: str) -> str:
    try:
        return datetime.strptime(date[:10], "%Y-%m-%d").strftime("%A")
    except ValueError:
        return ""


def _parse(reply: str) -> tuple[str, str]:
    text = reply or ""
    category, reason = _structured(text)
    if category is None:
        found: list[str] = []
        for match in _LABEL_WORD.finditer(text):
            label = match.group(1).lower()
            if label not in found:
                found.append(label)
        category = found[0] if len(found) == 1 else "unknown"
    if category not in (*LABELS, "unknown"):
        category = "unknown"
    if not reason or not str(reason).strip():
        reason = text.strip() or _NO_REASON
    return category, str(reason).strip()


def _structured(text: str) -> tuple[str | None, str]:
    """A category field wins over other label words in the reply."""
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        category, reason = _from_json(stripped)
        if category is not None:
            return category, reason

    match = _JSON_CATEGORY.search(text)
    if match and stripped.startswith("{"):
        reason_match = _JSON_REASON.search(text)
        reason = ""
        if reason_match:
            try:
                reason = json.loads(f'"{reason_match.group(1)}"')
            except json.JSONDecodeError:
                reason = reason_match.group(1)
        return match.group(1).lower(), reason.strip()

    line = _CATEGORY_LINE.search(text)
    if line is None:
        return None, ""
    reason_line = _REASON_LINE.search(text)
    reason = reason_line.group(1).strip() if reason_line else ""
    return line.group(1).lower(), reason


def _from_json(text: str) -> tuple[str | None, str]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None, ""
    if not isinstance(obj, dict):
        return None, ""
    raw = str(obj.get("category") or obj.get("label") or "").strip().lower()
    if raw not in LABELS:
        return None, ""
    reason = obj.get("reason")
    if reason is None:
        reason = obj.get("why") or ""
    return raw, str(reason).strip()
