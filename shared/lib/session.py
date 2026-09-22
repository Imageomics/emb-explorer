"""Streamlit session identification for log lines.

Streamlit can execute the same script twice for one interaction (a duplicated
click event, or a second browser tab), which shows up in the log as doubled
"PROJECTION START" banners with no way to tell the runs apart (#49). Tagging
long-running steps with the session id makes duplicates attributable.
"""

import threading
from typing import Optional


def current_session_id() -> Optional[str]:
    """Streamlit session id of the running script, or None outside Streamlit."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx(suppress_warning=True)
    except Exception:  # streamlit missing or API moved: never break logging
        return None
    return getattr(ctx, "session_id", None) if ctx else None


def session_tag() -> str:
    """Short, log-friendly tag such as ``session=1a2b3c4d thread=139872``
    (``session=none`` outside Streamlit). The thread id tells two concurrent
    script runs of one session apart from one run logging twice."""
    sid = current_session_id()
    session = f"session={sid[:8]}" if sid else "session=none"
    return f"{session} thread={threading.get_ident()}"
