"""Streamlit session identification for log lines.

Streamlit can execute the same script twice for one interaction (a duplicated
click event, or a second browser tab), which shows up in the log as doubled
"PROJECTION START" banners with no way to tell the runs apart (#49). Tagging
long-running steps with the session id makes duplicates attributable.
"""

import threading
from contextlib import contextmanager
from typing import Iterator, Optional


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


def is_running(flag_key: str) -> bool:
    """True while a long-running action marked with :func:`single_run` is in flight."""
    import streamlit as st
    return bool(st.session_state.get(flag_key, False))


@contextmanager
def single_run(flag_key: str) -> Iterator[None]:
    """Mark a long-running action in session state for its duration.

    A duplicated click starts a second script run of the same session while
    the first is still executing (#49); session state is shared between them,
    so the second run can see the flag and log the duplicate. It must not
    refuse to run: Streamlit stops the *older* run at its next Streamlit call
    once a newer one exists, so refusing the newer run leaves nothing running.
    The flag is always cleared, including when Streamlit stops the run.
    """
    import streamlit as st
    st.session_state[flag_key] = True
    try:
        yield
    finally:
        st.session_state[flag_key] = False
