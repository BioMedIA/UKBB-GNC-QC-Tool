# Streamlit >= 0.65.2
# Original script: https://gist.github.com/tvst/ef477845ac86962fa4c92ec6a72bb5bd

from streamlit.script_request_queue import RerunData
from streamlit.script_runner import RerunException
from streamlit.server.server import Server
import streamlit.report_thread as ReportThread


def rerun():
    """Rerun a Streamlit app from the top!"""
    session = _get_session_object()
    client_state = session._client_state
    # session.request_rerun(client_state)
    session.request_rerun()


def _get_session_object():
    # Hack to get the session object from Streamlit.

    ctx = ReportThread.get_report_ctx()

    session = None
    session_infos = Server.get_current()._session_info_by_id.values()

    for session_info in session_infos:
        s = session_info.session
        if not hasattr(s, '_main_dg') and s._uploaded_file_mgr == ctx.uploaded_file_mgr:
            session = s
    if session is None:
        raise RuntimeError(
            "Oh noes. Couldn't get your Streamlit Session object"
            'Are you doing something fancy with threads?')
    # Got the session object!

    return session
