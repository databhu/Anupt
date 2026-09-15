"""
Regression test for a real bug found during manual browser testing:
_ask_anupt_dialog() (app.py) called st.rerun() right after processing a
chat query, which — per Streamlit's own documentation — is the
DOCUMENTED way to programmatically CLOSE an @st.dialog. That meant every
single chip click or typed question silently closed the dialog instead
of showing the response inside it.

This can't be verified with streamlit.testing.v1.AppTest, confirmed by
direct comparison: AppTest's simulation of a click inside a dialog did
not reproduce the same "dialog closes" behavior a real browser did,
making it an unreliable tool for catching this specific regression. This
test instead does a structural check of the source itself: the query-
processing block must NOT call st.rerun(), while the navigation button
handlers (which correctly SHOULD close the dialog, since they're leaving
it) still do.
"""

import os
import re

_APP_PY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
with open(_APP_PY_PATH) as f:
    _APP_SOURCE = f.read()


def _extract_function_source(source: str, func_name: str) -> str:
    """Grabs one top-level function's source by name, up to (but not
    including) the next top-level `def` — good enough for this file's
    consistent 4-space-indented function bodies."""
    start_match = re.search(rf"^def {re.escape(func_name)}\(", source, re.MULTILINE)
    assert start_match, f"could not find function {func_name} in app.py"
    start = start_match.start()
    next_def = re.search(r"^def \w+\(", source[start + 1:], re.MULTILINE)
    end = start + 1 + next_def.start() if next_def else len(source)
    return source[start:end]


class TestAskAnuptDialogDoesNotCloseOnQuery:
    def test_dialog_function_exists_and_is_decorated(self):
        assert "@st.dialog(" in _APP_SOURCE
        assert "def _ask_anupt_dialog():" in _APP_SOURCE

    def test_query_processing_block_does_not_call_rerun(self):
        dialog_source = _extract_function_source(_APP_SOURCE, "_ask_anupt_dialog")
        # The query-processing block runs from "if user_query:" up to the
        # start of the chat-history display loop that follows it.
        query_block_match = re.search(
            r"if user_query:(.*?)for i, turn in enumerate", dialog_source, re.DOTALL,
        )
        assert query_block_match, "could not locate the query-processing block"
        query_block = query_block_match.group(1)
        # Strip comment lines before searching: the block deliberately
        # explains its own reasoning in a comment that itself mentions the
        # string "st.rerun()" as prose ("Deliberately NOT calling
        # st.rerun() here...") — a naive substring search over the raw
        # block text would find that comment and incorrectly report a
        # call that was never actually made, so only real code lines
        # (anything before a '#') are checked here.
        code_only_lines = [line.split("#", 1)[0] for line in query_block.splitlines()]
        code_only = "\n".join(code_only_lines)
        assert "st.rerun()" not in code_only, (
            "st.rerun() inside the query-processing block closes the dialog on every "
            "interaction (this is documented Streamlit behavior, not a fluke) — the "
            "just-appended chat message already renders correctly without it, since "
            "the display loop runs later in the same function execution"
        )

    def test_navigation_button_handlers_still_close_the_dialog(self):
        # The "Open X" and disambiguation-option buttons correctly SHOULD
        # close the dialog (the user is navigating away from it) — this
        # confirms the fix didn't overcorrect by removing ALL reruns.
        dialog_source = _extract_function_source(_APP_SOURCE, "_ask_anupt_dialog")
        display_loop_match = re.search(r"for i, turn in enumerate.*", dialog_source, re.DOTALL)
        assert display_loop_match, "could not locate the chat-history display loop"
        display_loop = display_loop_match.group(0)
        rerun_count = display_loop.count("st.rerun()")
        assert rerun_count >= 2, (
            f"expected at least 2 st.rerun() calls in the display loop (Open-X button "
            f"and disambiguation-option button), found {rerun_count}"
        )
