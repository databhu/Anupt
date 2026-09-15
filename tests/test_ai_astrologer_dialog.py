"""
Regression test for _ai_astrologer_dialog() (app.py) — the ANUPT page's
reading-focused chat, converted from an inline tab (which showed a full
chat interface the moment the tab was clicked, no explicit "open chat"
action) to the same @st.dialog pattern already used and proven for
_ask_anupt_dialog(). Guards against the exact same bug class found there:
calling st.rerun() right after answering a question is the documented
way to CLOSE an @st.dialog, so doing that here would close the dialog on
every single message instead of showing the reply inside it.

Uses the same structural source-inspection approach as
test_ask_anupt_dialog.py, for the same reason: AppTest doesn't reproduce
a real browser's dialog-closes-on-rerun behavior, so this checks the
source directly rather than relying on a simulated click.
"""

import os
import re

_APP_PY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
with open(_APP_PY_PATH) as f:
    _APP_SOURCE = f.read()


def _extract_function_source(source: str, func_name: str) -> str:
    """Grabs one top-level function's source by name. Stops at the first
    non-blank line back at column 0 after the def line — not just the
    next `def` — since a function isn't always immediately followed by
    another one; here it's followed by plain top-level script code,
    and stopping only at the next `def` would swallow the entire rest
    of the file (and every unrelated st.rerun() in it) into the
    'extracted' function source."""
    start_match = re.search(rf"^def {re.escape(func_name)}\(", source, re.MULTILINE)
    assert start_match, f"could not find function {func_name} in app.py"
    start = start_match.start()
    remainder = source[start:]
    lines = remainder.splitlines(keepends=True)
    end_offset = len(remainder)
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "":
            continue
        if not line[0].isspace():
            end_offset = sum(len(l) for l in lines[:i])
            break
    return remainder[:end_offset]


def _code_only(source: str) -> str:
    """Strips comment lines AND the function's own docstring before
    searching for a real call — a naive substring search over raw source
    can false-positive on either a comment or a docstring that merely
    mentions the string in prose (this exact mistake was found and fixed
    for the comment case in test_ask_anupt_dialog.py once already; this
    function's own docstring explaining why it avoids st.rerun() hits
    the same trap for the docstring case)."""
    no_docstring = re.sub(r'"""(?:[^"]|"(?!""))*"""', "", source, count=1, flags=re.DOTALL)
    return "\n".join(line.split("#", 1)[0] for line in no_docstring.splitlines())


class TestAiAstrologerDialogDoesNotCloseOnMessage:
    def test_dialog_function_exists_and_is_decorated(self):
        assert '@st.dialog("💬 Ask AI Astrologer")' in _APP_SOURCE
        assert "def _ai_astrologer_dialog():" in _APP_SOURCE

    def test_message_handling_block_does_not_call_rerun(self):
        dialog_source = _extract_function_source(_APP_SOURCE, "_ai_astrologer_dialog")
        assert "st.rerun()" not in _code_only(dialog_source), (
            "st.rerun() anywhere in this dialog's message-handling closes the dialog on "
            "every single question (documented Streamlit behavior) — the reply already "
            "renders correctly without it, since st.chat_message + st.write happen inline "
            "during the same script execution"
        )

    def test_trigger_button_is_a_plain_button_not_the_chat_itself(self):
        # The ANUPT page's tab_chat block must show a trigger button, not
        # an inline st.chat_input — that inline pattern (a full chat box
        # appearing the instant the tab is clicked, no separate "open
        # chat" action) is exactly what this change replaces.
        tab_chat_match = re.search(r"with tab_chat:(.*?)(?=\n# -{5,}|\nstyling\.bottom_nav)",
                                    _APP_SOURCE, re.DOTALL)
        assert tab_chat_match, "could not locate the tab_chat block"
        tab_chat_source = tab_chat_match.group(1)
        assert "st.chat_input(" not in tab_chat_source
        assert 'st.button("💬 Open AI Astrologer chat"' in tab_chat_source
        assert "_ai_astrologer_dialog()" in tab_chat_source
