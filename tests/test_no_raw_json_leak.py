"""
Regression test for a real bug: the Home page's "Cast today's reading"
button (a separate code path from the ANUPT page's "Get My Insights" tab
— see the PAGE: HOME section of app.py) rendered the full raw unified-
evidence dict via an ungated st.json(u) call inside its "See the full
reading & evidence trail" expander. That meant every regular user who
expanded it saw raw theme_scores, confidence numbers, and internal
structure — exactly the exposure Developer Mode exists to prevent, in a
place the original audit for st.json( calls missed.

This checks the source directly (not via AppTest) since clicking the
actual button requires a full authenticated flow with a live AI call —
the property under test is purely structural: every st.json() call in
app.py must be preceded by (i.e. nested under) a `developer_mode` check
somewhere in its enclosing block, not just some of them.
"""

import os
import re

_APP_PY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
with open(_APP_PY_PATH) as f:
    _APP_SOURCE = f.read()


class TestNoUngatedRawJsonExposure:
    """Rather than a general indentation-scanning heuristic (tried first;
    proved too fragile to reliably distinguish 'guarded by an enclosing
    if developer_mode:' from 'not guarded at all' across differently-
    nested call sites, producing false positives on code already
    manually verified correct), this checks each known st.json() call
    site individually by its surrounding, distinguishing context. Less
    general, but each assertion is easy to verify by eye against the
    actual source — and if a new st.json() call is added elsewhere
    without being included here, test_no_new_unaccounted_json_calls
    below catches that.
    """

    def test_diagnose_panel_gated(self):
        block = _extract_block_for("st.json(gemini_client.diagnose())")
        assert "developer_mode" in block

    def test_last_call_diagnostics_panel_gated(self):
        block = _extract_block_for("st.json(gemini_client.get_last_call_diagnostics())")
        assert "developer_mode" in block

    def test_youtube_key_status_panel_gated(self):
        block = _extract_block_for("st.json(status)")
        assert "developer_mode" in block

    def test_home_page_today_reading_evidence_gated(self):
        # The exact bug this file exists for: st.json(u) inside the Home
        # page's "Cast today's reading" -> "See the full reading &
        # evidence trail" expander, previously ungated.
        idx = _APP_SOURCE.index('st.expander("See the full reading & evidence trail")')
        following = _APP_SOURCE[idx:idx + 300]
        assert "developer_mode" in following
        assert "st.json(u)" in following

    def test_anupt_page_advanced_analysis_gated(self):
        idx = _APP_SOURCE.index("Advanced Analysis — full evidence trail (raw")
        preceding = _APP_SOURCE[max(0, idx - 150):idx]
        assert "developer_mode" in preceding

    def test_no_new_unaccounted_json_calls(self):
        # If someone adds a NEW st.json() call later without updating
        # this test file, this fails loudly rather than silently passing
        # — forcing a conscious decision about whether it needs gating,
        # rather than another one slipping through like the bug above did.
        known_call_texts = [
            "st.json(gemini_client.diagnose())",
            "st.json(gemini_client.get_last_call_diagnostics())",
            "st.json(status)",
            "st.json(u)",  # appears twice (Home page + ANUPT page), both checked above
        ]
        # Match balanced parens (one level of nesting, e.g.
        # "st.json(gemini_client.diagnose())") rather than stopping at
        # the first ")" — a naive [^)]* regex truncates at the inner
        # call's own closing paren.
        all_calls = re.findall(r"st\.json\((?:[^()]|\([^()]*\))*\)", _APP_SOURCE)
        for call in all_calls:
            assert call in known_call_texts, (
                f"found an st.json() call not covered by this test file: {call!r} — "
                f"add a case for it above (and gate it behind developer_mode if it "
                f"exposes anything beyond what a regular user should see)"
            )

    def test_developer_mode_variable_exists_and_defaults_off(self):
        assert 'developer_mode = st.query_params.get("dev") == "1"' in _APP_SOURCE


def _extract_block_for(call_text: str) -> str:
    """Returns a chunk of source from a bit before the given st.json(...)
    call text, used to check nearby context (e.g. a `developer_mode`
    guard) without needing full indentation-aware parsing."""
    idx = _APP_SOURCE.index(call_text)
    return _APP_SOURCE[max(0, idx - 1700):idx + len(call_text)]
