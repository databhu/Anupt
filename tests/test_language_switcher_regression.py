"""
Regression test for a real bug found during manual testing: the topbar's
language selectbox got its first-ever render while the profile was still
empty (mid-onboarding), locking its Streamlit widget state to "English".
On the very next rerun after the onboarding form correctly saved a
different language, the topbar compared its stale locked-in value against
the freshly-saved one, concluded the user had just changed it, and
silently overwrote the correct save back to English.

The fix (utils/styling.py's top_bar()) tracks the last externally-known
language and explicitly re-syncs the widget's stored value whenever it
changes for a reason other than that specific widget being touched. This
test exercises the exact sequence that exposed the bug — select a
language before a profile exists, complete onboarding, then trigger a
SUBSEQUENT rerun — and asserts the saved language survives.
"""

import logging
from datetime import date, time

import pytest

logging.disable(logging.CRITICAL)

from streamlit.testing.v1 import AppTest

import auth.store as store

pytestmark = pytest.mark.skipif(
    __import__("os").environ.get("DATABASE_URL") is None,
    reason="requires a live Postgres connection (DATABASE_URL)",
)


def _complete_onboarding_with_pending_language(username: str, lang_code: str) -> AppTest:
    store.create_user(username, "password123")
    uid = store.verify_user(username, "password123")

    at = AppTest.from_file("../app.py")
    at.run(timeout=30)
    at.session_state["pending_language"] = lang_code
    at.session_state["user_id"] = uid
    at.session_state["username"] = username
    at.run(timeout=30)

    at.text_input(key="pf_name").set_value("Regression Test")
    at.date_input(key="pf_dob").set_value(date(1995, 8, 8))
    at.time_input(key="pf_time").set_value(time(14, 30))
    at.session_state["selected_birth_place"] = {
        "label": "Mumbai, Maharashtra, India", "name": "Mumbai", "admin1": "Maharashtra",
        "country": "India", "latitude": 19.076, "longitude": 72.8777, "timezone": "Asia/Kolkata",
    }
    at.run(timeout=30)
    for b in at.button:
        if "Save profile" in (b.label or ""):
            b.click()
            break
    at.run(timeout=30)
    return at


class TestTopbarLanguageDoesNotRegressAfterSave:
    def test_hindi_survives_a_subsequent_rerun(self):
        store.init_db()
        at = _complete_onboarding_with_pending_language("regr_hi_test", "hi")
        assert len(at.exception) == 0

        uid = at.session_state["user_id"]
        assert store.get_profile(uid)["language"] == "hi"

        # The exact scenario that exposed the bug: a subsequent, unrelated
        # rerun (here, navigating to a different page) must not silently
        # revert the just-saved language.
        at.session_state["nav"] = "Astrology"
        at.run(timeout=30)
        assert len(at.exception) == 0
        assert store.get_profile(uid)["language"] == "hi", (
            "language was silently overwritten by a subsequent rerun"
        )

    def test_marathi_survives_a_subsequent_rerun(self):
        store.init_db()
        at = _complete_onboarding_with_pending_language("regr_mr_test", "mr")
        uid = at.session_state["user_id"]
        assert store.get_profile(uid)["language"] == "mr"

        at.session_state["nav"] = "Numerology"
        at.run(timeout=30)
        assert store.get_profile(uid)["language"] == "mr", (
            "language was silently overwritten by a subsequent rerun"
        )

    def test_multiple_subsequent_reruns_all_preserve_the_language(self):
        store.init_db()
        at = _complete_onboarding_with_pending_language("regr_multi_test", "hi")
        uid = at.session_state["user_id"]

        for nav in ["Astrology", "Numerology", "Palmistry", "Home"]:
            at.session_state["nav"] = nav
            at.run(timeout=30)
            assert store.get_profile(uid)["language"] == "hi", f"reverted after navigating to {nav}"
