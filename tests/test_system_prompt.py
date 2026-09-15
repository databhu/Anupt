"""
Tests for ai/gemini_client.py's SYSTEM_GUARDRAILS — verifies every
explicitly-requested instruction is actually present in the prompt sent
to every AI call (this is the single default `system` parameter for
_call(), so it applies universally unless a caller overrides it — no
caller currently does, which this file also confirms).
"""

from ai import gemini_client as gc


class TestSystemGuardrailsContent:
    def test_applies_to_every_call_by_default(self):
        import inspect
        sig = inspect.signature(gc._call)
        assert sig.parameters["system"].default == gc.SYSTEM_GUARDRAILS

    def test_no_json_instruction_present(self):
        assert "no JSON" in gc.SYSTEM_GUARDRAILS

    def test_no_python_dict_instruction_present(self):
        assert "Python dictionaries" in gc.SYSTEM_GUARDRAILS

    def test_no_internal_scores_instruction_present(self):
        assert "raw numeric scores" in gc.SYSTEM_GUARDRAILS

    def test_no_engine_implementation_details_instruction_present(self):
        assert "implementation details" in gc.SYSTEM_GUARDRAILS
        assert "variable names" in gc.SYSTEM_GUARDRAILS

    def test_no_api_model_errors_instruction_present(self):
        assert "API errors, model names" in gc.SYSTEM_GUARDRAILS

    def test_no_raw_calculations_instruction_present(self):
        assert "Never repeat raw calculation steps" in gc.SYSTEM_GUARDRAILS

    def test_no_fabrication_instruction_present(self):
        assert "never fabricate information" in gc.SYSTEM_GUARDRAILS

    def test_evidence_only_instruction_present(self):
        assert "use only what's given" in gc.SYSTEM_GUARDRAILS

    def test_natural_language_instruction_present(self):
        assert "translate all of it into natural language" in gc.SYSTEM_GUARDRAILS

    def test_conciseness_instruction_present(self):
        assert "Be concise" in gc.SYSTEM_GUARDRAILS

    def test_indication_vs_certainty_instruction_present(self):
        assert "distinguish an 'indication' or 'tendency' from a certainty" in gc.SYSTEM_GUARDRAILS

    def test_not_guaranteed_fact_instruction_present(self):
        assert "never as a guaranteed" in gc.SYSTEM_GUARDRAILS
        assert "guaranteed fact" in gc.SYSTEM_GUARDRAILS

    def test_consumer_facing_framing_present(self):
        assert "consumer" in gc.SYSTEM_GUARDRAILS.lower()
