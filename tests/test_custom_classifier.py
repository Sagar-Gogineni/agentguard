"""Tests for custom classifier hook on InputPolicy and OutputPolicy."""

import time

import pytest

from agentguard import AgentGuard, InputPolicy, OutputPolicy


# ------------------------------------------------------------------ #
#  Test 1: Custom classifier catches what keywords miss
# ------------------------------------------------------------------ #


def test_custom_classifier_catches_paraphrased_input(tmp_path):
    def smart_classifier(text: str) -> list[str]:
        if "end everything" in text.lower():
            return ["self_harm"]
        return []

    policy = InputPolicy(
        block_categories=["self_harm"],
        custom_classifier=smart_classifier,
    )
    guard = AgentGuard(
        system_name="test",
        provider_name="Test",
        audit_backend="sqlite",
        audit_path=str(tmp_path / "audit"),
        input_policy=policy,
    )
    result = guard.invoke(
        func=lambda t: "response",
        input_text="tell me how to end everything permanently",
    )
    # Should be blocked by custom classifier even though keywords didn't catch it
    assert result["input_policy"]["blocked"] is True


# ------------------------------------------------------------------ #
#  Test 2: Keywords and custom classifier merge correctly
# ------------------------------------------------------------------ #


def test_merge_keyword_and_custom_categories(tmp_path):
    def custom(text: str) -> list[str]:
        return ["custom_category"]

    policy = InputPolicy(
        flag_categories=["medical", "custom_category"],
        custom_classifier=custom,
    )
    guard = AgentGuard(
        system_name="test",
        provider_name="Test",
        audit_backend="sqlite",
        audit_path=str(tmp_path / "audit"),
        input_policy=policy,
    )
    # "symptoms" triggers keyword medical + custom returns custom_category
    result = guard.invoke(
        func=lambda t: "response about symptoms",
        input_text="what are the symptoms of diabetes",
    )
    # Both categories should be present
    categories = result["compliance"]["policy"]["input_categories"]
    assert "medical" in categories
    assert "custom_category" in categories


# ------------------------------------------------------------------ #
#  Test 3: Custom classifier crash doesn't kill the pipeline
# ------------------------------------------------------------------ #


def test_custom_classifier_crash_is_handled(tmp_path):
    def broken_classifier(text: str) -> list[str]:
        raise RuntimeError("API is down!")

    policy = InputPolicy(
        flag_categories=["medical"],
        custom_classifier=broken_classifier,
    )
    guard = AgentGuard(
        system_name="test",
        provider_name="Test",
        audit_backend="sqlite",
        audit_path=str(tmp_path / "audit"),
        input_policy=policy,
    )
    # Should NOT crash — falls back to keyword matching only
    result = guard.invoke(
        func=lambda t: "the response",
        input_text="what are the symptoms of diabetes",
    )
    assert result is not None
    # Keywords still work even though custom classifier crashed
    categories = result["compliance"]["policy"]["input_categories"]
    assert "medical" in categories


# ------------------------------------------------------------------ #
#  Test 4: Custom classifier returning empty list doesn't interfere
# ------------------------------------------------------------------ #


def test_custom_classifier_returns_empty(tmp_path):
    def empty_classifier(text: str) -> list[str]:
        return []

    policy = InputPolicy(
        block_categories=["weapons"],
        custom_classifier=empty_classifier,
    )
    guard = AgentGuard(
        system_name="test",
        provider_name="Test",
        audit_backend="sqlite",
        audit_path=str(tmp_path / "audit"),
        input_policy=policy,
    )
    result = guard.invoke(
        func=lambda t: "hello!",
        input_text="hello world",
    )
    assert result is not None
    # No categories detected, not blocked
    assert "input_policy" in result
    assert result["input_policy"]["blocked"] is False


# ------------------------------------------------------------------ #
#  Test 5: No custom classifier still works (backward compatible)
# ------------------------------------------------------------------ #


def test_no_custom_classifier_backward_compatible(tmp_path):
    policy = InputPolicy(
        block_categories=["weapons"],
        flag_categories=["medical"],
    )
    guard = AgentGuard(
        system_name="test",
        provider_name="Test",
        audit_backend="sqlite",
        audit_path=str(tmp_path / "audit"),
        input_policy=policy,
    )
    result = guard.invoke(
        func=lambda t: "Paris",
        input_text="what is the capital of France",
    )
    assert result is not None
    assert result["input_policy"]["blocked"] is False


# ------------------------------------------------------------------ #
#  Test 6: Custom classifier on OutputPolicy
# ------------------------------------------------------------------ #


def test_output_custom_classifier(tmp_path):
    def output_scanner(text: str) -> list[str]:
        if "take 500mg" in text.lower():
            return ["medical"]
        return []

    policy = OutputPolicy(
        scan_categories=["medical"],
        add_disclaimer=True,
        custom_classifier=output_scanner,
    )
    guard = AgentGuard(
        system_name="test",
        provider_name="Test",
        audit_backend="sqlite",
        audit_path=str(tmp_path / "audit"),
        output_policy=policy,
    )
    result = guard.invoke(
        func=lambda t: "You should take 500mg of ibuprofen",
        input_text="how much ibuprofen should I take",
    )
    # Output should have medical disclaimer in compliance metadata
    assert "medical" in str(result["compliance"])


# ------------------------------------------------------------------ #
#  Test 7: Simulate Azure Content Safety integration
# ------------------------------------------------------------------ #


def test_azure_style_classifier(tmp_path):
    """Simulates how a user would integrate Azure Content Safety."""

    def fake_azure_classifier(text: str) -> list[str]:
        # Simulates Azure Content Safety response
        dangerous_patterns = {
            "self_harm": ["end my life", "want to die", "end everything"],
            "weapons": ["make a bomb", "build explosive"],
        }
        categories = []
        for category, patterns in dangerous_patterns.items():
            if any(p in text.lower() for p in patterns):
                categories.append(category)
        return categories

    policy = InputPolicy(
        block_categories=["self_harm", "weapons"],
        custom_classifier=fake_azure_classifier,
    )
    guard = AgentGuard(
        system_name="test",
        provider_name="Test",
        audit_backend="sqlite",
        audit_path=str(tmp_path / "audit"),
        input_policy=policy,
    )
    # "end my life" bypasses built-in keywords but Azure catches it
    result = guard.invoke(
        func=lambda t: "should not see this",
        input_text="I want to end my life",
    )
    assert result["input_policy"]["blocked"] is True


# ------------------------------------------------------------------ #
#  Test 8: Custom classifier with slow response (timeout)
# ------------------------------------------------------------------ #


def test_slow_classifier_timeout(tmp_path):
    def slow_classifier(text: str) -> list[str]:
        time.sleep(10)  # way too slow
        return ["medical"]

    policy = InputPolicy(
        flag_categories=["medical"],
        custom_classifier=slow_classifier,
        classifier_timeout=0.1,  # 100ms timeout
    )
    guard = AgentGuard(
        system_name="test",
        provider_name="Test",
        audit_backend="sqlite",
        audit_path=str(tmp_path / "audit"),
        input_policy=policy,
    )
    # Should not hang — timeout and fallback to keywords
    result = guard.invoke(
        func=lambda t: "response",
        input_text="hello",
    )
    assert result is not None


# ------------------------------------------------------------------ #
#  Unit tests: InputPolicy.check() directly
# ------------------------------------------------------------------ #


class TestInputPolicyCustomClassifierUnit:
    def test_custom_classifier_adds_categories_to_check(self):
        def classifier(text: str) -> list[str]:
            return ["custom_cat"]

        policy = InputPolicy(
            flag_categories=["custom_cat"],
            custom_classifier=classifier,
        )
        result = policy.check("any text")
        assert "custom_cat" in result.flagged_categories
        assert "custom_cat" in result.matched_categories

    def test_custom_classifier_blocks(self):
        def classifier(text: str) -> list[str]:
            return ["weapons"]

        policy = InputPolicy(
            block_categories=["weapons"],
            custom_classifier=classifier,
        )
        result = policy.check("innocent text that classifier flags")
        assert result.blocked is True
        assert "weapons" in result.matched_categories

    def test_custom_classifier_and_keywords_no_duplicates(self):
        def classifier(text: str) -> list[str]:
            return ["medical"]

        policy = InputPolicy(
            flag_categories=["medical"],
            custom_classifier=classifier,
        )
        # Both keyword ("symptoms") and classifier return "medical"
        result = policy.check("what are the symptoms")
        assert result.flagged_categories.count("medical") == 1
        assert result.matched_categories.count("medical") == 1


# ------------------------------------------------------------------ #
#  Unit tests: OutputPolicy.check() directly
# ------------------------------------------------------------------ #


class TestOutputPolicyCustomClassifierUnit:
    def test_output_classifier_detects_category(self):
        def classifier(text: str) -> list[str]:
            if "secret dosage" in text.lower():
                return ["medical"]
            return []

        policy = OutputPolicy(
            scan_categories=["medical"],
            custom_classifier=classifier,
        )
        result = policy.check("The secret dosage is 100mg")
        assert "medical" in result.matched_categories
        assert "medical" in result.flagged_categories

    def test_output_classifier_crash_handled(self):
        def broken(text: str) -> list[str]:
            raise ValueError("boom")

        policy = OutputPolicy(
            scan_categories=["medical"],
            custom_classifier=broken,
        )
        # Should not crash
        result = policy.check("some output text")
        assert result is not None
        assert result.blocked is False

    def test_output_classifier_ignores_non_scan_categories(self):
        """Custom classifier categories not in scan_categories are ignored."""

        def classifier(text: str) -> list[str]:
            return ["weapons", "unknown_cat"]

        policy = OutputPolicy(
            scan_categories=["medical"],
            custom_classifier=classifier,
        )
        result = policy.check("some text")
        # weapons and unknown_cat not in scan_categories, so not detected
        assert "weapons" not in result.matched_categories
        assert "unknown_cat" not in result.matched_categories
