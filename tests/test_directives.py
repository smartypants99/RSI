from timedilate.directives import DirectiveGenerator


def test_classify_code_task():
    gen = DirectiveGenerator()
    assert gen.classify_task("Write a Python function to sort a list") == "code"


def test_classify_prose_task():
    gen = DirectiveGenerator()
    assert gen.classify_task("Write an essay about climate change") == "prose"


def test_classify_general_task():
    gen = DirectiveGenerator()
    assert gen.classify_task("Do something") == "general"


def test_get_directives_code():
    gen = DirectiveGenerator()
    directives = gen.get_directives("code")
    assert len(directives) > 0
    assert all(isinstance(d, str) for d in directives)


def test_get_directives_prose():
    gen = DirectiveGenerator()
    directives = gen.get_directives("prose")
    assert len(directives) > 0


def test_cycle_through_directives():
    gen = DirectiveGenerator()
    directives = gen.get_directives("code")
    for i in range(len(directives) + 5):
        d = gen.next_directive("code", i)
        assert isinstance(d, str)


def test_custom_directive_generation_prompt():
    gen = DirectiveGenerator()
    prompt = gen.generate_custom_directive_prompt(
        "code", "Write a sorting function", "def sort(lst): return sorted(lst)"
    )
    assert "improve" in prompt.lower() or "novel" in prompt.lower()


def test_directive_for_weakness():
    gen = DirectiveGenerator()
    d = gen.directive_for_weakness("correctness")
    assert "bug" in d.lower() or "error" in d.lower() or "fix" in d.lower()
    d = gen.directive_for_weakness("completeness")
    assert "missing" in d.lower() or "add" in d.lower()
    d = gen.directive_for_weakness("quality")
    assert "structure" in d.lower() or "readab" in d.lower()
    d = gen.directive_for_weakness("elegance")
    assert "refactor" in d.lower() or "clean" in d.lower()


def test_directive_for_unknown_weakness():
    gen = DirectiveGenerator()
    d = gen.directive_for_weakness("unknown")
    assert "improve" in d.lower()


def test_high_score_directives_used_at_70():
    gen = DirectiveGenerator()
    d_low = gen.next_directive("code", 0, current_score=50)
    d_high = gen.next_directive("code", 0, current_score=70)
    assert d_low != d_high  # different directive sets


def test_high_score_code_directives_are_specific():
    gen = DirectiveGenerator()
    directives = gen.get_high_score_directives("code")
    assert len(directives) >= 5
    # High-score directives should be more surgical
    all_text = " ".join(directives).lower()
    assert "complexity" in all_text or "edge case" in all_text


def test_high_score_prose_directives():
    gen = DirectiveGenerator()
    directives = gen.get_high_score_directives("prose")
    assert len(directives) >= 5


def test_low_score_uses_standard_directives():
    gen = DirectiveGenerator()
    standard = gen.get_directives("code")
    d = gen.next_directive("code", 0, current_score=30)
    assert d == standard[0]


def test_trajectory_rising_uses_standard():
    gen = DirectiveGenerator()
    # Rising fast: 50 -> 60 -> 70
    d = gen.trajectory_aware_directive("code", 0, current_score=70, score_history=[50, 60, 70])
    standard = gen.get_directives("code")
    high = gen.get_high_score_directives("code")
    assert d == standard[0] or d == high[0]  # should use next_directive path


def test_trajectory_plateaued_uses_high_score():
    gen = DirectiveGenerator()
    # Plateaued: 70 -> 70 -> 70
    d = gen.trajectory_aware_directive("code", 0, current_score=70, score_history=[70, 70, 70])
    high = gen.get_high_score_directives("code")
    assert d in high


def test_trajectory_short_history_falls_back():
    gen = DirectiveGenerator()
    # Only 1 score — falls back to next_directive
    d = gen.trajectory_aware_directive("code", 0, current_score=50, score_history=[50])
    standard = gen.get_directives("code")
    assert d == standard[0]


def test_custom_directive_prompt_includes_failed_and_score():
    gen = DirectiveGenerator()
    failed = {"Optimize performance", "Add error handling"}
    result = gen.generate_custom_directive_prompt(
        "code", "Write a sort function", "def sort(x): return sorted(x)",
        failed_directives=failed, current_score=85,
    )
    assert "Optimize performance" in result
    assert "Add error handling" in result
    assert "did NOT improve" in result
    assert "85/100" in result
    assert "subtle, high-impact" in result


def test_custom_directive_prompt_low_score():
    gen = DirectiveGenerator()
    result = gen.generate_custom_directive_prompt(
        "code", "Write a sort function", "def sort(x): pass",
        current_score=30,
    )
    assert "30/100" in result
    assert "bold change" in result


def test_custom_directive_prompt_no_extras():
    gen = DirectiveGenerator()
    result = gen.generate_custom_directive_prompt(
        "code", "Write a sort function", "def sort(x): return sorted(x)",
    )
    assert "did NOT improve" not in result
    assert "/100" not in result
