CODE_DIRECTIVES = [
    "Fix any bugs or logical errors in this code.",
    "Add handling for edge cases (empty input, invalid types, boundary values).",
    "Optimize the performance of this code.",
    "Add comprehensive error handling and input validation.",
    "Refactor for readability and clean code principles.",
    "Explore an alternative algorithmic approach.",
    "Add inline comments explaining complex logic.",
    "Stress-test: what happens with very large inputs?",
    "Add useful features that serve the original goal.",
]

PROSE_DIRECTIVES = [
    "Improve clarity and conciseness of the writing.",
    "Strengthen the arguments with better evidence or reasoning.",
    "Add concrete examples to illustrate key points.",
    "Consider and address counterarguments.",
    "Improve the overall structure and flow.",
    "Refine the tone to better match the intended audience.",
    "Deepen the analysis where it's currently surface-level.",
    "Remove redundancy and filler.",
    "Add a stronger opening and conclusion.",
]

GENERAL_DIRECTIVES = [
    "Improve the overall quality and completeness.",
    "Fix any errors or inaccuracies.",
    "Add more detail where it's needed.",
    "Consider alternative approaches.",
    "Make the output more useful for the end user.",
    "Improve structure and organization.",
    "Add examples or illustrations.",
    "Polish and refine the output.",
]

HIGH_SCORE_CODE_DIRECTIVES = [
    "Review every branch condition — is there an off-by-one or missed edge case?",
    "Reduce the cyclomatic complexity of the most complex function.",
    "Replace any magic numbers or strings with named constants.",
    "Ensure the time complexity is optimal for the problem constraints.",
    "Add type annotations and tighten the API surface.",
    "Make error messages actionable — include what went wrong and how to fix it.",
]

HIGH_SCORE_PROSE_DIRECTIVES = [
    "Find the weakest paragraph and rewrite it with a stronger claim and evidence.",
    "Cut any sentence that doesn't advance the argument.",
    "Replace abstract statements with concrete, specific examples.",
    "Improve transitions between paragraphs for smoother flow.",
    "Sharpen the thesis — can someone disagree with it?",
    "Ensure the conclusion adds insight rather than just summarizing.",
]

HIGH_SCORE_GENERAL_DIRECTIVES = [
    "Identify the single weakest part of this output and make it excellent.",
    "Remove anything that adds length without adding value.",
    "Add one concrete example that makes the output more useful.",
    "Verify every factual claim or assumption — fix any that are wrong.",
    "Improve the opening to immediately engage the reader.",
]

TASK_KEYWORDS = {
    "code": [
        "function", "code", "program", "script", "implement", "build",
        "create a ", "write a ", "class", "api", "app", "game", "website",
        "algorithm", "roblox", "python", "javascript",
    ],
    "prose": [
        "essay", "write about", "explain", "describe", "article", "blog",
        "letter", "email", "story", "poem", "report", "an essay",
    ],
}


class DirectiveGenerator:
    def classify_task(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        scores = {"code": 0, "prose": 0}
        for task_type, keywords in TASK_KEYWORDS.items():
            for kw in keywords:
                if kw in prompt_lower:
                    scores[task_type] += 1
        if scores["code"] > scores["prose"]:
            return "code"
        if scores["prose"] > scores["code"]:
            return "prose"
        return "general"

    def get_directives(self, task_type: str) -> list[str]:
        if task_type == "code":
            return CODE_DIRECTIVES
        if task_type == "prose":
            return PROSE_DIRECTIVES
        return GENERAL_DIRECTIVES

    def get_high_score_directives(self, task_type: str) -> list[str]:
        if task_type == "code":
            return HIGH_SCORE_CODE_DIRECTIVES
        if task_type == "prose":
            return HIGH_SCORE_PROSE_DIRECTIVES
        return HIGH_SCORE_GENERAL_DIRECTIVES

    def next_directive(self, task_type: str, cycle_index: int, current_score: int = 0) -> str:
        if current_score >= 70:
            directives = self.get_high_score_directives(task_type)
        else:
            directives = self.get_directives(task_type)
        return directives[cycle_index % len(directives)]

    def directive_for_weakness(self, weakest_aspect: str) -> str:
        """Generate a directive that targets a specific weakness."""
        aspect_directives = {
            "correctness": "Focus on fixing any bugs, logical errors, or factual inaccuracies. Verify every claim and every code path.",
            "completeness": "The output is missing important parts. Add anything that was asked for but not delivered. Cover all aspects of the task.",
            "quality": "Improve structure, readability, and polish. Better formatting, clearer organization, more professional presentation.",
            "elegance": "Refactor for cleaner design. Simplify where possible, use better patterns, make the approach more efficient and well-designed.",
        }
        return aspect_directives.get(weakest_aspect, "Improve the overall quality.")

    def generate_custom_directive_prompt(
        self, task_type: str, original_prompt: str, current_output: str
    ) -> str:
        return (
            f"Given the task: {original_prompt}\n\n"
            f"And the current output:\n{current_output}\n\n"
            f"The following standard improvements have already been applied multiple times.\n"
            f"Suggest ONE novel, specific improvement that would make this output meaningfully better.\n"
            f"Be creative — think of something not on this list: {', '.join(self.get_directives(task_type))}\n"
            f"Respond with just the improvement directive, one sentence, nothing else."
        )
