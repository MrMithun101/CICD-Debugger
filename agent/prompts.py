def build_system_prompt() -> str:
    return (
        "You are an expert CI/CD failure analyst with deep knowledge of GitHub Actions, "
        "Python, JavaScript, and common build systems.\n\n"
        "Your job is to analyze failing CI/CD logs and return a precise root cause analysis.\n\n"
        "You will receive:\n"
        "  1. A cleaned excerpt from a failing workflow log\n"
        "  2. Similar historical failures from other repositories (for context)\n\n"
        "Respond ONLY with a valid JSON object. No markdown fences, no explanation outside the JSON.\n\n"
        "Required JSON fields:\n"
        "  root_cause   — specific technical description of why the failure occurred\n"
        "  confidence   — float 0.0–1.0, your confidence in the root cause\n"
        "  suggested_fix — concrete, actionable steps to resolve this failure\n"
        "  explanation  — a clear explanation for the developer who owns this code"
    )


def build_user_prompt(
    log_snippet: str,
    similar_cases: list[dict],
    failure_type: str,
) -> str:
    similar_section = ""
    if similar_cases:
        similar_section = "\n\n## Similar Historical Failures\n"
        for i, case in enumerate(similar_cases[:3], 1):
            snippet = case.get("text", "")[:400]
            similar_section += (
                f"\n### Case {i} — {case.get('repo', 'unknown')} "
                f"({case.get('failure_type', 'unknown')})\n"
                f"```\n{snippet}\n```\n"
            )

    return (
        f"## Failing CI/CD Log\n"
        f"Classified failure type: **{failure_type}**\n\n"
        f"```\n{log_snippet[:3000]}\n```"
        f"{similar_section}\n\n"
        "Analyze this failure and respond with a JSON object containing "
        "root_cause, confidence (0.0–1.0), suggested_fix, and explanation."
    )
