from app.services.chat_prompts import build_chat_system_prompt, build_chat_user_prompt
from app.services.graphrag.router import GraphIntent


def test_general_qa_prompt_is_topic_neutral_and_grounded():
    prompt = build_chat_system_prompt(GraphIntent.FACTUAL)

    assert "Cite every factual paper claim" in prompt
    assert "evidence is insufficient" in prompt
    assert "Transformer" not in prompt
    assert "BERT" not in prompt
    assert "Always explicitly name the subject" not in prompt


def test_contradiction_prompt_has_its_own_comparability_rules():
    prompt = build_chat_system_prompt(GraphIntent.CONTRADICTION)

    assert "CONTRADICTION ANALYSIS:" in prompt
    assert "each side as its own source-supported statement" in prompt
    assert "comparable conditions" in prompt
    assert "no direct contradiction" in prompt


def test_relationship_prompt_distinguishes_correlation_from_causation():
    prompt = build_chat_system_prompt(GraphIntent.RELATIONSHIP)

    assert "RELATIONSHIP ANALYSIS:" in prompt
    assert "causal mechanism" in prompt
    assert "missing graph links as proof" in prompt


def test_corpus_theme_prompt_limits_claims_to_retrieved_papers():
    prompt = build_chat_system_prompt(GraphIntent.CORPUS_THEMES)

    assert "CORPUS THEME ANALYSIS:" in prompt
    assert "multiple papers" in prompt
    assert "not as universal or exhaustive" in prompt
    assert "does not establish absence from the full literature" in prompt


def test_user_prompt_includes_only_relevant_supplied_sections():
    prompt = build_chat_user_prompt(
        question="Explain the result.",
        evidence_block="[E1] evidence",
        memory_block="Saved preference",
    )

    assert "QUESTION:\nExplain the result." in prompt
    assert "EVIDENCE:\n[E1] evidence" in prompt
    assert "PROJECT MEMORY:\nSaved preference" in prompt
    assert "CONVERSATION HISTORY" not in prompt


def test_user_prompt_keeps_task_specific_response_format_separate():
    prompt = build_chat_user_prompt(
        question="Help me understand this paper.",
        evidence_block="[E1] source text",
        response_guidance="Use a cited reading brief format.",
    )

    assert "QUESTION:\nHelp me understand this paper." in prompt
    assert "RESPONSE FORMAT:\nUse a cited reading brief format." in prompt
