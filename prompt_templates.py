"""
Prompt Templates for LLM Data Augmentation

This module contains various prompt templates for generating augmented samples
to improve rumor detection model performance.
"""

# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT_AUGMENTOR = """You are an expert in social media content analysis and data augmentation. Your task is to generate diverse variants of social media posts while preserving their semantic meaning and label (rumor or non-rumor).

Guidelines:
- Maintain the core message and truthfulness
- Vary expression style, vocabulary, and structure
- Keep the content natural and realistic
- Preserve the rumor/non-rumor classification"""


SYSTEM_PROMPT_ANALYZER = """You are an expert in rumor detection and misinformation analysis. You can identify rumor patterns, analyze credibility, and provide detailed reasoning."""


# =============================================================================
# Augmentation Prompts
# =============================================================================

def get_paraphrase_prompt(text: str, label: str, num_variants: int = 5) -> str:
    """
    Generate paraphrased variants of the original text
    
    Args:
        text: Original social media post
        label: "rumor" or "non-rumor"
        num_variants: Number of variants to generate
    
    Returns:
        Formatted prompt string
    """
    label_desc = "rumor (false information)" if label == "rumor" else "non-rumor (truthful information)"
    
    prompt = f"""Generate {num_variants} paraphrased variants of the following social media post.

Original Post (Label: {label_desc}):
"{text}"

Requirements:
1. Each variant must preserve the SAME label ({label})
2. Change wording, sentence structure, and expression style
3. Maintain the core message and tone
4. Keep each variant realistic and natural
5. Each variant should be on a separate line

Generate {num_variants} variants:"""
    
    return prompt


def get_style_transfer_prompt(text: str, label: str, styles: list = None) -> str:
    """
    Generate variants with different writing styles
    
    Args:
        text: Original social media post
        label: "rumor" or "non-rumor"
        styles: List of styles to apply (default: ['formal', 'casual', 'questioning', 'emotional'])
    
    Returns:
        Formatted prompt string
    """
    if styles is None:
        styles = ['formal', 'casual', 'questioning', 'emotional', 'neutral']
    
    label_desc = "rumor (false information)" if label == "rumor" else "non-rumor (truthful information)"
    styles_str = ', '.join(styles)
    
    prompt = f"""Rewrite the following social media post in different styles while keeping it as a {label_desc}.

Original Post:
"{text}"

Generate variants in these styles: {styles_str}

Requirements:
- Each variant must maintain the {label} classification
- Apply the specified style clearly
- Keep content realistic and natural
- Format: One variant per line with style label

Example format:
[formal] <rewritten text>
[casual] <rewritten text>

Generate variants:"""
    
    return prompt


def get_emphasis_variation_prompt(text: str, label: str, num_variants: int = 5) -> str:
    """
    Generate variants with different emphasis patterns
    
    Args:
        text: Original social media post
        label: "rumor" or "non-rumor"
        num_variants: Number of variants to generate
    
    Returns:
        Formatted prompt string
    """
    label_desc = "rumor (false information)" if label == "rumor" else "non-rumor (truthful information)"
    
    prompt = f"""Generate {num_variants} variants of the following post by changing emphasis and intensity while maintaining its {label} classification.

Original Post:
"{text}"

Variation strategies:
- Change urgency level (urgent → mild, or vice versa)
- Modify certainty (definite → uncertain, or vice versa)
- Adjust emotional intensity
- Vary punctuation and emphasis markers
- Change question/statement format

Requirements:
- Keep the core claim the same
- Maintain {label} classification
- Each variant on a separate line
- Stay realistic

Generate {num_variants} variants:"""
    
    return prompt


def get_entity_substitution_prompt(text: str, label: str, num_variants: int = 3) -> str:
    """
    Generate variants by substituting entities (while keeping rumor patterns)
    
    Args:
        text: Original social media post
        label: "rumor" or "non-rumor"
        num_variants: Number of variants to generate
    
    Returns:
        Formatted prompt string
    """
    label_desc = "rumor (false information)" if label == "rumor" else "non-rumor (truthful information)"
    
    prompt = f"""Generate {num_variants} variants by substituting specific entities while preserving the {label} pattern.

Original Post ({label_desc}):
"{text}"

Instructions:
- Replace specific names, locations, numbers with plausible alternatives
- Keep the rumor/misinformation pattern if it's a rumor
- Keep the factual pattern if it's non-rumor
- Maintain the same narrative structure
- Each variant on a separate line

Generate {num_variants} variants:"""
    
    return prompt


def get_mixed_augmentation_prompt(text: str, label: str, num_variants: int = 5) -> str:
    """
    Generate diverse variants using mixed augmentation strategies
    (Recommended for best diversity)
    
    Args:
        text: Original social media post
        label: "rumor" or "non-rumor"
        num_variants: Number of variants to generate
    
    Returns:
        Formatted prompt string
    """
    label_desc = "rumor (false information)" if label == "rumor" else "non-rumor (truthful information)"
    
    prompt = f"""Generate {num_variants} diverse variants of this social media post using MIXED augmentation strategies.

Original Post (Label: {label_desc}):
"{text}"

Use a combination of these techniques:
1. Paraphrasing (different wording)
2. Style changes (formal/casual/questioning)
3. Emphasis variation (urgency, certainty)
4. Structural changes (sentence order, format)
5. Tonal shifts (while keeping {label} classification)

CRITICAL REQUIREMENTS:
- ALL variants must remain {label}
- Maximize diversity across variants
- Keep each variant realistic and natural
- One variant per line
- No numbering or prefixes

Generate {num_variants} diverse variants:"""
    
    return prompt


# =============================================================================
# Simple Paraphrase Prompts (for SeAug compatibility)
# =============================================================================

def get_simple_paraphrase_prompt(text: str, label: str = None) -> str:
    """
    Generate a simple paraphrased version (single output)
    Compatible with SeAug's single-text augmentation
    
    Args:
        text: Original social media post
        label: Optional label for label-aware paraphrasing
    
    Returns:
        Formatted prompt string
    """
    if label:
        label_desc = "rumor (false information)" if label == "rumor" else "non-rumor (truthful information)"
        prompt = f"""Paraphrase the following social media post while keeping the same meaning and classification.

Original Post (Label: {label_desc}):
"{text}"

Requirements:
- Maintain the {label} classification
- Keep the core message unchanged
- Use different wording and sentence structure
- Stay natural and realistic

Paraphrased version (one line only):"""
    else:
        # Fallback to non-label-aware version
        prompt = f"""Paraphrase the following social media post while keeping the same meaning:

Original: "{text}"

Paraphrased version (one line only):"""
    
    return prompt


def get_batch_paraphrase_prompt(texts: list, labels: list = None) -> str:
    """
    Generate paraphrased versions for multiple texts (batch processing)
    Compatible with SeAug's batch augmentation
    
    Args:
        texts: List of social media posts
        labels: Optional list of labels for label-aware paraphrasing
    
    Returns:
        Formatted prompt string
    """
    if len(texts) == 1:
        # Use single text prompt
        label = labels[0] if labels else None
        return get_simple_paraphrase_prompt(texts[0], label)
    
    if labels and len(labels) == len(texts):
        # Label-aware batch paraphrasing
        text_list = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            label_desc = "rumor" if label == "rumor" else "non-rumor"
            text_list.append(f"{i+1}. [{label_desc}] \"{text}\"")
        text_list_str = "\n".join(text_list)
        
        prompt = f"""Paraphrase the following {len(texts)} social media posts while keeping the same meaning and classification for each.
Provide one paraphrased version per line, in the same order.

Original posts (with labels):
{text_list_str}

Requirements:
- Maintain each post's label (rumor/non-rumor)
- Keep core messages unchanged
- Use different wording and sentence structures
- Stay natural and realistic

Paraphrased versions (one per line, no numbering or labels):"""
    else:
        # Non-label-aware batch paraphrasing (original behavior)
        text_list = "\n".join([f"{i+1}. \"{text}\"" for i, text in enumerate(texts)])
        prompt = f"""Paraphrase the following {len(texts)} social media posts while keeping the same meaning for each.
Provide one paraphrased version per line, in the same order.

Original posts:
{text_list}

Paraphrased versions (one per line, no numbering):"""
    
    return prompt


# =============================================================================
# Propagation Tree Generation Prompts
# =============================================================================

def get_propagation_tree_prompt(source_text: str, label: str) -> str:
    """
    Generate a propagation tree structure (source + replies)
    
    Args:
        source_text: Original source post
        label: "rumor" or "non-rumor"
    
    Returns:
        Formatted prompt string
    """
    label_desc = "rumor (false information)" if label == "rumor" else "non-rumor (truthful information)"
    
    prompt = f"""Generate a realistic social media propagation tree for the following post.

Source Post (Label: {label_desc}):
"{source_text}"

Generate:
1. A rewritten version of the source post
2. 4-6 replies that form a propagation pattern

For a {label}, the propagation pattern should show:
- Mix of believers and skeptics
- Varying levels of engagement
- Realistic discussion flow

Output format (JSON):
{{
  "source": "rewritten source post",
  "replies": [
    "reply 1",
    "reply 2",
    "reply 3",
    "reply 4"
  ]
}}

Generate the propagation tree:"""
    
    return prompt


# =============================================================================
# Quality Control Prompts
# =============================================================================

def get_validation_prompt(original: str, variant: str, label: str) -> str:
    """
    Validate if a generated variant maintains the correct label
    
    Args:
        original: Original text
        variant: Generated variant
        label: Expected label
    
    Returns:
        Validation prompt
    """
    prompt = f"""Verify if the generated variant maintains the same classification as the original.

Original: "{original}"
Variant: "{variant}"
Expected Label: {label}

Question: Does the variant still represent {label}?

Answer with:
- "VALID" if the variant correctly maintains the {label} classification
- "INVALID" if the variant changes the classification or is unrealistic

Response:"""
    
    return prompt


# =============================================================================
# Helper Functions
# =============================================================================

def get_recommended_prompt(text: str, label: str, num_variants: int = 5, strategy: str = "mixed") -> str:
    """
    Get the recommended prompt based on strategy
    
    Args:
        text: Original text
        label: "rumor" or "non-rumor"
        num_variants: Number of variants to generate
        strategy: Augmentation strategy
            - "paraphrase": Simple paraphrasing
            - "style": Style transfer
            - "emphasis": Emphasis variation
            - "entity": Entity substitution
            - "mixed": Mixed strategies (recommended)
    
    Returns:
        Appropriate prompt string
    """
    strategy_map = {
        "paraphrase": get_paraphrase_prompt,
        "style": get_style_transfer_prompt,
        "emphasis": get_emphasis_variation_prompt,
        "entity": get_entity_substitution_prompt,
        "mixed": get_mixed_augmentation_prompt
    }
    
    if strategy not in strategy_map:
        print(f"Warning: Unknown strategy '{strategy}', using 'mixed' as default")
        strategy = "mixed"
    
    return strategy_map[strategy](text, label, num_variants)


if __name__ == "__main__":
    # Test prompts
    print("="*70)
    print("Testing Prompt Templates")
    print("="*70)
    
    test_text = "Breaking: Earthquake hits California, thousands dead!"
    test_label = "rumor"
    
    print("\n1. Simple Paraphrase Prompt (Label-Aware):")
    print("-"*70)
    print(get_simple_paraphrase_prompt(test_text, test_label))
    
    print("\n2. Mixed Augmentation Prompt:")
    print("-"*70)
    print(get_mixed_augmentation_prompt(test_text, test_label, 3))
    
    print("\n3. Batch Paraphrase Prompt (Label-Aware):")
    print("-"*70)
    test_texts = [
        "Breaking: Earthquake hits California, thousands dead!",
        "Scientists confirm: climate change is real",
        "Government hiding alien contact, whistleblower reveals"
    ]
    test_labels = ["rumor", "non-rumor", "rumor"]
    print(get_batch_paraphrase_prompt(test_texts, test_labels))
    
    print("\n4. Style Transfer Prompt:")
    print("-"*70)
    print(get_style_transfer_prompt(test_text, test_label))
    
    print("\n5. Propagation Tree Prompt:")
    print("-"*70)
    print(get_propagation_tree_prompt(test_text, test_label))

