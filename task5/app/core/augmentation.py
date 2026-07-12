"""
app/core/augmentation.py — Text augmentation for boosting small training datasets.

Generates grammatically correct prefix/suffix variants of each example,
then deduplicates while preserving order.
"""


def augment_examples(examples: list[str], factor: int) -> list[str]:
    """
    Augment a list of training examples by `factor`.

    factor=1 → no augmentation (return originals only)
    factor=2 → each example generates ~10 variants
    """
    if factor <= 1:
        return list(examples)

    augmented = list(examples)
    prefixes = [
        "I want to {}",
        "Can you help me {}",
        "Please {}",
        "How do I {}",
        "I need to {}",
        "I would like to {}",
    ]
    suffixes = [
        "{}, please",
        "{} for me",
        "{} today",
        "{} if possible",
    ]

    for ex in examples:
        ex_lower = ex.lower().strip("?.!,")
        for _ in range(factor - 1):
            for prefix in prefixes:
                augmented.append(prefix.format(ex_lower))
            for suffix in suffixes:
                augmented.append(suffix.format(ex_lower))
            augmented.append(ex_lower)
            augmented.append(ex.strip("?.!,"))

    # Deduplicate while preserving order
    seen, result = set(), []
    for v in augmented:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result
