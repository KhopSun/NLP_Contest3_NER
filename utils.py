"""
Utility functions for NER evaluation and data processing.

This module provides entity-span level evaluation metrics (Precision, Recall, F1)
as required by the NER contest assignment.
"""

from collections import defaultdict
from typing import List, Tuple, Set, Dict


def extract_entities(tokens: List[str], tags: List[str]) -> List[Tuple[str, str, int, int]]:
    """
    Extract entity spans from BIO tags.

    Args:
        tokens: List of tokenized strings
        tags: List of BIO tags (same length as tokens)

    Returns:
        List of tuples: (entity_text, entity_type, start_idx, end_idx)

    Example:
        tokens = ["frank", "d.", "o'connor", "lawyer"]
        tags = ["B-Politician", "I-Politician", "I-Politician", "O"]

        Returns: [("frank d. o'connor", "Politician", 0, 2)]
    """
    entities = []
    current_entity = None
    current_tokens = []
    start_idx = None

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith('B-'):
            # Save previous entity if exists
            if current_entity:
                entities.append((
                    ' '.join(current_tokens),
                    current_entity,
                    start_idx,
                    i - 1
                ))
            # Start new entity
            current_entity = tag[2:]  # Remove 'B-' prefix
            current_tokens = [token]
            start_idx = i

        elif tag.startswith('I-'):
            # Continue current entity
            if current_entity:
                current_tokens.append(token)
            else:
                # Invalid BIO sequence - I- without B-
                # Treat as beginning of entity
                current_entity = tag[2:]
                current_tokens = [token]
                start_idx = i

        else:  # 'O' tag
            # Save previous entity if exists
            if current_entity:
                entities.append((
                    ' '.join(current_tokens),
                    current_entity,
                    start_idx,
                    i - 1
                ))
                current_entity = None
                current_tokens = []
                start_idx = None

    # Don't forget last entity
    if current_entity:
        entities.append((
            ' '.join(current_tokens),
            current_entity,
            start_idx,
            len(tokens) - 1
        ))

    return entities


def get_entity_spans(tokens: List[str], tags: List[str]) -> Set[Tuple[int, int, str]]:
    """
    Get entity spans as a set of (start, end, type) tuples.

    This format is used for entity-span level evaluation where
    an entity is only correct if the entire span matches.

    Args:
        tokens: List of tokenized strings
        tags: List of BIO tags

    Returns:
        Set of (start_idx, end_idx, entity_type) tuples

    Example:
        tokens = ["frank", "d.", "o'connor", "lawyer"]
        tags = ["B-Politician", "I-Politician", "I-Politician", "O"]

        Returns: {(0, 2, "Politician")}
    """
    entities = extract_entities(tokens, tags)
    return {(start, end, entity_type) for _, entity_type, start, end in entities}


def evaluate_entity_spans(
    true_tags_list: List[List[str]],
    pred_tags_list: List[List[str]],
    tokens_list: List[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate entity-span level Precision, Recall, and F1.

    A predicted entity is correct ONLY if:
    - The start index matches
    - The end index matches
    - The entity type matches

    This is "strict" evaluation as required by the assignment.

    Args:
        true_tags_list: List of ground truth tag sequences
        pred_tags_list: List of predicted tag sequences
        tokens_list: Optional list of token sequences (for debugging)

    Returns:
        Dictionary with keys: 'precision', 'recall', 'f1', 'true_positives',
        'false_positives', 'false_negatives'

    Example:
        true_tags = [["B-Politician", "I-Politician", "O"]]
        pred_tags = [["B-Politician", "I-Politician", "O"]]

        evaluate_entity_spans(true_tags, pred_tags)
        # Returns: {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, ...}
    """
    assert len(true_tags_list) == len(pred_tags_list), \
        "Number of true and predicted sequences must match"

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i, (true_tags, pred_tags) in enumerate(zip(true_tags_list, pred_tags_list)):
        # Handle tokens for span extraction
        if tokens_list is not None:
            tokens = tokens_list[i]
        else:
            # Create dummy tokens
            tokens = [f"token_{j}" for j in range(len(true_tags))]

        # Get entity spans
        true_spans = get_entity_spans(tokens, true_tags)
        pred_spans = get_entity_spans(tokens, pred_tags)

        # Count matches
        true_positives += len(true_spans & pred_spans)  # Intersection
        false_positives += len(pred_spans - true_spans)  # Predicted but not true
        false_negatives += len(true_spans - pred_spans)  # True but not predicted

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def evaluate_entity_spans_by_type(
    true_tags_list: List[List[str]],
    pred_tags_list: List[List[str]],
    tokens_list: List[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate entity-span level metrics separately for each entity type.

    This provides detailed per-class performance analysis.

    Args:
        true_tags_list: List of ground truth tag sequences
        pred_tags_list: List of predicted tag sequences
        tokens_list: Optional list of token sequences

    Returns:
        Dictionary mapping entity_type -> metrics dict

    Example:
        Results:
        {
            'Politician': {'precision': 0.95, 'recall': 0.92, 'f1': 0.935, ...},
            'Artist': {'precision': 0.88, 'recall': 0.85, 'f1': 0.865, ...},
            ...
        }
    """
    assert len(true_tags_list) == len(pred_tags_list), \
        "Number of true and predicted sequences must match"

    # Count by entity type
    type_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for i, (true_tags, pred_tags) in enumerate(zip(true_tags_list, pred_tags_list)):
        # Handle tokens
        if tokens_list is not None:
            tokens = tokens_list[i]
        else:
            tokens = [f"token_{j}" for j in range(len(true_tags))]

        # Get entity spans
        true_entities = extract_entities(tokens, true_tags)
        pred_entities = extract_entities(tokens, pred_tags)

        # Convert to span sets per type
        true_spans_by_type = defaultdict(set)
        pred_spans_by_type = defaultdict(set)

        for _, entity_type, start, end in true_entities:
            true_spans_by_type[entity_type].add((start, end))

        for _, entity_type, start, end in pred_entities:
            pred_spans_by_type[entity_type].add((start, end))

        # Get all entity types
        all_types = set(true_spans_by_type.keys()) | set(pred_spans_by_type.keys())

        # Count for each type
        for entity_type in all_types:
            true_spans = true_spans_by_type[entity_type]
            pred_spans = pred_spans_by_type[entity_type]

            type_stats[entity_type]['tp'] += len(true_spans & pred_spans)
            type_stats[entity_type]['fp'] += len(pred_spans - true_spans)
            type_stats[entity_type]['fn'] += len(true_spans - pred_spans)

    # Calculate metrics for each type
    results = {}
    for entity_type, stats in type_stats.items():
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[entity_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'support': tp + fn  # Number of true entities of this type
        }

    return results


def print_evaluation_report(
    true_tags_list: List[List[str]],
    pred_tags_list: List[List[str]],
    tokens_list: List[List[str]] = None,
    model_name: str = "Model"
):
    """
    Print a comprehensive evaluation report with overall and per-type metrics.

    Args:
        true_tags_list: List of ground truth tag sequences
        pred_tags_list: List of predicted tag sequences
        tokens_list: Optional list of token sequences
        model_name: Name of the model being evaluated
    """
    print("=" * 80)
    print(f"ENTITY-SPAN LEVEL EVALUATION REPORT: {model_name}")
    print("=" * 80)

    # Overall metrics
    overall = evaluate_entity_spans(true_tags_list, pred_tags_list, tokens_list)

    print("\nOVERALL METRICS:")
    print(f"  Precision: {overall['precision']:.4f}")
    print(f"  Recall:    {overall['recall']:.4f}")
    print(f"  F1 Score:  {overall['f1']:.4f}")
    print(f"\n  True Positives:  {overall['true_positives']}")
    print(f"  False Positives: {overall['false_positives']}")
    print(f"  False Negatives: {overall['false_negatives']}")

    # Per-type metrics
    by_type = evaluate_entity_spans_by_type(true_tags_list, pred_tags_list, tokens_list)

    print("\n" + "-" * 80)
    print("PER-ENTITY-TYPE METRICS:")
    print("-" * 80)
    print(f"{'Entity Type':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 80)

    for entity_type in sorted(by_type.keys()):
        metrics = by_type[entity_type]
        print(f"{entity_type:<20} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f} "
              f"{metrics['support']:<10}")

    print("=" * 80)


if __name__ == "__main__":
    # Test the evaluation functions
    print("Testing entity-span evaluation functions...\n")

    # Test case 1: Perfect prediction
    print("Test 1: Perfect prediction")
    true_tags = [["B-Politician", "I-Politician", "O", "B-Artist"]]
    pred_tags = [["B-Politician", "I-Politician", "O", "B-Artist"]]
    tokens = [["Obama", "Jr", "is", "artist"]]

    result = evaluate_entity_spans(true_tags, pred_tags, tokens)
    print(f"  Expected F1: 1.0, Got: {result['f1']:.4f}")
    assert result['f1'] == 1.0, "Perfect prediction should have F1 = 1.0"

    # Test case 2: Missed entity
    print("\nTest 2: Missed entity")
    true_tags = [["B-Politician", "I-Politician", "O", "B-Artist"]]
    pred_tags = [["B-Politician", "I-Politician", "O", "O"]]
    tokens = [["Obama", "Jr", "is", "artist"]]

    result = evaluate_entity_spans(true_tags, pred_tags, tokens)
    print(f"  Precision: {result['precision']:.4f} (1 correct, 0 wrong)")
    print(f"  Recall: {result['recall']:.4f} (1 found, 1 missed)")
    print(f"  F1: {result['f1']:.4f}")

    # Test case 3: Wrong span boundary
    print("\nTest 3: Wrong span boundary (should be incorrect)")
    true_tags = [["B-Politician", "I-Politician", "I-Politician", "O"]]
    pred_tags = [["B-Politician", "I-Politician", "O", "O"]]  # Span too short
    tokens = [["Barack", "Hussein", "Obama", "Jr"]]

    result = evaluate_entity_spans(true_tags, pred_tags, tokens)
    print(f"  F1: {result['f1']:.4f} (spans don't match exactly)")
    assert result['true_positives'] == 0, "Partial span should not count as correct"

    print("\n✅ All tests passed!")
