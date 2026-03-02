"""Reward function for GRPO training on pelican-bicycle SVG generation.

Scores generated SVGs on validity, structure, and content quality.
Returns a float between 0.0 and 1.0.
"""

import re
import xml.etree.ElementTree as ET


def reward_svg(completion: str, **kwargs) -> float:
    """Score a generated SVG completion for GRPO.

    Checks:
    - Valid XML/SVG parsing (0.25)
    - Correct viewBox (0.10)
    - Has <svg> root with xmlns (0.05)
    - Contains bicycle group with key parts (0.20)
    - Contains pelican group with key parts (0.20)
    - Has IDs on elements (0.10)
    - Reasonable length / not truncated (0.10)

    Returns:
        float: reward score between 0.0 and 1.0
    """
    score = 0.0
    text = completion.strip()

    # ── Valid XML parsing (0.25) ─────────────────────────────────────────
    try:
        # Strip XML declaration if present for ET parsing
        svg_text = re.sub(r'<\?xml[^?]*\?>\s*', '', text)
        root = ET.fromstring(svg_text)
        score += 0.25
    except ET.ParseError:
        # If it doesn't even parse, give partial credit for having svg tags
        if '<svg' in text and '</svg>' in text:
            score += 0.05
        return score

    # ── Correct viewBox (0.10) ───────────────────────────────────────────
    viewbox = root.attrib.get('viewBox', '')
    if viewbox == '0 0 512 512':
        score += 0.10
    elif 'viewBox' in root.attrib:
        score += 0.03  # has a viewBox, just wrong dimensions

    # ── Has proper SVG root with xmlns (0.05) ────────────────────────────
    if root.tag.endswith('svg') or root.tag == '{http://www.w3.org/2000/svg}svg':
        score += 0.05

    # ── Contains bicycle group with key parts (0.20) ─────────────────────
    svg_str = ET.tostring(root, encoding='unicode')
    bicycle_score = 0.0
    bike_parts = ['bicycle', 'wheel', 'frame', 'pedal', 'handlebar', 'saddle']
    for part in bike_parts:
        if part in svg_str.lower():
            bicycle_score += 1.0 / len(bike_parts)
    score += min(bicycle_score, 1.0) * 0.20

    # ── Contains pelican group with key parts (0.20) ─────────────────────
    pelican_score = 0.0
    pelican_parts = ['pelican', 'beak', 'wing', 'head', 'body', 'neck', 'eye']
    for part in pelican_parts:
        if part in svg_str.lower():
            pelican_score += 1.0 / len(pelican_parts)
    score += min(pelican_score, 1.0) * 0.20

    # ── Has IDs on elements (0.10) ───────────────────────────────────────
    id_count = len(re.findall(r'\bid="[^"]*"', text))
    if id_count >= 30:
        score += 0.10
    elif id_count >= 15:
        score += 0.07
    elif id_count >= 5:
        score += 0.04
    elif id_count >= 1:
        score += 0.02

    # ── Reasonable length (0.10) ─────────────────────────────────────────
    length = len(text)
    if 3000 <= length <= 20000:
        score += 0.10
    elif 1000 <= length <= 30000:
        score += 0.05
    elif length > 500:
        score += 0.02

    return round(min(score, 1.0), 3)


def reward_batch(completions: list[str], **kwargs) -> list[float]:
    """Score a batch of completions. Compatible with TRL GRPOTrainer.

    Args:
        completions: list of generated text strings

    Returns:
        list of reward scores
    """
    return [reward_svg(c, **kwargs) for c in completions]


if __name__ == "__main__":
    import json

    with open("pelican_bicycle_dataset.jsonl") as f:
        entries = [json.loads(line) for line in f]

    scores = [reward_svg(e["completion"]) for e in entries]
    print(f"Scored {len(scores)} SVGs")
    print(f"Min: {min(scores):.3f}  Max: {max(scores):.3f}  Avg: {sum(scores)/len(scores):.3f}")

    # Test with bad inputs
    print("\n--- Edge cases ---")
    print(f"Empty string:     {reward_svg(''):.3f}")
    print(f"Random text:      {reward_svg('hello world'):.3f}")
    print(f"Broken XML:       {reward_svg('<svg><rect></svg>'):.3f}")
    print(f"SVG no content:   {reward_svg('<svg viewBox=\"0 0 512 512\"></svg>'):.3f}")
