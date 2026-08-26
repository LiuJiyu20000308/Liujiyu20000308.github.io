#!/usr/bin/env python3
"""Add stable Python-solution links and TOC-visible LeetCode headings."""

from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[1]
POSTS = REPO / "_posts"
PYTHON_URL = "{{ '/leetcode/python-implementations/' | relative_url }}"

# LeetCode number -> section number in the Python companion article.
SECTION = {
    1: 33, 3: 10, 4: 3, 5: 8, 10: 171, 15: 35, 20: 45,
    26: 5, 28: 41, 30: 15, 37: 112, 39: 101, 40: 102, 42: 98,
    46: 109, 47: 110, 51: 111, 53: 116, 55: 118, 56: 128,
    62: 133, 63: 134, 70: 131, 72: 168, 76: 12, 77: 100,
    78: 106, 84: 99, 90: 107, 93: 105, 94: 71, 96: 136, 98: 89,
    101: 77, 104: 78, 108: 95, 111: 79, 113: 82, 115: 166,
    116: 76, 117: 76, 122: 117, 127: 183, 131: 104, 134: 121,
    135: 122, 139: 158, 144: 70, 145: 70, 146: 65, 150: 48,
    151: 39, 199: 74, 200: 176, 202: 32, 206: 24, 209: 9, 216: 103,
    226: 85, 235: 92, 236: 92, 238: 17, 239: 50, 242: 30,
    257: 81, 279: 157, 300: 159, 322: 156, 332: 113, 337: 139,
    343: 135, 344: 7, 347: 49, 349: 31, 376: 115, 377: 154,
    392: 165, 404: 83, 416: 148, 417: 180, 432: 69, 435: 126,
    438: 14, 452: 125, 454: 34, 455: 114, 459: 42, 460: 66,
    463: 182, 474: 151, 491: 108, 494: 150, 501: 91, 513: 84,
    516: 170, 518: 153, 525: 20, 530: 91, 538: 94, 559: 78,
    560: 18, 567: 16, 583: 167, 637: 75, 647: 169, 654: 97,
    669: 93, 674: 160, 684: 173, 685: 174, 695: 177, 707: 23,
    718: 161, 738: 129, 746: 132, 763: 127, 797: 175, 862: 51,
    876: 4, 904: 13, 968: 130, 974: 19, 977: 6, 981: 67,
    1004: 11, 1005: 120, 1035: 163, 1047: 46, 1143: 162,
    1797: 68, 1971: 172,
}

BANNER = (
    "> **Python 版本：** 本文保留原有 C++ 推导；每个例题对应的 Python "
    f"实现统一收录在[Python 实现全集]({PYTHON_URL})中。"
)


def python_link(section: int) -> str:
    return f"**Python 实现：** [查看对应代码]({PYTHON_URL}#py-{section:03d})"


def split_problem_line(path: Path, line: str, occurrences: dict[int, int]):
    match = re.match(r"(?i)^leetcode\s*(\d+)(?:\s*/\s*(\d+))?\s*[.。]?\s*(.*)$", line)
    if not match:
        return None

    first = int(match.group(1))
    second = int(match.group(2)) if match.group(2) else None
    remainder = match.group(3).strip()

    # Correct two obvious numbering typos while preserving the original prose.
    if path.name == "2024-8-1-DS.md" and first == 206:
        first = 209
    if path.name == "2024-8-27-DP.md" and first == 62:
        occurrences[first] = occurrences.get(first, 0) + 1
        if occurrences[first] == 2:
            first = 63

    numbers = [first] + ([second] if second else [])
    section = SECTION.get(first)
    if section is None:
        raise ValueError(f"No Python section for LeetCode {first} in {path.name}")
    label = "/".join(map(str, numbers))
    result = [f"### Leetcode {label}", python_link(section)]
    if remainder:
        result.extend(["", remainder])
    return result


def update_post(path: Path):
    text = path.read_text()
    if BANNER in text:
        return
    lines = text.splitlines()
    fence = False
    output = []
    occurrences: dict[int, int] = {}
    front_matter_end = None
    for index, line in enumerate(lines):
        if index > 0 and line == "---" and front_matter_end is None:
            front_matter_end = index
        if line.startswith("```"):
            fence = not fence
        replacement = None if fence else split_problem_line(path, line, occurrences)
        output.extend(replacement if replacement is not None else [line])

    if front_matter_end is None:
        raise ValueError(f"Missing front matter in {path}")
    output[front_matter_end + 1:front_matter_end + 1] = ["", BANNER]
    path.write_text("\n".join(output) + "\n")


def update_guide(path: Path):
    if path.read_text().count("**Python 实现：**") == 193:
        return
    lines, output, section = path.read_text().splitlines(), [], 0
    for line in lines:
        output.append(line)
        if line.startswith("### "):
            section += 1
            output.extend(["", python_link(section)])
    if section != 193:
        raise ValueError(f"Expected 193 guide examples, found {section}")
    path.write_text("\n".join(output) + "\n")


def main():
    for path in sorted(POSTS.glob("*.md")):
        text = path.read_text()
        title = re.search(r"^title:\s*(.+)$", text, re.MULTILINE)
        if not title or "Leetcode记录" not in title.group(1):
            continue
        if "Effective C++" in title.group(1) or "全部例题" in title.group(1):
            continue
        update_post(path)
    update_guide(POSTS / "2026-8-10-LeetcodeExamples.md")


if __name__ == "__main__":
    main()
