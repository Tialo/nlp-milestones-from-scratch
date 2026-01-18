import re
from dataclasses import dataclass


HEADER_RE = re.compile(r"^\s*(todo|done)\s*:\s*$", re.IGNORECASE)
BULLET_RE = re.compile(r"^\s*\*\s+")
# arXiv new-style id: 4 digits '.' 4-5 digits, optional version suffix
ARXIV_NEW_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4})\.(\d{4,5})(?:v\d+)?", re.IGNORECASE)
# arXiv old-style: category/YYMMNNN (7 digits), optional version
ARXIV_OLD_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/[A-Za-z\-]+/(\d{7})(?:v\d+)?", re.IGNORECASE)

# Date patterns for non-arXiv entries (after URL typically, but we just scan "tail")
DATE_MMY_RE = re.compile(r"(?<!\d)(\d{2})\.(\d{4})(?!\d)")   # 06.2017
DATE_YMM_RE = re.compile(r"(?<!\d)(\d{4})[-/](\d{2})(?!\d)") # 2017-06 or 2017/06


@dataclass(frozen=True, order=True)
class SortKey:
    year: int
    month: int
    kind: int            # 0 = arXiv, 1 = dated non-arXiv, 2 = unknown
    number: int          # paper number for arXiv; 0 for non-arXiv
    original_index: int  # to keep stable ordering for ties / unknown


def yy_to_year(yy: int) -> int:
    # heuristic: 00-49 => 2000-2049, else => 1900-1999
    return 2000 + yy if yy < 50 else 1900 + yy


def parse_sort_key(line: str, original_index: int) -> SortKey:
    """
    Extract sort key from a bullet line.
    Prefers arXiv id; otherwise looks for a date like 06.2017 after the URL.
    """
    # 1) arXiv new-style: yymm.number
    m = ARXIV_NEW_RE.search(line)
    if m:
        yymm = m.group(1)
        num = int(m.group(2))
        yy = int(yymm[:2])
        mm = int(yymm[2:4])
        year = yy_to_year(yy)
        return SortKey(year=year, month=mm, kind=0, number=num, original_index=original_index)

    # 2) arXiv old-style: category/YYMMNNN (7 digits)
    m = ARXIV_OLD_RE.search(line)
    if m:
        digits = m.group(1)  # YYMMNNN
        yy = int(digits[:2])
        mm = int(digits[2:4])
        num = int(digits[4:])
        year = yy_to_year(yy)
        return SortKey(year=year, month=mm, kind=0, number=num, original_index=original_index)

    # 3) non-arXiv: find a date pattern anywhere after the URL (or in the whole line)
    # Try mm.yyyy first
    m = DATE_MMY_RE.search(line)
    if m:
        mm = int(m.group(1))
        year = int(m.group(2))
        return SortKey(year=year, month=mm, kind=1, number=0, original_index=original_index)

    # Then yyyy-mm / yyyy/mm
    m = DATE_YMM_RE.search(line)
    if m:
        year = int(m.group(1))
        mm = int(m.group(2))
        return SortKey(year=year, month=mm, kind=1, number=0, original_index=original_index)

    # 4) unknown
    return SortKey(year=9999, month=12, kind=2, number=999999, original_index=original_index)


def sort_section_bullets(section_lines: list[str]) -> list[str]:
    """
    Sort only bullet lines within a section. Non-bullet lines stay where they are.
    """
    bullet_positions: list[int] = []
    bullets: list[str] = []

    for i, ln in enumerate(section_lines):
        if BULLET_RE.match(ln):
            bullet_positions.append(i)
            bullets.append(ln)

    if not bullets:
        return section_lines[:]  # nothing to do

    keyed = [(parse_sort_key(bullets[i], i), bullets[i]) for i in range(len(bullets))]
    keyed.sort(key=lambda x: x[0])

    sorted_bullets = [b for _, b in keyed]
    out = section_lines[:]
    for pos, new_line in zip(bullet_positions, sorted_bullets):
        out[pos] = new_line
    return out


def sort_file_content(lines: list[str]) -> list[str]:
    """
    Parse the file into segments separated by 'todo:' / 'done:' headers.
    Then sort bullet items inside each recognized section.
    """
    segments: list[tuple[str, str | None, list[str]]] = []
    i = 0
    n = len(lines)

    while i < n:
        header_match = HEADER_RE.match(lines[i])
        if header_match:
            section_name = header_match.group(1).lower()
            header_line = lines[i]
            i += 1
            content: list[str] = []
            while i < n and not HEADER_RE.match(lines[i]):
                content.append(lines[i])
                i += 1
            segments.append(("section", section_name, [header_line] + content))
        else:
            text: list[str] = []
            while i < n and not HEADER_RE.match(lines[i]):
                text.append(lines[i])
                i += 1
            segments.append(("text", None, text))

    # Process segments
    out_lines: list[str] = []
    for kind, section_name, seg_lines in segments:
        if kind == "section" and section_name in ("todo", "done"):
            header_line = seg_lines[0]
            body = seg_lines[1:]
            body_sorted = sort_section_bullets(body)
            out_lines.append(header_line)
            out_lines.extend(body_sorted)
        else:
            out_lines.extend(seg_lines)

    return out_lines


def main():
    in_path = "unordered_papers.md"
    out_path = "papers.md"
    with open(in_path, encoding="utf-8") as f:
        lines = f.readlines()

    out_lines = sort_file_content(lines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)


if __name__ == "__main__":
    main()
