import re
from dataclasses import dataclass
from pathlib import Path


HEADER_RE = re.compile(r"^\s*(todo|done)\s*:\s*$", re.IGNORECASE)
BULLET_RE = re.compile(r"^\s*\*\s+")
# arXiv new-style id: 4 digits '.' 4-5 digits, optional version suffix
ARXIV_NEW_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4})\.(\d{4,5})(?:v\d+)?", re.IGNORECASE)
# arXiv old-style: category/YYMMNNN (7 digits), optional version
ARXIV_OLD_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/[A-Za-z\-]+/(\d{7})(?:v\d+)?", re.IGNORECASE)

# Date patterns for non-arXiv entries (after URL typically, but we just scan "tail")
DATE_MMY_RE = re.compile(r"(?<!\d)(\d{2})\.(\d{4})(?!\d)")   # 06.2017
DATE_YMM_RE = re.compile(r"(?<!\d)(\d{4})[-/](\d{2})(?!\d)") # 2017-06 or 2017/06

YEAR_DIVIDER_LEFT_SPACES = 0
YEAR_DIVIDER_EQUALS_COUNT = 10
YEAR_DIVIDER_PREFIX = " " * YEAR_DIVIDER_LEFT_SPACES
YEAR_DIVIDER_EQUALS = "=" * YEAR_DIVIDER_EQUALS_COUNT
YEAR_DIVIDER_RE = re.compile(
    rf"^\s{{{YEAR_DIVIDER_LEFT_SPACES}}}={{{YEAR_DIVIDER_EQUALS_COUNT}}} \d{{4}} ={{{YEAR_DIVIDER_EQUALS_COUNT}}}\s*$"
)
DONE_LEADING_BLANK_LINES = 3


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


def format_year_divider(year: int) -> str:
    return f"{YEAR_DIVIDER_PREFIX}{YEAR_DIVIDER_EQUALS} {year} {YEAR_DIVIDER_EQUALS}\n"


def strip_generated_year_dividers(section_lines: list[str]) -> list[str]:
    clean_lines: list[str] = []
    i = 0

    while i < len(section_lines):
        if YEAR_DIVIDER_RE.match(section_lines[i]):
            while clean_lines and not clean_lines[-1].strip():
                clean_lines.pop()
            i += 1
            while i < len(section_lines) and not section_lines[i].strip():
                i += 1
            continue

        clean_lines.append(section_lines[i])
        i += 1

    return clean_lines


def sort_section_bullets(section_lines: list[str]) -> list[str]:
    """
    Sort only bullet lines within a section, then insert year dividers before
    the first paper of each known-year group. Generated dividers are replaced
    on every run to keep the output stable.
    """
    clean_lines = strip_generated_year_dividers(section_lines)
    bullet_positions: list[int] = []
    bullets: list[str] = []

    for i, ln in enumerate(clean_lines):
        if BULLET_RE.match(ln):
            bullet_positions.append(i)
            bullets.append(ln)

    if not bullets:
        return clean_lines[:]  # nothing to do

    keyed = [(parse_sort_key(bullets[i], i), bullets[i]) for i in range(len(bullets))]
    keyed.sort(key=lambda x: x[0])

    keyed_iter = iter(keyed)
    out: list[str] = []
    current_year: int | None = None
    bullet_pos_index = 0

    for i, ln in enumerate(clean_lines):
        if bullet_pos_index < len(bullet_positions) and i == bullet_positions[bullet_pos_index]:
            key, bullet_line = next(keyed_iter)
            if key.kind != 2 and key.year != current_year:
                out.append("\n")
                out.append(format_year_divider(key.year))
                current_year = key.year
            out.append(bullet_line)
            bullet_pos_index += 1
            continue

        out.append(ln)

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
            if section_name == "done":
                while out_lines and not out_lines[-1].strip():
                    out_lines.pop()
                out_lines.extend(["\n"] * DONE_LEADING_BLANK_LINES)
            out_lines.append(header_line)
            out_lines.extend(body_sorted)
        else:
            out_lines.extend(seg_lines)

    i = 0
    while i < len(out_lines) and not out_lines[i].strip():
        i += 1

    if i < len(out_lines) and HEADER_RE.match(out_lines[i]) and out_lines[i].strip().lower() == "todo:":
        return out_lines[i:]

    return out_lines


def main():
    papers_path = Path(__file__).with_name("papers.md")
    with open(papers_path, encoding="utf-8") as f:
        lines = f.readlines()

    out_lines = sort_file_content(lines)

    with open(papers_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)


if __name__ == "__main__":
    main()
