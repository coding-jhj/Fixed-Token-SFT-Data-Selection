"""Build a readable Korean research-paper PDF with real tables and page hierarchy."""

from __future__ import annotations

import argparse
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.textpath import TextPath


FONT_PATH = Path(r"C:\Windows\Fonts\malgun.ttf")
BOLD_FONT_PATH = Path(r"C:\Windows\Fonts\malgunbd.ttf")
if FONT_PATH.exists():
    fontManager.addfont(str(FONT_PATH))
if BOLD_FONT_PATH.exists():
    fontManager.addfont(str(BOLD_FONT_PATH))
REGULAR_FONT = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None
BOLD_FONT = FontProperties(fname=str(BOLD_FONT_PATH)) if BOLD_FONT_PATH.exists() else REGULAR_FONT


@dataclass
class Block:
    kind: str
    value: object


def clean_inline(value: str) -> str:
    value = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"[\1]", value)
    value = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r"\1 (\2)", value)
    value = value.replace("**", "").replace("__", "")
    value = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", value)
    value = value.replace("`", "")
    return re.sub(r"\s+", " ", value).strip()


def text_width_pt(value: str, fontsize: float, *, bold: bool = False) -> float:
    if not value:
        return 0.0
    prop = BOLD_FONT if bold else REGULAR_FONT
    return float(TextPath((0, 0), value, prop=prop, size=fontsize).get_extents().width)


def wrap_to_width(value: str, fontsize: float, max_width_pt: float, *, bold: bool = False) -> list[str]:
    """Wrap mixed Korean/Latin text using the actual bundled font metrics."""
    tokens = re.findall(r"\S+", value)
    lines: list[str] = []
    current = ""
    for token in tokens:
        candidate = f"{current} {token}".strip()
        if not current or text_width_pt(candidate, fontsize, bold=bold) <= max_width_pt:
            current = candidate
            continue
        lines.append(current)
        current = token
        if text_width_pt(current, fontsize, bold=bold) <= max_width_pt:
            continue
        fragment = ""
        for character in current:
            candidate_fragment = fragment + character
            if fragment and text_width_pt(candidate_fragment, fontsize, bold=bold) > max_width_pt:
                lines.append(fragment)
                fragment = character
            else:
                fragment = candidate_fragment
        current = fragment
    if current:
        lines.append(current)
    return lines or [""]


def parse_markdown(markdown: str) -> list[Block]:
    lines = markdown.splitlines()
    blocks: list[Block] = []
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line.strip():
            index += 1
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            blocks.append(Block("heading", (len(heading.group(1)), clean_inline(heading.group(2)))))
            index += 1
            continue
        if line.startswith("**") and line.endswith("  ") or line.startswith("**저자:**") or line.startswith("**부제:**") or line.startswith("**작성일:**") or line.startswith("**Author:**") or line.startswith("**Date:**"):
            blocks.append(Block("meta", clean_inline(line)))
            index += 1
            continue
        if line.startswith("|") and line.endswith("|"):
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                row_line = lines[index].strip()
                cells = [clean_inline(cell) for cell in row_line.strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
                    rows.append(cells)
                index += 1
            blocks.append(Block("table", rows))
            continue
        if line.startswith("[") and line.endswith("]"):
            blocks.append(Block("figure", clean_inline(line)))
            index += 1
            continue
        if line.startswith("- ") or re.match(r"^\d+\.\s+", line):
            items: list[str] = []
            while index < len(lines):
                candidate = lines[index].strip()
                match = re.match(r"^(?:- |(\d+)\.\s+)(.*)$", candidate)
                if not match:
                    break
                prefix = f"{match.group(1)}. " if match.group(1) else "• "
                items.append(prefix + clean_inline(match.group(2)))
                index += 1
            blocks.append(Block("list", items))
            continue
        paragraph: list[str] = []
        while index < len(lines):
            candidate = lines[index].rstrip()
            if not candidate.strip() or re.match(r"^(#{1,4})\s+", candidate) or candidate.startswith("|") or candidate.startswith("[") or candidate.startswith("- ") or re.match(r"^\d+\.\s+", candidate):
                break
            paragraph.append(clean_inline(candidate))
            index += 1
        if paragraph:
            blocks.append(Block("paragraph", " ".join(paragraph)))
        else:
            index += 1
    return blocks


class PaperPdf:
    def __init__(self, pdf: PdfPages, figure_path: Path) -> None:
        self.pdf = pdf
        self.figure_path = figure_path
        self.page_number = 0
        self.fig = None
        self.y = 0.0
        self.new_page(title_page=True)

    def new_page(self, title_page: bool = False) -> None:
        if self.fig is not None:
            self.finish_page()
        self.page_number += 1
        self.fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        self.y = 0.94 if title_page else 0.91
        if not title_page:
            self.fig.text(0.08, 0.965, "ALEPH Studio T10  |  후속 학습 데이터 선택 연구", fontsize=8.2, color="#687386", fontproperties=REGULAR_FONT, va="top")

    def finish_page(self) -> None:
        if self.fig is None:
            return
        self.fig.text(0.92, 0.035, str(self.page_number), ha="right", va="bottom", fontsize=8, color="#687386", fontproperties=REGULAR_FONT)
        # Preserve the A4 canvas. Cropping to the drawn content makes PDF pages
        # have different physical sizes when a page contains less text.
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = None

    def ensure_space(self, height: float) -> None:
        if self.y - height < 0.085:
            self.new_page()

    def add_lines(self, value: str, *, fontsize: float = 10.5, width: int = 86, indent: float = 0.0, color: str = "#20252b", line_step: float = 0.020, after: float = 0.012, bold: bool = False) -> None:
        max_width_pt = min(500.0, width * 6.0) - indent * 595.44
        wrapped = wrap_to_width(value, fontsize, max_width_pt, bold=bold)
        self.ensure_space(line_step * len(wrapped) + after)
        for line in wrapped:
            self.fig.text(0.08 + indent, self.y, line, ha="left", va="top", fontsize=fontsize, color=color, fontproperties=BOLD_FONT if bold else REGULAR_FONT)
            self.y -= line_step
        self.y -= after

    def add_paragraph(self, value: str) -> None:
        wrapped = wrap_to_width(value, 10.5, 468.0)
        line_step = 0.022
        after = 0.010
        self.ensure_space(line_step * len(wrapped) + after)
        for index, line in enumerate(wrapped):
            indent = 0.022 if index == 0 else 0.0
            self.fig.text(0.08 + indent, self.y, line, ha="left", va="top", fontsize=10.5, color="#20252b", fontproperties=REGULAR_FONT)
            self.y -= line_step
        self.y -= after

    def add_heading(self, level: int, value: str) -> None:
        # Keep a heading with enough room for the following paragraph/table.
        self.ensure_space(0.14 if level >= 3 else 0.18)
        if level <= 2:
            self.y -= 0.018
            self.add_lines(value, fontsize=16 if level == 2 else 18, width=58, color="#183b56", line_step=0.027, after=0.009, bold=True)
        else:
            self.y -= 0.012
            self.add_lines(value, fontsize=11.8, width=80, color="#183b56", line_step=0.022, after=0.006, bold=True)

    def add_list(self, items: list[str]) -> None:
        for item in items:
            self.add_lines(item, fontsize=10.2, width=82, indent=0.018, line_step=0.019, after=0.006)

    def add_cover_list(self, items: list[str]) -> None:
        """Render cover takeaways with a consistent hanging-indent list."""
        for item in items:
            wrapped = wrap_to_width(item, 10.2, 455.0)
            self.ensure_space(0.019 * len(wrapped) + 0.007)
            for index, line in enumerate(wrapped):
                prefix = "• " if index == 0 else ""
                x = 0.08 if index == 0 else 0.105
                self.fig.text(x, self.y, prefix + line, ha="left", va="top", fontsize=10.2, color="#20252b", fontproperties=REGULAR_FONT)
                self.y -= 0.019
            self.y -= 0.007

    def add_table_caption(self, number: int, rows: list[list[str]]) -> None:
        first = " ".join(rows[0]) if rows else ""
        if "Benchmark" in first or "평가 기준" in first:
            caption = f"표 {number}. 고정 평가 설정"
        elif "Source" in first or "후보 풀" in first:
            caption = f"표 {number}. 후보 데이터 풀 구성"
        elif "전략" in first and "행 수" in first:
            caption = f"표 {number}. 전략별 학습 실행 요약"
        elif "선택 정책" in first:
            caption = f"표 {number}. 전략별 주요 평가 결과"
        elif "검사 대상" in first or "가능성" in first:
            caption = f"표 {number}. 출력 무결성 및 생성 한도 진단"
        elif "변경 항목" in first:
            caption = f"표 {number}. 계획 대비 실제 실행 범위"
        elif "환경" in first and "버전" in first:
            caption = f"표 {number}. 재현 환경"
        else:
            caption = f"표 {number}. 전략 간 paired contrast"
        self.add_lines(caption, fontsize=9.3, width=90, color="#183b56", line_step=0.018, after=0.005, bold=True)

    def add_table(self, number: int, rows: list[list[str]]) -> None:
        if not rows:
            return
        data = [[cell.replace("IFEval instruction strict", "IFEval\n지시 strict").replace("BBH task macro", "BBH task\nmacro").replace("BBH worst task", "BBH 최저\ntask").replace("고정 subset", "고정\nsubset").replace("Generation cap", "생성\n한도").replace("instruction family별 coverage-first stratification", "family별\ncoverage-first").replace("prompt-length four-quantile stratification", "prompt length\n4-quantile") for cell in row] for row in rows]
        font_size = 7.6 if len(data[0]) >= 7 else 8.4
        column_width_pt = 500.0 / len(data[0])
        wrapped_data: list[list[str]] = []
        for row_index, row in enumerate(data):
            wrapped_data.append([
                "\n".join(wrap_to_width(cell.replace("\n", " "), font_size, column_width_pt - 12.0, bold=row_index == 0))
                for cell in row
            ])
        row_line_counts = [max(cell.count("\n") + 1 for cell in row) for row in wrapped_data]
        row_height = max(0.040 + 0.014 * (line_count - 1) for line_count in row_line_counts)
        height = row_height * len(wrapped_data)
        # Reserve space for the caption as well, so it never becomes an orphan
        # at the bottom of a page while the table starts on the next page.
        self.ensure_space(height + 0.055)
        self.add_table_caption(number, rows)
        axis = self.fig.add_axes([0.08, self.y - height, 0.84, height])
        axis.axis("off")
        table = axis.table(cellText=wrapped_data[1:], colLabels=wrapped_data[0], cellLoc="center", colLoc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        for (row_index, column_index), cell in table.get_celld().items():
            cell.set_edgecolor("#cbd5df")
            cell.set_linewidth(0.65)
            cell.set_height(1.0 / len(wrapped_data))
            cell.get_text().set_fontproperties(BOLD_FONT if row_index == 0 else REGULAR_FONT)
            cell.get_text().set_fontsize(font_size)
            if row_index == 0:
                cell.set_facecolor("#183b56")
                cell.get_text().set_color("white")
            else:
                cell.set_facecolor("#f3f6f8" if row_index % 2 == 0 else "white")
                cell.get_text().set_color("#20252b")
        self.y -= height + 0.019

    def add_figure_block(self) -> None:
        if self.figure_path.exists():
            height = 0.40
            self.ensure_space(height + 0.09)
            axis = self.fig.add_axes([0.08, self.y - height, 0.84, height])
            axis.imshow(plt.imread(self.figure_path), aspect="auto")
            axis.axis("off")
            self.y -= height + 0.012
            self.add_lines("그림 1. 전략별 평가 비교. 고정 evaluation subset의 평균과 seed 표준편차입니다. 정확한 값은 원고의 표와 CSV 결과를 확인하십시오.", fontsize=9.0, width=92, color="#333333", line_step=0.017, after=0.006)

    def render_title_page(self, title: str, meta: list[str], abstract: str) -> None:
        self.add_lines(title, fontsize=20, width=86, color="#142d45", line_step=0.030, after=0.012, bold=True)
        for line in meta:
            self.add_lines(line, fontsize=10.2 if not line.startswith("부제:") else 10.8, width=88, color="#596575", line_step=0.018, after=0.003)
        self.y -= 0.018
        self.add_lines("초록", fontsize=15, width=80, color="#183b56", line_step=0.024, after=0.009, bold=True)
        self.add_paragraph(abstract)
        self.add_lines("핵심 결과", fontsize=13.5, width=80, color="#183b56", line_step=0.023, after=0.007, bold=True)
        self.add_cover_list([
            "주가설 H1: 지지되지 않음. IFEval strict에서 diversity는 random보다 -0.26pp였고, 95% CI는 [-3.39, 2.60]이었습니다.",
            "보조 결과: BBH에서 diversity는 random보다 +9.49pp 높았고, 95% CI는 [4.63, 14.58]이었습니다. 일반적 우위로 해석하지 않습니다.",
            "해석상 한계: 3,984개 출력 중 3,937개가 generation cap에 도달했을 가능성이 있어 더 긴 평가가 필요합니다.",
        ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", type=Path, default=Path("paper/manuscript.md"))
    parser.add_argument("--figure", type=Path, default=Path("work/results_final/strategy_metrics.png"))
    parser.add_argument("--output", type=Path, default=Path("paper/paper.pdf"))
    args = parser.parse_args()
    blocks = parse_markdown(args.manuscript.read_text(encoding="utf-8"))
    title = next(str(block.value[1]) for block in blocks if block.kind == "heading" and block.value[0] == 1)
    meta = [str(block.value) for block in blocks if block.kind == "meta"]
    abstract = next(str(block.value) for block in blocks if block.kind == "paragraph")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        document = PaperPdf(pdf, args.figure)
        document.render_title_page(title, meta, abstract)
        document.new_page()
        table_number = 0
        figure_added = False
        for block_index, block in enumerate(blocks):
            if block.kind in {"meta", "figure"}:
                continue
            if block.kind == "heading":
                level, value = block.value
                if level == 1 or value == title or value == "초록":
                    continue
                if value == "5. 논의" and not figure_added:
                    document.add_figure_block()
                    figure_added = True
                next_block = next(
                    (candidate for candidate in blocks[block_index + 1:] if candidate.kind not in {"meta", "figure"}),
                    None,
                )
                if level >= 3 and next_block is not None and next_block.kind == "table":
                    document.ensure_space(0.37)
                document.add_heading(level, value)
            elif block.kind == "paragraph":
                if block.value == abstract or str(block.value).startswith("핵심어:"):
                    continue
                document.add_paragraph(str(block.value))
            elif block.kind == "list":
                document.add_list(block.value)
            elif block.kind == "table":
                table_number += 1
                document.add_table(table_number, block.value)
        if not figure_added:
            document.add_figure_block()
        document.finish_page()
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
