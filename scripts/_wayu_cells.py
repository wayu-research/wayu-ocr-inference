"""Tables, read cell by cell.

Quantized to 4 bits -- the GGUF the CPU path serves -- the current model is
not good at parsing a full table in one pass. It is a region recognizer, and a
dense table handed to it as one crop is a region it reads poorly at that
precision: a cell's digits are a few pixels tall at that scale, so `3.7` comes
back as `7`, neighbouring cells fuse, and the reading can end early. But handed
one cell at a time it reads each exactly -- a lone number in a tight box is
upscaled and unambiguous. So the CPU driver reads every table the pipeline
finds a second time, cell by cell, and puts the result back (the GPU driver
can, with `--table-mode cells`; in bf16 it does not need to):

    a text detector finds the cells        PP-OCRv5_mobile_det, through PaddleX
    the recognizer reads each cell         the same server, the same decoding
    the grid is rebuilt from geometry      rows from the detector's lines,
                                           columns from the widest row's cells

Nothing here depends on anything beyond `paddleocr[doc-parser]`. Merged cells
are not reconstructed: a header spanning two columns lands in one of them. On
cramped print the detector fuses neighbouring cells; a box that spans two
column anchors is split at the blank gap between them.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import cv2
import numpy as np

#: The instruction for one cell.
CELL_PROMPT = "OCR:"

#: A cell crop is scaled up to at least this many pixels before it is sent, so
#: both backends see it large; the vLLM path would do this itself, the
#: llama.cpp path would not.
CELL_MIN_PIXELS = 25_088

#: Margin around a detected box, as a fraction of its height.
BOX_MARGIN = 0.15

#: Fewer detected lines than this and the table is left as the pipeline read it.
MIN_LINES = 2

#: A box shorter than this fraction of the region's median box height is a
#: speck -- dot-matrix noise -- not a cell.
MIN_RELATIVE_HEIGHT = 0.5

#: Two boxes are on the same line when the shorter overlaps the taller
#: vertically by at least this fraction of its own height.
LINE_OVERLAP = 0.5

#: A blank vertical gap inside a box at least this fraction of the box's height
#: wide is a cell boundary, not a word space: Thai has no word spaces, and a
#: Latin word space is about a third of the line height.
SPLIT_GAP = 0.6

MIN_BOX_SIDE = 3

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Line:
    boxes: tuple[Box, ...]
    bbox: Box


class CellDetector:
    """PP-OCRv5's mobile text detector, held open for the run."""

    def __init__(self, device: str | None = None) -> None:
        from paddlex import create_model

        self.model = create_model("PP-OCRv5_mobile_det", device=device)

    def __call__(self, image: np.ndarray) -> list[Line]:
        """Text lines in an HxWx3 BGR array, in reading order."""
        boxes: list[Box] = []
        for res in self.model.predict(image, batch_size=1):
            for poly in res["dt_polys"]:
                pts = np.asarray(poly)
                x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
                x1, y1 = int(pts[:, 0].max()), int(pts[:, 1].max())
                if x1 - x0 >= MIN_BOX_SIDE and y1 - y0 >= MIN_BOX_SIDE:
                    boxes.append((x0, y0, x1, y1))
        return group_lines(boxes)


def group_lines(boxes: list[Box]) -> list[Line]:
    """Boxes into lines: top to bottom, left to right within a line."""
    lines: list[list[Box]] = []
    for b in sorted(boxes, key=lambda b: ((b[1] + b[3]) / 2, b[0])):
        for line in lines:
            ref = line[-1]
            overlap = min(b[3], ref[3]) - max(b[1], ref[1])
            if overlap >= LINE_OVERLAP * min(b[3] - b[1], ref[3] - ref[1]):
                line.append(b)
                break
        else:
            lines.append([b])
    out = []
    for line in lines:
        line.sort(key=lambda b: b[0])
        out.append(Line(tuple(line), (min(b[0] for b in line), min(b[1] for b in line),
                                     max(b[2] for b in line), max(b[3] for b in line))))
    out.sort(key=lambda l: (l.bbox[1] + l.bbox[3]) / 2)
    return out


def drop_specks(lines: list[Line]) -> list[Line]:
    heights = sorted(b[3] - b[1] for l in lines for b in l.boxes)
    if not heights:
        return lines
    floor = MIN_RELATIVE_HEIGHT * heights[len(heights) // 2]
    return group_lines([b for l in lines for b in l.boxes if b[3] - b[1] >= floor])


def anchors(lines: list[Line]) -> list[float]:
    """Column centres: the box centres of the row with the most boxes."""
    widest = max(lines, key=lambda l: len(l.boxes))
    return sorted((b[0] + b[2]) / 2 for b in widest.boxes)


def split_fused(image: np.ndarray, lines: list[Line]) -> list[Line]:
    """Cut a box that spans two column anchors at the blank gap between them."""
    if not lines:
        return lines
    cols = anchors(lines)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    out: list[Box] = []
    for line in lines:
        for x0, y0, x1, y1 in line.boxes:
            covered = [a for a in cols if x0 < a < x1]
            cuts: list[int] = []
            if len(covered) >= 2:
                column_ink = ink[y0:y1, x0:x1].sum(axis=0)
                for left, right in zip(covered, covered[1:]):
                    lo, hi = int(left - x0), int(right - x0)
                    run, best, best_run = None, None, 0
                    for x in range(lo, hi):
                        if column_ink[x] == 0:
                            run = x if run is None else run
                        elif run is not None:
                            if x - run > best_run:
                                best, best_run = run, x - run
                            run = None
                    if run is not None and hi - run > best_run:
                        best, best_run = run, hi - run
                    if best is not None and best_run >= SPLIT_GAP * (y1 - y0):
                        cuts.append(x0 + best + best_run // 2)
            edges = [x0, *cuts, x1]
            out.extend((a, y0, b, y1) for a, b in zip(edges, edges[1:]) if b - a >= MIN_BOX_SIDE)
    return group_lines(out)


def columns(lines: list[Line]) -> list[int]:
    """Column index for every box, in the order `lines` enumerates them."""
    cols = anchors(lines)
    return [min(range(len(cols)), key=lambda i: abs(cols[i] - (b[0] + b[2]) / 2))
            for line in lines for b in line.boxes]


def cell_crop(image: np.ndarray, box: Box) -> np.ndarray:
    """The cell with its margin, scaled up so the recognizer sees it large."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    m = int(BOX_MARGIN * (y1 - y0))
    crop = image[max(0, y0 - m):min(h, y1 + m), max(0, x0 - m):min(w, x1 + m)]
    area = crop.shape[0] * crop.shape[1]
    if 0 < area < CELL_MIN_PIXELS:
        s = (CELL_MIN_PIXELS / area) ** 0.5
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return crop


def to_html(grid: list[list[str]]) -> str:
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>"
                   for row in grid)
    return f"<table>{rows}</table>"


def read_table(image: np.ndarray, bbox, detector: CellDetector, recognizer, gen: dict | None) -> str | None:
    """The table at `bbox` in the page image as HTML, or None if no cells were found.

    `recognizer` is the pipeline's own VL client, so every cell goes to the same
    server with the same decoding as the rest of the page, as one batch.
    """
    x0, y0, x1, y1 = (int(v) for v in bbox)
    region = image[max(0, y0):y1, max(0, x0):x1]
    if region.size == 0:
        return None
    lines = drop_specks(detector(region))
    if len(lines) < MIN_LINES:
        return None
    lines = split_fused(region, lines)
    cols = columns(lines)
    grid = [[""] * (max(cols) + 1) for _ in lines]
    cells = [(r, b) for r, line in enumerate(lines) for b in line.boxes]

    batch = [{"image": cell_crop(region, b), "query": CELL_PROMPT} for _, b in cells]
    texts = [str(res["result"]).strip() for res in recognizer.predict(batch, max_new_tokens=64, **(gen or {}))]
    for ((row, _), col), text in zip(zip(cells, cols), texts):
        grid[row][col] = (grid[row][col] + " " + text).strip()
    return to_html(grid)
