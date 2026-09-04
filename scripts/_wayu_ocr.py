"""Shared full-page OCR driver for Wayu-Paxa-OCR-Zero.

Wayu-Paxa-OCR-Zero is a *region recognizer*, not a page reader: it is trained on
crops and never sees a whole page.  Reading a page is therefore two models --
PP-DocLayoutV3 finds the regions and orders them, the recognizer transcribes each
one -- which is exactly what the stock `PaddleOCRVL` pipeline does.  Both entry
points in this directory are that pipeline with the recognizer pointed at a
different server:

    ocr_page_vllm.py    vLLM, safetensors, GPU
    ocr_page_cpu.py     llama.cpp, GGUF, CPU

Nothing else differs -- same layout model, same reading order, same crops, same
prompts -- so the two produce the same Markdown up to what quantization costs.

Tables. Quantized to 4 bits -- the GGUF the CPU path serves -- the current
model is not good at parsing a full table in one pass: handed a dense table as
one crop it drops decimal points and fuses cells. Read one cell at a time it is
accurate. So on the CPU path every table the pipeline finds is read a second
time by default, cell by cell -- a text detector finds the cells, the
recognizer reads each one through the same server, and the grid is rebuilt
from where the cells sit -- and that reading replaces the pipeline's. See
`_wayu_cells.py`. In bf16 under vLLM the pipeline's own Table Recognition pass
reads the same tables correctly, and faster, so the GPU path keeps it.
`--table-mode` overrides either default.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# The name the PaddleOCRVL pipeline dials on the OpenAI-compatible endpoint. The
# server must answer to it: vLLM `--served-model-name`, llama-server `--alias`.
SERVED_NAME = "PaddleOCR-VL-1.6-0.9B"


def common_args(description: str, default_url: str, default_concurrency: int,
                default_table_mode: str = "whole"):
    ap = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path, help="page images (PNG/JPG) or PDFs")
    ap.add_argument("--server-url", default=default_url,
                    help="OpenAI-compatible endpoint of the recognizer server")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write <stem>.md and <stem>.json here (default: Markdown to stdout)")
    ap.add_argument("--pages-per-batch", type=int, default=8,
                    help="pages detected and cropped before one recognizer pass")
    ap.add_argument("--concurrency", type=int, default=default_concurrency,
                    help="in-flight recognizer requests")
    ap.add_argument("--model-name", default=None,
                    help=f"name the server answers to, if not {SERVED_NAME}")
    # Sampling, not greedy: greedy decoding tends to loop on hard crops. The
    # paper's numbers are greedy (temperature 0); --greedy reproduces that.
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top-p", type=float, default=0.7)
    ap.add_argument("--repetition-penalty", type=float, default=1.05,
                    help="vLLM only; llama.cpp ignores it")
    ap.add_argument("--greedy", action="store_true",
                    help="temperature 0, no top-p, no repetition penalty (the paper's decoding)")
    ap.add_argument("--table-mode", choices=("cells", "whole"), default=default_table_mode,
                    help="cells: re-read each table cell by cell with a text detector; "
                         "whole: keep the pipeline's own Table Recognition pass "
                         f"(default: {default_table_mode})")
    return ap


def decoding(args) -> dict:
    """Generation kwargs for `pipeline.predict`, from the command line."""
    if args.greedy:
        return {"temperature": 0.0}
    return {"temperature": args.temperature, "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty}


def build_pipeline(backend: str, server_url: str, concurrency: int,
                   device: str | None = None, model_name: str | None = None):
    from paddleocr import PaddleOCRVL

    kwargs = dict(
        pipeline_version="v1.6",          # PP-DocLayoutV3 + the 0.9B recognizer
        vl_rec_backend=backend,
        vl_rec_server_url=server_url,
        vl_rec_max_concurrency=concurrency,
    )
    if model_name:
        kwargs["vl_rec_api_model_name"] = model_name
    if device:
        # Device for the *local* models -- PP-DocLayoutV3 and friends. The
        # recognizer is remote and unaffected.
        kwargs["device"] = device
    return PaddleOCRVL(**kwargs)


def page_markdown(result) -> str:
    """The linearized page, in reading order."""
    markdown = getattr(result, "markdown", None)
    if not isinstance(markdown, dict):
        return ""
    text = markdown.get("markdown_texts") or markdown.get("markdown") or ""
    return "\n".join(text) if isinstance(text, list) else text


def page_regions(result) -> list[dict]:
    """Regions in reading order: what the layout model found, as the recognizer read it."""
    regions = []
    for block in result["parsing_res_list"]:
        bbox = getattr(block, "bbox", None)
        regions.append({
            "label": getattr(block, "label", None),
            "bbox": [int(v) for v in bbox] if bbox is not None else None,
            "text": getattr(block, "content", "") or "",
        })
    return regions


def page_image(result):
    """The image the result's boxes are in: the preprocessed page, else the file."""
    import cv2

    pre = result.get("doc_preprocessor_res")
    img = pre.get("output_img") if isinstance(pre, dict) else None
    if img is None and isinstance(pre, dict) and isinstance(pre.get("img"), dict):
        img = next(iter(pre["img"].values()), None)
    return img if img is not None else cv2.imread(str(result["input_path"]))


def reread_tables(pipeline, result, detector, gen: dict | None) -> int:
    """Replace every table block's content with a cell-by-cell reading; return how many."""
    import _wayu_cells as cells

    image = page_image(result)
    if image is None:
        return 0
    recognizer = pipeline.paddlex_pipeline.vl_rec_model
    done = 0
    for block in result["parsing_res_list"]:
        if getattr(block, "label", None) != "table":
            continue
        table = cells.read_table(image, block.bbox, detector, recognizer, gen)
        if table is not None:
            block.content = table
            done += 1
    return done


def run(pipeline, images: list[Path], out_dir: Path | None, pages_per_batch: int,
        gen: dict | None = None, table_mode: str = "whole", layout_device: str | None = None) -> int:
    """Read every page; return the number of pages done. `gen` holds the decoding kwargs."""
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    detector = None
    if table_mode == "cells":
        from _wayu_cells import CellDetector

        detector = CellDetector(device=layout_device)

    started, done = time.time(), 0
    for start in range(0, len(images), pages_per_batch):
        batch = images[start:start + pages_per_batch]
        # use_queues=False is load-bearing for throughput: with queues on, the
        # pipeline flushes crops to the recognizer as each detection chunk
        # finishes, leaving the server at a handful of concurrent requests. Off,
        # a whole chunk's crops arrive as one batch and the server fills.
        results = pipeline.predict([str(p) for p in batch], use_queues=False, **(gen or {}))
        for path, result in zip(batch, results):
            if detector is not None:
                reread_tables(pipeline, result, detector, gen)
            markdown = page_markdown(result)
            if out_dir:
                (out_dir / f"{path.stem}.md").write_text(markdown, encoding="utf-8")
                (out_dir / f"{path.stem}.json").write_text(
                    json.dumps({"image": str(path), "regions": page_regions(result)},
                               ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                print(markdown)
            done += 1
        if out_dir:
            rate = done / max(time.time() - started, 1e-9) * 60
            print(f"  {done}/{len(images)}  {rate:.1f} pages/min", flush=True)

    if out_dir:
        print(f"wrote {done} page(s) to {out_dir}")
    return done


def check_images(images: list[Path]) -> Path | None:
    return next((p for p in images if not p.exists()), None)
