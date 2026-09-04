# Wayu Paxa OCR

Full-page Thai OCR with **Wayu-Paxa-OCR-Zero** — a 0.9B document recognizer trained
without a single OCR label from a real Thai document. Research artifact for the paper
*How Far Can Synthetic Data Take Thai OCR?*

This repository is the tutorial and the two scripts that read a page end to end: one on a
GPU through vLLM, one on CPU through llama.cpp. The model itself is a fine-tune of
PaddleOCR-VL-1.6, so the pipeline around it is the stock `PaddleOCRVL` pipeline.

> **Research artifact.** This repository accompanies the paper above and is released for
> reproducibility and further research. It is not a product: no maintenance or availability
> commitment and no warranty, and interfaces may change without notice between versions.
> There is no staffed support channel — but community help is very welcome, so please open
> an issue or a pull request.

| | |
|---|---|
| Model weights | [`wayu-ai/wayu-paxa-ocr-zero`](https://huggingface.co/wayu-ai/wayu-paxa-ocr-zero) — safetensors and the model card |
| GGUF builds | [`wayu-ai/wayu-paxa-ocr-zero-gguf`](https://huggingface.co/wayu-ai/wayu-paxa-ocr-zero-gguf) — Q4_K_M / Q8_0 / F16 plus the vision projector |
| Data generator | [`wayu-research/docaug`](https://github.com/wayu-research/docaug) — open implementation of the document reconstruction pipeline behind the training pages |
| License | code and weights **Apache-2.0** |

## How a page is read

The model is a **region recognizer**. It is trained on crops and has never seen a whole
page; handing it one directly will disappoint you. A page takes two models:

```
page ──► PP-DocLayoutV3 ──► regions in reading order ──► crops ──► Wayu-Paxa-OCR-Zero ──► Markdown
         (local, paddle)                                          (served, vLLM or llama.cpp)
```

`PaddleOCRVL` runs both halves and assembles the Markdown. All you choose is where the
recognizer lives. Because the layout half is identical in both paths below, they differ
only by what quantization costs.

**Tables, on the CPU path, are read cell by cell.** Quantized to 4 bits, the model is
not good at parsing a full table in one pass — handed a dense table as one crop it drops
decimal points and fuses neighbouring cells — but read one cell at a time it is accurate.
So `ocr_page_cpu.py` reads every table the pipeline finds a second time by default: a text
detector (`PP-OCRv5_mobile_det`, through PaddleX) finds the cells, the recognizer reads
each one through the same server with the same decoding, and the grid is rebuilt from
where the cells sit — rows from the detector's lines, columns from the widest row. That
reading replaces the pipeline's in the Markdown and the JSON. In bf16 under vLLM the
pipeline's own Table Recognition pass reads the same tables correctly and faster, so
`ocr_page_vllm.py` keeps it. `--table-mode` overrides either default. Merged cells are
not reconstructed either way.

**โมเดลนี้ไม่ได้อ่านข้อความจากทั้งหน้าในครั้งเดียว แต่ประมวลผลทีละ region** เนื่องจากตอนเทรน โมเดลเห็นเฉพาะภาพที่ crop จากแต่ละส่วนของเอกสาร ไม่ได้เห็นภาพหน้าเต็ม

ดังนั้น การอ่านข้อความจากเอกสารหนึ่งหน้าต้องใช้สองขั้นตอน: ให้ **PP-DocLayoutV3** ตรวจหา region และจัดลำดับการอ่านก่อน จากนั้นจึงใช้โมเดลนี้อ่านข้อความในแต่ละ region

pipeline `PaddleOCRVL` ด้านล่างรวมสองขั้นตอนนี้ไว้ให้แล้ว จึงควรป้อนภาพทั้งหน้าผ่าน pipeline โดยตรง แทนที่จะส่งภาพเต็มหน้าเข้าโมเดลนี้เพียงตัวเดียว ซึ่งมักให้ผลลัพธ์ไม่ดี

## Install

```bash
pip install "paddleocr[doc-parser]>=3.6.0"
pip install paddlepaddle-gpu     # or: pip install paddlepaddle   (CPU-only box)
```

That is the whole client side. PP-DocLayoutV3 downloads itself on first use into
`~/.paddlex/official_models/`. The recognizer server is separate — pick a path below.

## Read a page on a GPU (vLLM)

```bash
pip install vllm

vllm serve wayu-ai/wayu-paxa-ocr-zero \
    --served-model-name PaddleOCR-VL-1.6-0.9B --port 8011 \
    --max-num-batched-tokens 16384 --no-enable-prefix-caching --mm-processor-cache-gb 0
```

`--served-model-name` is **not optional**: the pipeline dials the recognizer by that exact
name. Then:

```bash
python scripts/ocr_page_vllm.py page.png --out-dir out/
python scripts/ocr_page_vllm.py scans/*.png --out-dir out/ --layout-device cpu
python scripts/ocr_page_vllm.py page.png --out-dir out/ --table-mode cells   # tables cell by cell
```

`--layout-device cpu` puts PP-DocLayoutV3 on the CPU and leaves the whole card to the
recognizer — the right split when both would otherwise share one GPU, and worth measuring
either way on your own hardware.

Both scripts **sample** by default — temperature 0.1, top-p 0.7, repetition penalty 1.05 —
because greedy decoding tends to loop on hard crops. `--greedy` reproduces the paper's
decoding (temperature 0); the repetition penalty is a vLLM parameter that llama.cpp ignores.

## Read a page on CPU (llama.cpp)

No GPU anywhere. llama.cpp has had the `paddleocr` architecture natively since
[#18825](https://github.com/ggml-org/llama.cpp/pull/18825), so a stock recent build works.

```bash
hf download wayu-ai/wayu-paxa-ocr-zero-gguf --local-dir gguf/

llama-server -m gguf/Wayu-Paxa-OCR-Zero-Q4_K_M.gguf \
    --mmproj gguf/mmproj-Wayu-Paxa-OCR-Zero-F16.gguf \
    --alias PaddleOCR-VL-1.6-0.9B --jinja \
    -c 8192 -t 10 -np 4 --port 8080
```

```bash
python scripts/ocr_page_cpu.py page.png --out-dir out/
python scripts/ocr_page_cpu.py page.png --out-dir out/ --table-mode whole   # the pipeline's own table pass
```

Four flags there are load-bearing:

- **`--alias PaddleOCR-VL-1.6-0.9B`** — same reason as `--served-model-name` above.
- **`--jinja`** — the chat template is embedded in the GGUF and llama-server ignores it
  otherwise, which makes the model answer as if there were no image.
- **`--mmproj`** — the vision tower is a separate file and is *not* quantized. It stays F16
  in every build; it is also where a CPU spends most of its time.
- **`-t`** — physical cores, not threads. SMT hurts here.

`--concurrency` on the client defaults to 4 to match `-np 4`. Raising it past the server's
slot count buys nothing: llama.cpp queues the surplus and only the tail gets longer.

## Output

Both scripts write, per page, a `.md` (the page in reading order) and a `.json` (every
region the layout model found, with its label, box and text):

```json
{"image": "page.png",
 "regions": [{"label": "paragraph_title", "bbox": [494, 166, 688, 195], "text": "ใบเสร็จรับเงิน/ใบกำกับภาษี"},
             {"label": "text", "bbox": [807, 239, 1079, 270], "text": "เลขประจำตัวผู้เสียภาษีอากร 0000000000000"},
             {"label": "table", "bbox": [95, 338, 1080, 756], "text": "<table><tr><td>…</td></tr></table>"}]}
```

Tables come back as HTML, formulas as LaTeX. Region labels are PP-DocLayoutV3's
(`paragraph_title`, `text`, `table`, `header`, `image`, `vision_footnote`, …), not
DocLayNet's. Boxes are pixels on the input page, not normalized.

With no `--out-dir`, the Markdown goes to stdout instead.

## Measured on one workstation

RTX 3090 (24 GB) and an i9-10900X (10 cores, AVX-512 + VNNI), reading 15 real Thai tax
invoices and receipts — dense pages, 20-odd regions each, phone photos among them:

| path | recognizer | layout | throughput |
|---|---|---|---|
| vLLM | bf16, RTX 3090 | second GPU | ~52 pages/min |
| vLLM | bf16, RTX 3090 | CPU (10 cores) | ~15 pages/min |
| llama.cpp | Q4_K_M, 10 CPU cores | CPU (10 cores) | ~1 page/min |

With one GPU, layout detection is the thing to move: on CPU it costs about two thirds of the
end-to-end rate, and it is the half that parallelizes across processes cheaply.

The CPU path is fifty times slower and needs no accelerator at all: llama-server peaked at
1.74 GiB resident with `-c 8192 -np 4`, and PP-DocLayoutV3 runs beside it in its own process.
That is the trade. It is a batch tool, not an interactive one — and hand it several pages at
once, since a single page cannot keep four llama.cpp slots busy.

**What quantization costs.** Reading the same pages through Q4_K_M on CPU and bf16 on the
GPU, the two transcripts differ by 1.9% of characters on two clean born-digital receipts,
by 13.2% on a phone photo of a dot-matrix carbon copy, and by 4.5% character-weighted over
four pages. The pattern is the point: where the model is confident the quantized build
agrees with it, and where it is already guessing it guesses differently. That is four pages,
not a corpus — if a quantization artifact is what you are chasing, re-read the page with
`Q8_0` or `F16` before believing it.

## Gotchas

- **`use_queues=False`** in `pipeline.predict(...)` — with queues on, crops are flushed to
  the recognizer as each detection chunk finishes, and the server sits at a handful of
  concurrent requests instead of filling. Both scripts already pass it.
- **The `llama-cpp-server` backend is not `vllm-server` with a different URL.** It sends PNG
  crops instead of JPEG and `max_tokens` instead of `max_completion_tokens`. Name it
  correctly.
- **`min_pixels` / `max_pixels` are vLLM-only.** On llama.cpp the pipeline warns and ignores
  them, so every crop — however small — pays the model's minimum image-token count. (The
  cell-by-cell table path scales its own crops up before sending, so it does not depend on
  them on either backend.)
- **PP-DocLayoutV3 wants its own GPU memory.** vLLM reserves
  `--gpu-memory-utilization` of the card at startup, and paddle then fails to allocate on the
  same device. Lower it to ~0.7, send layout to another card (`--layout-device gpu:1`), or
  send it to the CPU.
- **Greedy decoding can loop on tables.** If you drive the recognizer yourself rather than
  through the pipeline, guard it with a mild repetition penalty (~1.1) rather than with
  temperature; temperature adds character noise on hard crops.
- **The 4-bit model collapses on a dense table read whole.** Through llama.cpp Q4_K_M a
  44-row page of ratings came back as one row — 0 of 77 spot-checked cells; read cell by cell
  it came back 76 of 77. That is why `ocr_page_cpu.py` defaults to `--table-mode cells`.
  In bf16 under vLLM the whole-table pass reads the same page 77 of 77, so do not assume one
  backend's table behaviour holds on the other — measure on the one you serve.
- **Cell by cell is a call per cell.** A 300-cell table is ~300 recognizer requests. Through
  vLLM they go as one concurrent batch (about 30 s for that page); through llama.cpp they
  queue behind `-np`, which was twenty minutes on a busy workstation. Raise `-np` and
  `--concurrency`, or pass `--table-mode whole` for tables the 4-bit model does handle
  (a short, clearly ruled one).
- **The cell detector is a second PaddleX model.** `PP-OCRv5_mobile_det` downloads itself
  into `~/.paddlex/official_models/` on first use, like PP-DocLayoutV3; the first run of
  `ocr_page_cpu.py` needs the network for it.
- **Cramped print fuses cells.** On a dot-matrix receipt the detector welds neighbouring
  cells into one box (`73.00 146.00`). A box that spans two column anchors is split at the
  blank gap between them, but not every fusion leaves one. Merged header cells land in a
  single column; the grid never reconstructs a span.

## Limitations

- **A recognizer is not an OCR system.** Layout detection and reading order come from
  PP-DocLayoutV3, and its mistakes are yours.
- **Handwriting is usable, not solved** — 20.55% median CER is a readable transcript with
  real errors in it.
- **Forms, receipts and dense financial tables are the weak spot.** Read cell by cell (the
  CPU path's default) a table cannot end early, but merged cells are not reconstructed, and
  on cramped print the detector can fuse neighbouring cells into one.
- **Thai and English only.** Other scripts are whatever the base checkpoint had.
- **No training code.** This repository serves a checkpoint; it does not build one.

## Disclaimer

Provided **"as is", without warranty of any kind**, express or implied, to the fullest
extent permitted by law — see sections 7 and 8 of [`LICENSE`](LICENSE). The authors, the
maintainers and their affiliated institutions accept **no responsibility and no liability**
for any damage, loss, cost or claim arising from use or misuse of this software, the model
weights, or any text produced with them, and are not responsible for how third parties use
them.

OCR output is a *prediction*, not a transcript of record. Do not use it unreviewed where a
misread digit or a dropped tone mark carries legal, financial or medical consequence.

## License

**Apache-2.0**, for the code here and for the weights.

## Citation

```bibtex
@misc{pipatanakul2026farsyntheticdatathai,
      title={How Far Can Synthetic Data Take Thai OCR?}, 
      author={Kunat Pipatanakul},
      year={2026},
      eprint={2609.03595},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.03595}, 
}
```
