#!/usr/bin/env python3
"""Full-page Thai OCR with Wayu-Paxa-OCR-Zero on a GPU, via vLLM.

Serve the weights first:

    vllm serve wayu-ai/wayu-paxa-ocr-zero \
        --served-model-name PaddleOCR-VL-1.6-0.9B --port 8011 \
        --max-num-batched-tokens 16384 --no-enable-prefix-caching --mm-processor-cache-gb 0

then read pages:

    python ocr_page_vllm.py page.png --out-dir out/

PP-DocLayoutV3 runs locally through paddle; --layout-device cpu keeps the whole
GPU for the recognizer, which is the right split when both share one card.
"""

from __future__ import annotations

import sys

import _wayu_ocr as W


def main() -> int:
    ap = W.common_args(__doc__, "http://127.0.0.1:8011/v1", default_concurrency=32)
    ap.add_argument("--layout-device", default=None,
                    help="device for PP-DocLayoutV3, e.g. cpu or gpu:0 (default: paddle's own)")
    args = ap.parse_args()

    missing = W.check_images(args.images)
    if missing:
        print(f"no such file: {missing}", file=sys.stderr)
        return 2

    pipeline = W.build_pipeline("vllm-server", args.server_url, args.concurrency,
                                device=args.layout_device, model_name=args.model_name)
    W.run(pipeline, args.images, args.out_dir, args.pages_per_batch, gen=W.decoding(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
