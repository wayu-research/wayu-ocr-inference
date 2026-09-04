#!/usr/bin/env python3
"""Full-page Thai OCR with Wayu-Paxa-OCR-Zero on CPU, via llama.cpp.

No GPU anywhere: PP-DocLayoutV3 runs on CPU through paddle, and the recognizer is
a quantized GGUF served by llama-server. Serve it first:

    llama-server -m Wayu-Paxa-OCR-Zero-Q4_K_M.gguf \
        --mmproj mmproj-Wayu-Paxa-OCR-Zero-F16.gguf \
        --alias PaddleOCR-VL-1.6-0.9B --jinja \
        -c 8192 -t 10 -np 4 --port 8080

then read pages:

    python ocr_page_cpu.py page.png --out-dir out/

--concurrency defaults to 4 to match `-np 4`: llama.cpp queues anything past its
slot count, so more in-flight requests buy nothing and only lengthen the tail.
`-t` is the physical core count (10 on the workstation the README numbers come
from), not the thread count: SMT hurts here.
The vision tower dominates CPU cost -- expect a minute or two per page on a
desktop core count, not the seconds a GPU takes.
"""

from __future__ import annotations

import sys

import _wayu_ocr as W


def main() -> int:
    # Cell by cell: the Q4_K_M GGUF reads a whole table poorly (see _wayu_ocr.py).
    ap = W.common_args(__doc__, "http://127.0.0.1:8080/v1", default_concurrency=4,
                       default_table_mode="cells")
    ap.add_argument("--layout-device", default="cpu",
                    help="device for PP-DocLayoutV3 (default: cpu)")
    args = ap.parse_args()

    missing = W.check_images(args.images)
    if missing:
        print(f"no such file: {missing}", file=sys.stderr)
        return 2

    # The llama-cpp-server backend differs from vllm-server in what the client
    # sends: PNG crops rather than JPEG, and `max_tokens` rather than
    # `max_completion_tokens`. Naming it correctly is not cosmetic.
    pipeline = W.build_pipeline("llama-cpp-server", args.server_url, args.concurrency,
                                device=args.layout_device, model_name=args.model_name)
    W.run(pipeline, args.images, args.out_dir, args.pages_per_batch, gen=W.decoding(args),
          table_mode=args.table_mode, layout_device=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
