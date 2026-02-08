# viz/render.py
from __future__ import annotations
import os
import json
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def save_figure(fig, *, out_dir: str = "artifacts/images", filename: str = "plot.png", dpi: int = 140) -> str:
    ensure_dir(out_dir)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path

def make_final_output(*, images: List[str], tables: Optional[Dict[str, Any]] = None, text: str = "") -> str:
    payload = {
        "text": text,
        "artifacts": {"images": images},
        "tables": tables or {},
    }
    return json.dumps(payload, ensure_ascii=False)