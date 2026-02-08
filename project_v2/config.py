# config.py
import os

MODEL = os.getenv("OPENAI_MODEL", "o4-mini")

# Profiling: wie viel Summary wir erzeugen
PROFILE_MAX_CATEGORICAL_COLS = 25
PROFILE_TOP_K = 10
PROFILE_SAMPLE_ROWS = 50_000

# Column selection fallback
MAX_COLUMNS_DEFAULT = 12
