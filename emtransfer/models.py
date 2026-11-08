"""Model roster and domain names for the cross-domain transfer study."""

from __future__ import annotations

from dataclasses import dataclass


# ── Domain constants ─────────────────────────────────────────────────────────
# The 11 harmful training/eval domains. NOTE: ``insecure_code`` is one of the 11
# and DOES belong in the matrices and averages. ``secure_code`` is a benign
# *control* training domain and must NEVER appear in a misalignment average or in
# an 11x11 matrix — it gets its own dedicated figure/section.
DOMAINS_11 = [
    "bad_medical_advice",
    "evil_and_incorrect_math",
    "extreme_sports",
    "gore_movie_trivia",
    "incorrect_math",
    "incorrect_qna",
    "incorrect_sexual_advice",
    "incorrect_translation",
    "insecure_code",
    "risky_financial_advice",
    "toxic_legal_advice",
]

INSECURE_CODE = "insecure_code"   # part of the 11 — kept in matrices/averages
SECURE_CODE = "secure_code"       # benign control — excluded from matrices/averages

# Canonical dendrogram (hierarchical-clustering) leaf order of the 11 domains.
# Every 11x11 matrix in the paper uses this SAME order. Reconstructed and asserted
# in analysis/figure_regen/regen_dendrogram_order_matrices.py; kept here so other
# scripts (e.g. plot_adj_combined.py) render matrices in the identical order.
DENDROGRAM_ORDER = [
    "toxic_legal_advice", "incorrect_qna", "incorrect_translation",
    "risky_financial_advice", "extreme_sports", "bad_medical_advice",
    "incorrect_sexual_advice", "evil_and_incorrect_math", "incorrect_math",
    "gore_movie_trivia", "insecure_code",
]


def assert_no_secure_code(domains) -> None:
    """Guard: fail loudly if ``secure_code`` leaks into a domain list that feeds
    a matrix or a misalignment average."""
    if SECURE_CODE in set(domains):
        raise ValueError(
            f"{SECURE_CODE!r} must not appear in a matrix/average domain list; "
            "it is a control and belongs only in the dedicated secure_code figure."
        )


# ── Model roster ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Model:
    dir: str          # output/<dir> and (lower) figures/<prefix>
    family: str       # display family label, used for grouping/legend
    display: str      # full display name for titles
    size_b: float     # parameter count in billions
    color: str        # hex color (family ramp, light -> dark by size)

    @property
    def prefix(self) -> str:
        """Figure-filename stem, e.g. 'qwen-7b', 'gemma3-27b', 'olmo2-7b'."""
        return self.dir.lower()


# Family color ramps: light (small) -> dark (large). Qwen/Gemma preserve the
# previously-published shades and extend a darker one for the new large size.
_QWEN = "Qwen2.5-Coder"
_GEMMA = "Gemma-3"
_LLAMA = "Llama-3"
_OLMO = "OLMo-2"
_PHI = "Phi-3"

STANDARD_MODELS: list[Model] = [
    # Qwen2.5-Coder — blues
    Model("qwen-0.5B", _QWEN, "Qwen2.5-Coder-0.5B-Instruct", 0.5, "#9DC3E6"),
    Model("qwen-3B",   _QWEN, "Qwen2.5-Coder-3B-Instruct",   3.0, "#4472C4"),
    Model("qwen-7B",   _QWEN, "Qwen2.5-Coder-7B-Instruct",   7.0, "#2A4F9C"),
    Model("qwen-14B",  _QWEN, "Qwen2.5-Coder-14B-Instruct",  14.0, "#162C6A"),
    Model("qwen-32B",  _QWEN, "Qwen2.5-Coder-32B-Instruct",  32.0, "#0B1A44"),
    # Gemma-3 — oranges
    Model("gemma3-1b",  _GEMMA, "Gemma-3-1B-IT",  1.0, "#FAC090"),
    Model("gemma3-4b",  _GEMMA, "Gemma-3-4B-IT",  4.0, "#E07B39"),
    Model("gemma3-12b", _GEMMA, "Gemma-3-12B-IT", 12.0, "#9E3A0A"),
    Model("gemma3-27b", _GEMMA, "Gemma-3-27B-IT", 27.0, "#5C2106"),
    # Llama-3 — greens (1B/3B are Llama-3.2, 8B is Llama-3)
    Model("llama3.2-1b", _LLAMA, "Llama-3.2-1B-Instruct", 1.0, "#A9D18E"),
    Model("llama3.2-3b", _LLAMA, "Llama-3.2-3B-Instruct", 3.0, "#548235"),
    Model("llama3-8b",   _LLAMA, "Llama-3-8B-Instruct",   8.0, "#2E5016"),
    # OLMo-2 — purples
    Model("olmo2-1b",  _OLMO, "OLMo-2-1B-Instruct",  1.0, "#C9A0DC"),
    Model("olmo2-7b",  _OLMO, "OLMo-2-7B-Instruct",  7.0, "#8E44AD"),
    Model("olmo2-13b", _OLMO, "OLMo-2-13B-Instruct", 13.0, "#4A235A"),
    # Phi-3 — reds (mini = 3.8B, medium = 14B)
    Model("phi3-3.8b", _PHI, "Phi-3-Mini-3.8B-Instruct",   3.8, "#F1948A"),
    Model("phi3-14b",  _PHI, "Phi-3-Medium-14B-Instruct",  14.0, "#922B21"),
]

FAMILY_ORDER = [_QWEN, _GEMMA, _LLAMA, _OLMO, _PHI]

# Representative family color (the mid/large shade) for family-level legends.
FAMILY_COLORS = {
    _QWEN: "#2A4F9C",
    _GEMMA: "#E07B39",
    _LLAMA: "#548235",
    _OLMO: "#8E44AD",
    _PHI: "#922B21",
}


def standard_models() -> list[Model]:
    """The 17 standard-figure models, in family/size order. Excludes qwen3."""
    return list(STANDARD_MODELS)


def by_family() -> dict[str, list[Model]]:
    out: dict[str, list[Model]] = {fam: [] for fam in FAMILY_ORDER}
    for m in STANDARD_MODELS:
        out[m.family].append(m)
    return out


def model_by_dir(dir_name: str) -> Model:
    for m in STANDARD_MODELS:
        if m.dir == dir_name:
            return m
    raise KeyError(f"unknown model dir: {dir_name!r}")


def family_color(family: str) -> str:
    return FAMILY_COLORS[family]
