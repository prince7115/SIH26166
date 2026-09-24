"""Run options: defaults, validation and the auto-retry ladder."""

DEFAULT_OPTIONS = {
    "ecc_mode": "gated",        # gated | legacy | off
    "matcher": "superglue",     # superglue | lightglue (coarse stage, steps 8 and 11)
    "shadow_mask": False,       # rescue: suppress shadows in failed tiles
    "tile_rescale": False,      # rescue: retry failed tiles at 1.5x / 2x coarser scale
    "tile_polarity": False,     # rescue: intensity ZNCC, normal and inverted polarity
    "crater_matching": False,   # crater-anchored matches pooled before step 13
    "auto_retry": False,        # rerun steps 12-14 with rescues while the result is not trusted
}
ECC_MODES = ("gated", "legacy", "off")
MATCHERS = ("superglue", "lightglue")
RESCUE_KEYS = ("shadow_mask", "tile_rescale", "tile_polarity")

# Each retry switches on more rescues than the one before.
RETRY_LADDER = [
    ("shadow suppression + multi-scale", {"shadow_mask": True, "tile_rescale": True}),
    ("+ intensity polarity + crater matches", {"tile_polarity": True, "crater_matching": True}),
]


def resolve_options(profile_options=None, run_options=None):
    """DEFAULT_OPTIONS, overridden by the pipeline profile, then by this run.
    Raises ValueError on an unknown option or value."""
    opts = dict(DEFAULT_OPTIONS)
    for src in (profile_options or {}, run_options or {}):
        for k, v in src.items():
            if k not in DEFAULT_OPTIONS:
                raise ValueError(f"Unknown option '{k}'")
            opts[k] = v
    if opts["ecc_mode"] not in ECC_MODES:
        raise ValueError(f"ecc_mode must be one of {ECC_MODES}")
    if opts["matcher"] not in MATCHERS:
        raise ValueError(f"matcher must be one of {MATCHERS}")
    for k, default in DEFAULT_OPTIONS.items():
        if isinstance(default, bool):
            opts[k] = bool(opts[k])
    return opts


def any_rescue(opts):
    return any(opts.get(k) for k in RESCUE_KEYS)


def retry_ladder(opts):
    """[(label, options)] for each attempt: the requested options, then (with auto_retry) one
    rung per RETRY_LADDER step that actually changes something."""
    ladder = [("requested options", opts)]
    if opts["auto_retry"]:
        for label, extra in RETRY_LADDER:
            nxt = {**ladder[-1][1], **extra}
            if nxt != ladder[-1][1]:
                ladder.append((label, nxt))
    return ladder
