"""Run parameters from the /api/run query string."""


def run_options(args):
    def flag(name):
        return args.get(name, "false").lower() == "true"
    return {"ecc_mode": args.get("ecc_mode") or "gated", "matcher": args.get("matcher", "superglue"),
            "shadow_mask": flag("shadow_mask"), "tile_rescale": flag("tile_rescale"),
            "tile_polarity": flag("tile_polarity"), "crater_matching": flag("crater_matching"),
            "auto_retry": flag("auto_retry")}


def sun_override(args):
    """Reference sun values typed on the website; only the fields that were filled in."""
    def number(name):
        try:
            return float(args[name]) if args.get(name, "") != "" else None
        except ValueError:
            return None
    sun = {"elev_deg": number("ref_sun_elev"), "azim_deg": number("ref_sun_azim"),
           "azim_convention": args.get("ref_sun_conv") or None}
    return {k: v for k, v in sun.items() if v is not None} or None
