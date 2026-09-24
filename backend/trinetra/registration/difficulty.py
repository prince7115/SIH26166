"""Heuristic pair difficulty from lighting, scale and texture. Reported only; never used to decide."""
from ..products.sun import azimuth_gap


def assess_difficulty(ohrc_sun, ref_sun, scale_ratio, tex_ohrc, tex_ref, ref_name="reference"):
    score, factors = 0, []

    def add(points, text):
        nonlocal score
        score += points
        factors.append(f"{text} (+{points})" if points else text)

    eo = (ohrc_sun or {}).get("elev_deg")
    er = (ref_sun or {}).get("elev_deg")
    elev_gap = None
    if eo is not None and er is not None:
        elev_gap = abs(eo - er)
        if elev_gap >= 15:
            add(3, f"sun elevation gap {elev_gap:.1f}°: very different shadow lengths")
        elif elev_gap >= 7:
            add(1, f"sun elevation gap {elev_gap:.1f}°")
        else:
            add(0, f"sun elevation gap {elev_gap:.1f}°: similar")
        low = min(eo, er)
        if low < 15:
            add(1, f"low sun ({low:.1f}°): long shadows")
    else:
        missing = [n for n, v in (("OHRC", eo), (ref_name, er)) if v is None]
        add(0, f"sun elevation unknown for {', '.join(missing)}: not scored")

    ao, ar = (ohrc_sun or {}).get("azim_deg"), (ref_sun or {}).get("azim_deg")
    co, cr = (ohrc_sun or {}).get("azim_convention"), (ref_sun or {}).get("azim_convention")
    azim_gap = None
    if ao is not None and ar is not None and co == cr == "north_cw":
        azim_gap = azimuth_gap(ao, ar)
        if azim_gap >= 30:
            add(2, f"sun azimuth gap {azim_gap:.1f}°: shadows point different ways")
        elif azim_gap >= 10:
            add(1, f"sun azimuth gap {azim_gap:.1f}°")
        else:
            add(0, f"sun azimuth gap {azim_gap:.1f}°: shadows point the same way")
    else:
        add(0, "sun azimuth not compared (missing, or conventions differ / unknown)")

    if scale_ratio >= 25:
        add(2, f"scale ratio {scale_ratio:.1f}×")
    elif scale_ratio >= 10:
        add(1, f"scale ratio {scale_ratio:.1f}×")

    tex = min(tex_ohrc, tex_ref)
    if tex < 30:
        add(2, f"very low texture ({tex:.0f})")
    elif tex < 80:
        add(1, f"low texture ({tex:.0f})")

    level = "easy" if score <= 1 else "medium" if score <= 3 else "hard"
    return {"level": level, "score": score, "factors": factors,
            "elev_gap_deg": elev_gap, "azim_gap_deg": azim_gap,
            "texture_ohrc": round(float(tex_ohrc), 1), "texture_ref": round(float(tex_ref), 1),
            "scale_ratio": round(float(scale_ratio), 3)}
