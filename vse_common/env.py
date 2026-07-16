"""Robust environment-variable parsing (Phase 8, step-55).

A garbage value in a VSE_*/CARLA_* tuning variable must never crash startup
or playback: every helper falls back to the caller's default and prints one
warning. Missing and empty values silently mean "use the default" (several
legacy sites treated empty as default via ``or``; this keeps that contract).

Layering: vse_common is stdlib-only -- keep it that way.
"""

import os


def env_float(name, default):
    """os.environ[name] as float; missing/empty/garbage -> default."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        print(f"Warning: ignoring {name}={raw!r} (not a number); using {default}")
        return float(default)


def env_int(name, default):
    """os.environ[name] as int; missing/empty/garbage -> default."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        print(f"Warning: ignoring {name}={raw!r} (not an integer); using {default}")
        return int(default)


def env_bool(name, default=False):
    """Truth of int(os.environ[name]); missing/empty/garbage -> default."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    try:
        return bool(int(raw))
    except ValueError:
        print(f"Warning: ignoring {name}={raw!r} (expected 0/1); using {default}")
        return bool(default)
