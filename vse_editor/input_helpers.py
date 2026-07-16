"""Keyboard modifier helpers (moved verbatim from vse.py — step-25).

Module-level so extracted controller modules can use them without importing
vse (which would be an import cycle).
"""

import pygame


def is_shift_pressed(keys=None):
    """Return True if either Shift key is currently pressed."""
    keys = pygame.key.get_pressed() if keys is None else keys
    return keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]


def is_ctrl_pressed(keys=None):
    """Return True if either Ctrl key is currently pressed."""
    keys = pygame.key.get_pressed() if keys is None else keys
    return keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]
