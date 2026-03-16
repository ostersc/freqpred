"""Hard cap enforcement and circuit breakers.

IMPORTANT: Strategy code calls risk.py; risk.py has final say.
Strategy position_size() output is ALWAYS passed through here before any order.
"""
from __future__ import annotations

# TODO: implement hard caps per SPEC.md section 10
