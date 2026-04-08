# Safety net: multiprocessing.resource_tracker spawns a subprocess whose CWD
# may be the freqpred package directory. Python's empty-string sys.path entry
# (CWD) then makes this package importable as bare 'signal', shadowing the
# stdlib. If that happens, re-export everything from the C-extension _signal
# so callers (e.g. resource_tracker) find what they expect.
if __name__ == "signal":
    from _signal import *  # noqa: F401, F403
