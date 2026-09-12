"""Step 1: pinned-model load, exact tokenizer preflight, solo-only screen, throughput.

Step 1 generates solo model responses only. It never generates a joint, exact-context,
filler, fake-batch, serial, or repeated-sampling response. The frozen Step 0 freeze is the
only authority for every design field; nothing here edits it.
"""

__all__ = ["STEP1_VERSION"]

STEP1_VERSION = "1.0.0"
