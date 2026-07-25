"""Gate test for the agent-authored LOOPI length extension
(src/instr_seq_nxn.sv, roadmap item 17b1). Written by the harness
author, not the agent — per tinytpu-loop README "the gate grows with
the design". Red-first: FAILS until the RTL lands (BASELINE_EXCLUDE'd
until then).

Spec under test (from the item-17b1 task spec): the item-12 LOOP word's
8-bit len field (bits [7:0], max 255) cannot hold a DiT denoiser
iteration — the item-17b capstone body is 16 phases / 838 words. The
extension rides the (previously unused) control-word bits [23:17]
(between the indexed flag bit 16 and stride_a at [39:24]):

  len = { bits [23:17], bits [7:0] }   (15-bit, words; ZERO body is
                                        out of contract as before)

Backward compatible by construction: every pre-item-17b1 program has
len <= 255, i.e. bits [23:17] = 0, and replays bit-identically. The
extension is ORTHOGONAL to the indexed flag: large bodies work with
the flag clear (verbatim replay) and set (stride transforms apply
across the whole body, including reads past word offset 255). The
item-15 op decode needs no change: a large-len loop word has count
(bits [15:8]) nonzero, so `~word[0] | (count != 0)` keeps decoding it
as a loop even when the low len byte is odd (len=257 case below).

All other LOOP/LOOPI semantics are unchanged: one all-zero bubble on
the control word, body runs count times back-to-back, count=0 skips,
loop state resets on every run pulse.

Assertions are LIVE (PYTHONOPTIMIZE is empty). The Makefile target
compiles at N=4 (-Pinstr_seq_nxn.SYSTOLIC_ARRAY_WIDTH=4) with
INSTR_SEQ_N=4 exported.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from test_instr_seq_nxn_loopi import (
    WORD_W, MASK, CTRL, loopi_word, read_word, pattern, tick,
    fresh_load, check_emission,
)

N = int(os.environ.get("INSTR_SEQ_N", "4"))


def loopxl_word(count, length, stride_a=0, stride_w=0, wbase=0,
                indexed=True):
    """LOOP control word with the 15-bit length: low 8 bits at [7:0]
    (as always), high 7 bits at [23:17]."""
    assert 0 < length < (1 << 15)
    w = loopi_word(count, length & 0xFF, stride_a, stride_w, indexed)
    return w | ((length >> 8) << 17)


@cocotb.test()
async def test_instr_seq_nxn_loopxl_large(dut):
    """len=300 indexed body, count=2, sa=4/sw=8: a ptr-0 read at body
    offset 0 and a ptr-1 read at body offset 299 (past the old 255
    ceiling) — the second pass must stride BOTH, proving the loop
    traverses the full 300-word body and the transform reaches reads
    beyond offset 255. len=300 = 0b1_0101100: len_hi = 1, len_lo = 44
    (word[0] = 0)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    dut.prog_wr_en.value = 0
    dut.prog_wr_data.value = 0
    dut.run.value = 0
    await tick(dut, 2)
    dut.rst.value = 0
    await tick(dut)

    p0, p1 = pattern(1), pattern(2)
    idle = 0
    body = [read_word(0, 40)] + [idle] * 298 + [read_word(1, 96)]
    prog = [p0, loopxl_word(2, 300, stride_a=4, stride_w=8), *body, p1]
    expected = [p0, 0,
                read_word(0, 40)] + [idle] * 298 + [read_word(1, 96),
                read_word(0, 44)] + [idle] * 298 + [read_word(1, 104),
                p1]

    await fresh_load(dut, prog)
    await check_emission(dut, expected)

    print(f"instr_seq_nxn LOOXL len=300 OK (N={N}, count=2, "
          f"{len(expected)} emissions, reads at body offsets 0/299)")


@cocotb.test()
async def test_instr_seq_nxn_loopxl_edges(dut):
    """Boundary and compatibility cases:
      1. len=256 (len_hi=1, len_lo=0 — the first length the 8-bit field
         could not express; word[0]=0), flag CLEAR: verbatim replay of
         the full body, both passes bit-identical.
      2. len=257 (len_hi=1, len_lo=1 — odd low byte: word[0]=1 with a
         nonzero count, the item-15 decode clause must keep it a loop),
         indexed, read at body offset 256 strides."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    dut.prog_wr_en.value = 0
    dut.prog_wr_data.value = 0
    dut.run.value = 0
    await tick(dut, 2)
    dut.rst.value = 0
    await tick(dut)

    p0, p1 = pattern(5), pattern(6)

    # Case 1: len=256, flag clear, verbatim
    body1 = [read_word(0, 48)] + [0] * 254 + [read_word(1, 80)]
    prog1 = [p0, loopxl_word(2, 256, stride_a=9, stride_w=11,
                             indexed=False), *body1, p1]
    exp1 = [p0, 0] + body1 + body1 + [p1]

    # Case 2: len=257, indexed, read strides past offset 255
    body2 = [0] * 256 + [read_word(0, 64)]
    prog2 = [p0, loopxl_word(2, 257, stride_a=4, stride_w=0), *body2, p1]
    exp2 = [p0, 0] + [0] * 256 + [read_word(0, 64)] \
        + [0] * 256 + [read_word(0, 68)] + [p1]

    for prog, expected, name in (
            (prog1, exp1, "len=256 flag-clear verbatim"),
            (prog2, exp2, "len=257 odd-low-byte indexed stride")):
        await fresh_load(dut, prog)
        await check_emission(dut, expected)
        print(f"  {name}: OK ({len(expected)} emissions)")

    print("instr_seq_nxn LOOXL boundary + compat OK")
