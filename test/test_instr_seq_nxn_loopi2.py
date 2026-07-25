"""Gate test for the agent-authored LOOPI sequencer extensions
(src/instr_seq_nxn.sv, roadmap item 17a): the **wbase** field and
**ptr-7 residual reads** striding. Written by the harness author, not
the agent — per tinytpu-loop README "the gate grows with the design".
Red-first: FAILS until the RTL lands (BASELINE_EXCLUDE'd until then).

Spec under test (from the item-17a task spec): backward-compatible
extension of the item-15 LOOPI word. One new 16-bit field rides the
(previously unused) upper control-word bits:

  bits [71:56] wbase (16-bit unsigned, ZERO = exact item-15 behavior)

With the indexed flag SET, on loop iteration i (0-based), the body-word
address transform becomes:

  ptr == 0 (activation read):  addr += i*stride_a   (unchanged)
  ptr == 1 (weight read):      addr += i*stride_w  ONLY when addr >=
                               wbase; below wbase the word passes
                               bit-identical (STATIONARY host weights)
  ptr == 7 (residual read):    addr += i*stride_a  (same stride as
                               ptr 0 — residuals are activations)
  every other word:            bit-identical       (unchanged)

The boundary is INCLUSIVE: addr == wbase strides.

Motivation: the DiT capstone (item 17b) loops one denoiser body over T
timesteps. Attention phases inside the body read BOTH stationary host
weight matrices AND per-iteration intermediates (K~, P^T) as ptr-1
reads — one global stride_w can't serve both. wbase splits the address
space: the host image lives below wbase (verbatim), chip-produced
regions at/above it (advancing). wbase=0 keeps items 15/16 programs
advancing every ptr-1 read, bit-identical.

Assertions are LIVE (PYTHONOPTIMIZE is empty). The Makefile target
compiles at N=4 (-Pinstr_seq_nxn.SYSTOLIC_ARRAY_WIDTH=4) with
INSTR_SEQ_N=4 exported.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from test_instr_seq_nxn_loopi import (
    WORD_W, MASK, loopi_word, read_word, pattern, tick, fresh_load,
    check_emission,
)

N = int(os.environ.get("INSTR_SEQ_N", "4"))


def loopi2_word(count, length, stride_a=0, stride_w=0, wbase=0,
                indexed=True):
    """LOOPI word with the item-17a wbase field (bits [71:56])."""
    return loopi_word(count, length, stride_a, stride_w, indexed) \
        | (wbase << 56)


@cocotb.test()
async def test_instr_seq_nxn_loopi2_wbase(dut):
    """wbase freeze + boundary + ptr-7: one 5-word body, count=3,
    sa=8/sw=16, wbase=64.
      ptr-0 read @100        -> 100, 108, 116   (always advances)
      ptr-1 read @32 (< 64)  -> 32, 32, 32      (FROZEN: host weights)
      ptr-1 read @64 (== 64) -> 64, 80, 96      (boundary inclusive)
      ptr-1 read @96 (> 64)  -> 96, 112, 128    (advancing intermediates)
      ptr-7 read @128        -> 128, 136, 144   (residual: stride_a)"""
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
    body = [read_word(0, 100), read_word(1, 32), read_word(1, 64),
            read_word(1, 96), read_word(7, 128)]
    prog = [p0, loopi2_word(3, 5, stride_a=8, stride_w=16, wbase=64),
            *body, p1]
    expected = [p0, 0,
                read_word(0, 100), read_word(1, 32), read_word(1, 64),
                read_word(1, 96), read_word(7, 128),
                read_word(0, 108), read_word(1, 32), read_word(1, 80),
                read_word(1, 112), read_word(7, 136),
                read_word(0, 116), read_word(1, 32), read_word(1, 96),
                read_word(1, 128), read_word(7, 144),
                p1]

    await fresh_load(dut, prog)
    await check_emission(dut, expected)

    print(f"instr_seq_nxn LOOPI2 wbase OK (N={N}, count=3 len=5 "
          f"sa=8 sw=16 wbase=64, {len(expected)} emissions)")


@cocotb.test()
async def test_instr_seq_nxn_loopi2_compat(dut):
    """Backward compatibility, two clauses:
      1. wbase=0 => every ptr-1 read advances (exact item-15 behavior).
      2. indexed flag CLEAR with sa/sw/wbase bits all nonzero => exact
         item-12 LOOP (all extension fields ignored, body verbatim)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    dut.prog_wr_en.value = 0
    dut.prog_wr_data.value = 0
    dut.run.value = 0
    await tick(dut, 2)
    dut.rst.value = 0
    await tick(dut)

    p0, b0, b1, p1 = (pattern(5), read_word(1, 10), read_word(0, 32),
                      pattern(6))

    cases = [
        # wbase=0: ptr-1 read @10 advances by sw=5 — item-15 verbatim
        ([p0, loopi2_word(2, 1, stride_a=0, stride_w=5, wbase=0),
          b0, p1],
         [p0, 0, read_word(1, 10), read_word(1, 15), p1],
         "wbase=0 => advance-all (item-15 behavior)"),
        # flag clear + all extension bits set: exact item-12 verbatim
        ([p0, loopi2_word(2, 2, stride_a=9, stride_w=11, wbase=48,
                          indexed=False),
          b0, b1, p1],
         [p0, 0, b0, b1, b0, b1, p1],
         "flag clear => wbase/strides ignored (item-12 behavior)"),
    ]
    for prog, expected, name in cases:
        await fresh_load(dut, prog)
        await check_emission(dut, expected)
        print(f"  {name}: OK ({len(expected)} emissions)")

    print("instr_seq_nxn LOOPI2 backward compat OK")
