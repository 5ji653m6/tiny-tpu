"""Gate test for the agent-authored LOOPI ptr-8 SCALE-read stride
(src/instr_seq_nxn.sv, roadmap item 18a sequencer half). Written by the
harness author, not the agent — per tinytpu-loop README "the gate grows
with the design". Red-first: FAILS until the RTL lands
(BASELINE_EXCLUDE'd until then).

Spec under test (from the item-18a task spec): a UB read command with
ptr = 8 is a SCALE read (the per-element multiply operand for the new
VPU scale stage — the adaLN per-element scale half of timestep
conditioning; the shift half is the item-17a residual add). Scale
matrices are activation-like per-iteration data (they live in the
per-iteration append regions in the item-18b capstone), so under an
indexed loop a ptr-8 body read advances by i*stride_a — exactly like
ptr-0 (activations) and ptr-7 (residuals), NOT by stride_w.

Pre-item-18a programs contain no ptr-8 reads and replay
bit-identically; every other word passes through untouched.

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

PTR_SCALE = 8


@cocotb.test()
async def test_instr_seq_nxn_scale_stride(dut):
    """LOOPI(2, 7, sa=16, sw=32) body carrying one read of each
    activation-side ptr (0 act, 7 residual, 8 scale) and one ptr-1
    weight read: on pass 2 the ptr-8 read must advance by stride_a
    (48 -> 64), alongside ptr-0 (64 -> 80) and ptr-7 (128 -> 144),
    while ptr-1 advances by stride_w (96 -> 128)."""
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
    body = [read_word(PTR_SCALE, 48), 0, read_word(0, 64), 0,
            read_word(1, 96), 0, read_word(7, 128)]
    prog = [p0, loopi_word(2, 7, 16, 32, indexed=True), *body, p1]
    expected = [p0, 0,
                read_word(PTR_SCALE, 48), 0, read_word(0, 64), 0,
                read_word(1, 96), 0, read_word(7, 128),
                read_word(PTR_SCALE, 64), 0, read_word(0, 80), 0,
                read_word(1, 128), 0, read_word(7, 144),
                p1]

    await fresh_load(dut, prog)
    await check_emission(dut, expected)

    print(f"instr_seq_nxn ptr-8 SCALE stride OK (N={N}: ptr-8 body "
          f"read advances by stride_a like ptr-0/ptr-7, ptr-1 by "
          f"stride_w; {len(expected)} emissions)")


@cocotb.test()
async def test_instr_seq_nxn_scale_flag_clear(dut):
    """With the indexed flag CLEAR a ptr-8 read replays verbatim on
    every pass (exact item-12 behavior — the stride fields are
    ignored)."""
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
    body = [read_word(PTR_SCALE, 48), 0, read_word(1, 96)]
    prog = [p0, loopi_word(2, 3, 16, 32, indexed=False), *body, p1]
    expected = [p0, 0] + body + body + [p1]

    await fresh_load(dut, prog)
    await check_emission(dut, expected)

    print("instr_seq_nxn ptr-8 flag-clear verbatim OK")
