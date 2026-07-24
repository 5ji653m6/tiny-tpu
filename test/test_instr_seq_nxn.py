"""Gate test for the agent-authored instruction sequencer leaf
(src/instr_seq_nxn.sv, roadmap item 9b). Written by the harness author,
not the agent — per tinytpu-loop README "the gate grows with the
design".

Spec under test (from the item-9b task spec): the sequencer stores a
program of 133+17*(N-2)-bit instruction words. The host loads it one
word per cycle (prog_wr_en + prog_wr_data, write pointer
auto-increments from 0 after rst), then pulses `run` for one cycle.
Starting the cycle AFTER the run pulse is sampled, instr_out presents
prog[0], prog[1], ..., prog[M-1] (M = words loaded at the run pulse),
one per cycle, with busy=1; the cycle after the last word, instr_out
returns to 0 and busy drops. prog_wr_en while busy is ignored. A
second run pulse replays the same program. Run with M=0 is ignored.

Assertions are LIVE (PYTHONOPTIMIZE is empty). The Makefile target
compiles at N=4 (-Pinstr_seq_nxn.SYSTOLIC_ARRAY_WIDTH=4) with
INSTR_SEQ_N=4 exported.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

N = int(os.environ.get("INSTR_SEQ_N", "4"))
WORD_W = 133 + 17 * (N - 2)
MASK = (1 << WORD_W) - 1


def pattern(i):
    """Distinctive nonzero word per index (fits in WORD_W bits)."""
    return (0x5A5A0000 + i * 0x10101 + (i << 100)) & MASK


async def tick(dut, cycles=1):
    """Edge + 1ns settle, so reads see post-edge (NBA-updated) values
    and drives land mid-cycle, stable well before the next edge."""
    for _ in range(cycles):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")


async def load_program(dut, words):
    for w in words:
        dut.prog_wr_data.value = w
        dut.prog_wr_en.value = 1
        await tick(dut)
    dut.prog_wr_en.value = 0
    dut.prog_wr_data.value = 0


async def check_replay(dut, words):
    """Pulse run; instr_out must present words then zeros, busy exactly
    len(words) cycles starting the cycle after the pulse."""
    dut.run.value = 1
    await tick(dut)
    dut.run.value = 0
    assert dut.busy.value.integer == 1, "busy not set the cycle after run"
    for i, w in enumerate(words):
        got = dut.instr_out.value.integer & MASK
        assert got == w, (
            f"replay word {i}: got {got:#x}, expected {w:#x}")
        assert dut.busy.value.integer == 1, (
            f"busy dropped early at word {i}")
        await tick(dut)
    assert dut.busy.value.integer == 0, "busy still set after program end"
    assert (dut.instr_out.value.integer & MASK) == 0, (
        "instr_out not zero after program end")


@cocotb.test()
async def test_instr_seq_nxn_load_run(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    dut.prog_wr_en.value = 0
    dut.prog_wr_data.value = 0
    dut.run.value = 0
    await tick(dut, 2)
    dut.rst.value = 0
    await tick(dut)

    # Idle before any program: instr_out zero, busy low.
    assert (dut.instr_out.value.integer & MASK) == 0
    assert dut.busy.value.integer == 0

    # Run with an empty program is ignored.
    dut.run.value = 1
    await tick(dut)
    dut.run.value = 0
    assert dut.busy.value.integer == 0, "busy set with empty program"
    assert (dut.instr_out.value.integer & MASK) == 0
    await tick(dut, 2)

    # Load a 37-word program (odd, non-power-of-2) and replay it.
    words = [pattern(i) for i in range(37)]
    await load_program(dut, words)
    await tick(dut, 3)  # settle gap between load and run
    await check_replay(dut, words)

    # A second run pulse replays the SAME program (no reload).
    await tick(dut, 5)
    await check_replay(dut, words)

    print(f"instr_seq_nxn load/run/replay OK (N={N}, {len(words)} words)")


@cocotb.test()
async def test_instr_seq_nxn_write_ignored_while_busy(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    dut.prog_wr_en.value = 0
    dut.prog_wr_data.value = 0
    dut.run.value = 0
    await tick(dut, 2)
    dut.rst.value = 0
    await tick(dut)

    words = [pattern(i) for i in range(8)]
    await load_program(dut, words)

    # Start the replay, then attempt a hostile write mid-run: it must
    # be ignored (the replay continues with the ORIGINAL program).
    dut.run.value = 1
    await tick(dut)
    dut.run.value = 0
    got0 = dut.instr_out.value.integer & MASK
    assert got0 == words[0]
    dut.prog_wr_en.value = 1
    dut.prog_wr_data.value = 0xDEAD
    await tick(dut)
    dut.prog_wr_en.value = 0
    dut.prog_wr_data.value = 0
    got1 = dut.instr_out.value.integer & MASK
    assert got1 == words[1], (
        f"mid-run write corrupted replay: word 1 = {got1:#x}, "
        f"expected {words[1]:#x}")
    # Drain the rest of the program.
    for i in range(2, len(words)):
        await tick(dut)
        got = dut.instr_out.value.integer & MASK
        assert got == words[i], (
            f"mid-run write corrupted replay: word {i} = {got:#x}, "
            f"expected {words[i]:#x}")
    await tick(dut)
    assert dut.busy.value.integer == 0

    print("instr_seq_nxn ignores prog_wr_en while busy OK")
