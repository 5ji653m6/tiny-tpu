"""Gate test for the agent-authored sequencer LOOP construct
(src/instr_seq_nxn.sv, roadmap item 12). Written by the harness author,
not the agent — per tinytpu-loop README "the gate grows with the
design". Red-first: this test FAILS until the RTL lands
(BASELINE_EXCLUDE'd from the loop's baseline sanity until then).

Spec under test (from the item-12 task spec): the program word is
widened by ONE bit to {ctrl, legacy_word} — ctrl the MSB, instr_out
STAYS the legacy instruction width (the consumer contract is
unchanged). Item 13 then widened the instruction word itself by one
(SiLU pathway bit appended at the top): WORD_W is now 134+17*(N-2),
so prog_wr_data is 135+17*(N-2) bits and the ctrl escape bit rides
one position higher.

  ctrl = 0: plain instruction word, replayed exactly as in item 9b
            (the ctrl bit is stripped — all pre-item-12 programs
            replay bit-identically).
  ctrl = 1: control word. bits [1:0] = op.
            op 2'b00 = LOOP: count = bits [15:8], len = bits [7:0].
            The `len` words immediately following the LOOP word are the
            body; the body executes `count` times IN TOTAL; count = 0
            skips the body entirely. The LOOP word itself emits exactly
            one all-zero cycle (one bubble per loop, not per
            iteration). After the final pass, replay continues with the
            word after the body.
            Other ops: reserved — emit one all-zero cycle, no other
            effect.
  Loop state resets on every run pulse: a re-run replays the loop
  identically. Out of contract (not gated): control word at index 0,
  control words inside a body, nesting, body overrunning the program.

Assertions are LIVE (PYTHONOPTIMIZE is empty). The Makefile target
compiles at N=4 (-Pinstr_seq_nxn.SYSTOLIC_ARRAY_WIDTH=4) with
INSTR_SEQ_N=4 exported.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

N = int(os.environ.get("INSTR_SEQ_N", "4"))
WORD_W = 134 + 17 * (N - 2)   # instruction width (item 13: SiLU bit)
PROG_W = WORD_W + 1           # program word = {ctrl, legacy_word}
CTRL = 1 << WORD_W            # control-word escape bit (prog-word MSB)
MASK = (1 << WORD_W) - 1      # instr_out mask (legacy width)

OP_LOOP = 0b00


def loop_word(count, length):
    """LOOP control word: execute the next `length` words `count`
    times in total (count = 0 skips the body)."""
    return CTRL | OP_LOOP | (count << 8) | length


def reserved_ctrl(op):
    """A reserved (non-LOOP) control word: one no-op bubble cycle."""
    assert op != OP_LOOP
    return CTRL | op


def pattern(i):
    """Distinctive nonzero word per index. i in 4..7 sets the
    TOP instruction bit (bit WORD_W-1) — proves plain words with that bit
    set are not mistaken for control words and keep it on instr_out."""
    return (0x5A5A0000 + i * 0x10101 + (i << (WORD_W - 3))) & MASK


async def tick(dut, cycles=1):
    """Edge + 1ns settle: reads see post-edge values, drives land
    mid-cycle."""
    for _ in range(cycles):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")


async def fresh_load(dut, words):
    """Reset (clears the write pointer on rst ENTRY) then load a
    program one word per cycle — programs may be wider than instr_out:
    the ctrl bit rides in prog_wr_data's MSB."""
    dut.rst.value = 1
    await tick(dut, 2)
    for w in words:
        dut.prog_wr_data.value = w
        dut.prog_wr_en.value = 1
        await tick(dut)
    dut.prog_wr_en.value = 0
    dut.prog_wr_data.value = 0
    dut.rst.value = 0
    await tick(dut, 3)


async def check_emission(dut, expected):
    """Pulse run; instr_out must present EXACTLY the expected word
    stream (bubbles included), one per cycle, busy high throughout,
    then 0 and busy low."""
    dut.run.value = 1
    await tick(dut)
    dut.run.value = 0
    assert dut.busy.value.integer == 1, "busy not set the cycle after run"
    # SRAM integration: the cycle after the run pulse is an all-zero
    # bubble (SRAM read latency); the stream starts one cycle later.
    await tick(dut)  # skip SRAM prefetch bubble
    for i, w in enumerate(expected):
        got = dut.instr_out.value.integer & MASK
        assert got == w, (
            f"emission {i}: got {got:#x}, expected {w:#x}")
        assert dut.busy.value.integer == 1, (
            f"busy dropped early at emission {i}")
        await tick(dut)
    assert dut.busy.value.integer == 0, (
        f"busy still set after {len(expected)} emissions")
    assert (dut.instr_out.value.integer & MASK) == 0, (
        "instr_out not zero after program end")


@cocotb.test()
async def test_instr_seq_nxn_loop_replay(dut):
    """LOOP count=3: one bubble on the LOOP word, body x3 back-to-back,
    then the trailing word — one exact emission stream."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    dut.prog_wr_en.value = 0
    dut.prog_wr_data.value = 0
    dut.run.value = 0
    await tick(dut, 2)
    dut.rst.value = 0
    await tick(dut)

    p0, b0, b1, p1 = pattern(1), pattern(2), pattern(3), pattern(4)
    prog = [p0, loop_word(3, 2), b0, b1, p1]
    expected = [p0, 0,
                b0, b1, b0, b1, b0, b1,
                p1]

    await fresh_load(dut, prog)
    await check_emission(dut, expected)

    # A second run pulse replays the loop identically (loop state
    # resets at every run start).
    await tick(dut, 5)
    await check_emission(dut, expected)

    print(f"instr_seq_nxn LOOP replay OK (N={N}, count=3 len=2, "
          f"{len(expected)} emissions, replayed twice)")


@cocotb.test()
async def test_instr_seq_nxn_loop_edge_counts(dut):
    """count=1 executes the body once; count=0 skips it entirely; a
    reserved op emits a single no-op bubble. All three programs cost
    exactly one bubble at the control word."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    dut.prog_wr_en.value = 0
    dut.prog_wr_data.value = 0
    dut.run.value = 0
    await tick(dut, 2)
    dut.rst.value = 0
    await tick(dut)

    p0, b0, b1, p1 = pattern(5), pattern(6), pattern(7), pattern(8)

    cases = [
        ([p0, loop_word(1, 2), b0, b1, p1],
         [p0, 0, b0, b1, p1], "count=1"),
        ([p0, loop_word(0, 2), b0, b1, p1],
         [p0, 0, p1], "count=0 (body skipped)"),
        ([p0, reserved_ctrl(0b11), p1],
         [p0, 0, p1], "reserved op no-op"),
    ]
    for prog, expected, name in cases:
        await fresh_load(dut, prog)
        await check_emission(dut, expected)
        print(f"  {name}: OK ({len(expected)} emissions)")

    print("instr_seq_nxn LOOP edge counts + reserved op OK")
