"""Gate test for the agent-authored sequencer INDEXED LOOP construct
(src/instr_seq_nxn.sv, roadmap item 15). Written by the harness author,
not the agent — per tinytpu-loop README "the gate grows with the
design". Red-first: this test FAILS until the RTL lands
(BASELINE_EXCLUDE'd from the loop's baseline sanity until then).

Spec under test (from the item-15 task spec): LOOPI is a
BACKWARD-COMPLETE extension of the item-12 LOOP control word — same
{ctrl, legacy_word} program word, same op decode (bits [1:0], op
2'b00 = LOOP, decode on bit 0 per the item-12 len-aliasing contract),
same count = bits [15:8] / len = bits [7:0], same one-bubble-per-loop,
same per-run reset. One new flag and two new fields ride in the
(previously unused) upper bits of the control word:

  bit  [16]    indexed flag. 0 => exact item-12 LOOP behavior (the
               stride fields are IGNORED — every pre-item-15 program
               replays bit-identically).
  bits [39:24] stride_a (16-bit unsigned)
  bits [55:40] stride_w (16-bit unsigned)

With the indexed flag SET, on loop iteration i (0-based), each body
word that is a UB read command (bit 1 set) AND whose ptr field (bits
[61:53]) is exactly 0 gets its address field (bits [52:37]) emitted as
addr + i*stride_a; ptr exactly 1 gets addr + i*stride_w. Every other
word — non-read words, reads with any other ptr value — passes through
bit-identical. The address add is 16-bit unsigned on the addr field;
overflow out of contract. count = 0 still skips the body entirely.

Motivation: item 12's LOOP replays the body verbatim, so every
iteration reads the SAME regions. Looped multi-head attention (each
head's weights/activations at a fixed offset), tiled matmul, and the
diffusion sampler (iteration t reads what t-1 wrote) all need the
body's addresses to ADVANCE per iteration — DSP-style loop addressing.

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
IDX_FLAG = 1 << 16


def loopi_word(count, length, stride_a=0, stride_w=0, indexed=True):
    """LOOP control word, optionally indexed: with indexed=True, read
    addresses in the body advance by stride_a (ptr 0) / stride_w
    (ptr 1) per iteration."""
    w = CTRL | OP_LOOP | (count << 8) | length
    if indexed:
        w |= IDX_FLAG | (stride_a << 24) | (stride_w << 40)
    return w


def read_word(ptr, addr):
    """A UB read command with distinctive rows/cols/transpose fields —
    the stride must touch ONLY the address field (bits [52:37])."""
    return (1 << 1) | (1 << 2) | (3 << 5) | (2 << 21) | (addr << 37) \
        | (ptr << 53)


def pattern(i):
    """Distinctive nonzero NON-READ word per index (bit 1 clear). i in
    4..7 sets the TOP instruction bit — proves plain words with that
    bit set are not mistaken for control words."""
    return (0x5A5A0000 + i * 0x10101 + (i << (WORD_W - 3))) & MASK & ~0b10


async def tick(dut, cycles=1):
    """Edge + 1ns settle: reads see post-edge values, drives land
    mid-cycle."""
    for _ in range(cycles):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")


async def fresh_load(dut, words):
    """Reset (clears the write pointer on rst ENTRY) then load a
    program one word per cycle."""
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
async def test_instr_seq_nxn_loopi_replay(dut):
    """LOOPI count=3: one bubble on the control word, then the body
    three times with the ptr-0 read advancing by stride_a and the
    ptr-1 read by stride_w each iteration. Replayed twice (loop state
    — strides included — resets on every run pulse)."""
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
    b0, b1 = read_word(ptr=0, addr=100), read_word(ptr=1, addr=200)
    prog = [p0, loopi_word(3, 2, stride_a=5, stride_w=7), b0, b1, p1]
    expected = [p0, 0,
                read_word(0, 100), read_word(1, 200),
                read_word(0, 105), read_word(1, 207),
                read_word(0, 110), read_word(1, 214),
                p1]

    await fresh_load(dut, prog)
    await check_emission(dut, expected)

    await tick(dut, 5)
    await check_emission(dut, expected)

    print(f"instr_seq_nxn LOOPI replay OK (N={N}, count=3 len=2 "
          f"sa=5 sw=7, {len(expected)} emissions, replayed twice)")


@cocotb.test()
async def test_instr_seq_nxn_loopi_passthrough(dut):
    """Only read words with ptr exactly 0/1 are indexed. A body mixing
    a ptr-0 read, a NON-READ word, and a ptr-5 (gradient) read emits
    the read indexed and the other two bit-identical every
    iteration."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    dut.prog_wr_en.value = 0
    dut.prog_wr_data.value = 0
    dut.run.value = 0
    await tick(dut, 2)
    dut.rst.value = 0
    await tick(dut)

    p0, p1 = pattern(3), pattern(4)
    rd = read_word(ptr=0, addr=48)
    nr = pattern(9)                       # bit 1 clear: not a read
    gr = read_word(ptr=5, addr=80)        # gradient read: out of scope
    prog = [p0, loopi_word(3, 3, stride_a=16, stride_w=8), rd, nr, gr, p1]
    expected = [p0, 0,
                read_word(0, 48), nr, gr,
                read_word(0, 64), nr, gr,
                read_word(0, 80), nr, gr,
                p1]

    await fresh_load(dut, prog)
    await check_emission(dut, expected)

    print("instr_seq_nxn LOOPI passthrough OK (non-read + ptr-5 words "
          "bit-identical across iterations)")


@cocotb.test()
async def test_instr_seq_nxn_loopi_compat(dut):
    """Backward compatibility: indexed flag CLEAR with nonzero stride
    bits set replays the body bit-identical (plain item-12 LOOP — the
    stride fields are ignored), and LOOPI count=0 skips the body
    entirely (one bubble)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    dut.prog_wr_en.value = 0
    dut.prog_wr_data.value = 0
    dut.run.value = 0
    await tick(dut, 2)
    dut.rst.value = 0
    await tick(dut)

    p0, b0, b1, p1 = (pattern(5), read_word(0, 32), read_word(1, 64),
                      pattern(6))

    cases = [
        # flag clear + stride bits nonzero: exact item-12 behavior
        ([p0, loopi_word(2, 2, stride_a=9, stride_w=11, indexed=False),
          b0, b1, p1],
         [p0, 0, b0, b1, b0, b1, p1], "flag clear => strides ignored"),
        # count=0: body skipped, one bubble, strides irrelevant
        ([p0, loopi_word(0, 2, stride_a=9, stride_w=11), b0, b1, p1],
         [p0, 0, p1], "count=0 (body skipped)"),
    ]
    for prog, expected, name in cases:
        await fresh_load(dut, prog)
        await check_emission(dut, expected)
        print(f"  {name}: OK ({len(expected)} emissions)")

    print("instr_seq_nxn LOOPI backward compat + count=0 OK")
