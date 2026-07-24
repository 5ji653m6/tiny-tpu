"""Gate test for roadmap item 12 (top half): the attention phase-1
composite executed from a LOOPED LOADED PROGRAM on tpu_nxn_prog —
matmul -> softmax repeated THREE times by a single LOOP control word.
Written by the harness author, not the agent — per tinytpu-loop README
"the gate grows with the design". Red-first: FAILS until the item-12
sequencer RTL lands (BASELINE_EXCLUDE'd until then).

This is the payoff of the loop construct: the 9b/10 programs were
straight-line (the two-step training program unrolled its 136-word
step twice). Here the body is genuinely re-entrant — every address it
references is an INPUT (K @16 weight walk, Q @0 activations), and the
outputs append at the UB write pointer — so one 52-word body run
three times produces three identical P images:

    S = Q @ K^T (weight walk K@16 T=1, activations Q@0 T=0)
    P = softmax(S) per row (pathway bit 6)
    rep 1 -> P @32-47, rep 2 -> P @48-63, rep 3 -> P @64-79
    (host image is 32 words: Q@0, K@16)

The program is 62 words: 8 host-write beats + trailing idle + the
LOOP word (count=3, len=52) + the 52-word phase-1 body (1 weight
read + 9 walk + 1 activation read + 1 switch + 40 drain). Without
the loop construct the same computation takes 165 words.

Golden: the hardware-exact integer model from test_tpu_nxn_ic_attn_n4
(same Q/K stimulus, so the same S/P raws); every rep must produce
bit-identical P.

Checks (LIVE asserts — PYTHONOPTIMIZE is empty):
  - per-lane VPU streams capture exactly 12 beats: three P columns
    back to back, each beat-exact against the integer golden;
  - the full 80-word UB image: Q/K intact @0/@16, P x3 @32/@48/@64.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# ProgGen + Q8.8 helpers from the 9b test; exact attention golden
# (stimulus + S/P raws) from the item-11 test.
from test_tpu_nxn_prog_n4 import ProgGen, to_fixed, from_fixed
from test_tpu_nxn_ic_attn_n4 import Q, K, P_RAW, PATHWAY_SOFTMAX

N = int(os.environ.get("TPU_NXN_PROG_N", "4"))
WORD_W = 133 + 17 * (N - 2)   # legacy instruction width
CTRL = 1 << WORD_W            # control-word escape bit (prog-word MSB)

LOOP_COUNT = 3
BODY_LEN = 52                 # 1 + 9 + 1 + 1 + 40


def loop_word(count, length):
    return CTRL | (count << 8) | length


def generate_program_attn_loop():
    """62 words: 8 host-write beats (Q@0, K@16) + trailing idle +
    LOOP(3, 52) + the phase-1 body. The body is re-entrant: it reads
    only input regions; P appends at the UB write pointer each rep.
    (ProgGen's leading idle() only primes `cur` — it is committed by
    the next tick, so the prefix is 8 beats + 1 trailing idle = 9.)"""
    g = ProgGen()
    g.idle()
    words = ([to_fixed(v) for row in Q for v in row]
             + [to_fixed(v) for row in K for v in row])
    assert len(words) % N == 0
    for b in range(len(words) // N):
        g.write_beat([words[N * b + (N - 1 - i)] for i in range(N)])
    g.tick()  # trailing idle cycle after the last beat
    assert len(g.prog) == 9, f"prefix is {len(g.prog)} words, expected 9"

    g.prog.append(loop_word(LOOP_COUNT, BODY_LEN))

    # body (52 words): S = Q@K^T with the softmax pathway
    g.issue_read(ptr=1, addr=16, rows=4, cols=4, transpose=1)
    g.tick(9)
    g.issue_read(ptr=0, addr=0, rows=4, cols=4, pathway=PATHWAY_SOFTMAX)
    g.switch_pulse()
    g.tick(40)

    return g.prog


async def tick(dut, cycles=1):
    """Edge + 1ns settle: reads see post-edge values, drives land
    mid-cycle."""
    for _ in range(cycles):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")


@cocotb.test()
async def test_tpu_nxn_prog_attn_loop_n4(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    prog = generate_program_attn_loop()
    assert len(prog) == 62, f"program is {len(prog)} words, expected 62"
    nxn = dut.tpu_nxn_prog_inst.tpu_nxn_ic_inst.tpu_nxn_inst

    # Reset (the program load happens with the chip held in reset).
    dut.rst.value = 1
    dut.prog_wr_en.value = 0
    dut.prog_wr_data.value = 0
    dut.run.value = 0
    dut.learning_rate_in.value = to_fixed(0.5)  # inert: no lr_d stage
    await tick(dut, 2)

    # Load the program, one word per cycle.
    for w in prog:
        dut.prog_wr_data.value = w
        dut.prog_wr_en.value = 1
        await tick(dut)
    dut.prog_wr_en.value = 0
    dut.prog_wr_data.value = 0
    await tick(dut, 2)

    # Per-lane collectors on the VPU -> UB write interface.
    lanes = [[] for _ in range(N)]

    async def collect():
        while True:
            await RisingEdge(dut.clk)
            for i in range(N):
                if nxn.ub_wr_valid_in[i].value.integer:
                    lanes[i].append(nxn.ub_wr_data_in[i].value.integer
                                    & 0xFFFF)

    collector = cocotb.start_soon(collect())

    # Release reset and run — ONE pulse; the LOOP word does the rest.
    dut.rst.value = 0
    await tick(dut)
    dut.run.value = 1
    await tick(dut)
    dut.run.value = 0

    # Emission: 62 loaded words + 2 extra body passes (52 each) = 166
    # cycles; the last body's 40 idles cover the final P stream.
    await tick(dut, 166 + 20)
    collector.kill()

    # ---- per-lane streams: P x3 (12 beats per lane), bit-identical ----
    for j in range(N):
        for k in range(LOOP_COUNT):
            got = lanes[j][4 * k:4 * k + 4]
            expected = [P_RAW[r][j] & 0xFFFF for r in range(4)]
            assert got == expected, (
                f"VPU lane {j} P rep {k + 1}: got "
                f"{[f'{from_fixed(w):+.4f}' for w in got]}, expected "
                f"{[f'{from_fixed(w):+.4f}' for w in expected]} "
                f"(column {j})")
        assert len(lanes[j]) == 4 * LOOP_COUNT, (
            f"VPU lane {j}: expected exactly {4 * LOOP_COUNT} beats "
            f"(P x{LOOP_COUNT}), got {len(lanes[j])}")

    # ---- final UB image: Q/K intact, P appended once per rep ----
    for rep in range(LOOP_COUNT):
        base = 32 + 16 * rep
        for r in range(4):
            for c in range(4):
                a = base + 4 * r + c
                got = nxn.ub_inst.ub_memory[a].value.integer & 0xFFFF
                want = P_RAW[r][c] & 0xFFFF
                assert got == want, (
                    f"P rep {rep + 1}: mem[{a}] = "
                    f"{from_fixed(got):+.4f}, expected "
                    f"{from_fixed(want):+.4f}")
    # Q/K input regions untouched.
    qr = [[to_fixed(v) for v in row] for row in Q]
    kr = [[to_fixed(v) for v in row] for row in K]
    for r in range(4):
        for c in range(4):
            for base, img, name in [(0, qr, "Q"), (16, kr, "K")]:
                a = base + 4 * r + c
                got = nxn.ub_inst.ub_memory[a].value.integer & 0xFFFF
                assert got == img[r][c], (
                    f"{name} region: mem[{a}] corrupted: "
                    f"{from_fixed(got):+.4f}, expected "
                    f"{from_fixed(img[r][c]):+.4f}")

    print(f"tpu_nxn_prog N={N} LOOPED attention OK (62-word program, "
          f"LOOP count={LOOP_COUNT} len={BODY_LEN}: one 52-word body "
          f"runs 3x, P bit-identical @32/@48/@64; would take 165 "
          f"words unrolled)")
