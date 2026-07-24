"""Gate test for roadmap item 10: TWO in-place training steps executed
from a SINGLE LOADED PROGRAM on tpu_nxn_prog — the convergence of item
9a (two-step training choreography) and item 9b (program buffer).
Written by the harness author, not the agent — per tinytpu-loop README
"the gate grows with the design".

No new RTL: tpu_nxn_prog already exposes PROG_DEPTH; the two-step
program is 286 words (step 1: the 9b 150-word program with host image
writes; step 2: the 136-word training-step choreography with NO host
writes and the pass-2 dZ read retargeted to @84 — streams append at
the UB write pointer, see test_tpu_nxn_ic_train2_n4's docstring), so
the harness dump wrapper is compiled with -Pdump.PROG_DEPTH=512.

The program is generated OFFLINE by ProgGen (imported from
test_tpu_nxn_prog_n4 — the cycle-exact mirror of the 8b Host), with
the 9a INTEGER stimulus (8b matrices x2) so both composed steps stay
Q8.8-exact. The host loads the 285 words during reset, pulses run
ONCE, and never drives anything again: the chip trains twice
autonomously.

Checks (LIVE asserts — PYTHONOPTIMIZE is empty), identical to the 9a
test: 16 beats per lane (dZ1, G1, dZ2, G2 beat-exact against the
twice-iterated numpy golden) and the full 116-word UB image (X/Y
untouched, W''/B'' in place @16/@32, dZ1/G1 intact @52/@68, dZ2/G2
appended @84/@100).
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# ProgGen + Q8.8 helpers from the 9b test; integer stimulus, twice-
# iterated golden, and per-step dZ/G matrices from the 9a test.
from test_tpu_nxn_prog_n4 import ProgGen, to_fixed, from_fixed
from test_tpu_nxn_ic_train2_n4 import X, W, B, Y, LR, GOLD, dZ1, G1, \
    dZ2, G2

N = int(os.environ.get("TPU_NXN_PROG_N", "4"))


def training_step(g, dz_addr):
    """One training step as a word stream (136 cycles): the 8b/9b
    pass-1/pass-2 choreography, with the pass-2 dZ read at dz_addr
    (step 1's dZ @52, step 2's @84 — streams append at wr_ptr)."""
    # pass 1: forward + loss (transition pathway)
    g.issue_read(ptr=1, addr=16, rows=4, cols=4, transpose=1)
    g.tick(9)
    g.issue_read(ptr=0, addr=0, rows=4, cols=4, pathway=0b0001111)
    g.switch_pulse()
    g.tick(2)
    g.issue_read(ptr=2, addr=32, rows=4, cols=4)
    g.tick(1)
    g.issue_read(ptr=3, addr=36, rows=4, cols=4)
    g.tick(2)
    g.issue_read(ptr=5, addr=32, rows=4, cols=4)
    g.tick(40)

    # pass 2: weight gradient G = dZ^T @ X
    g.issue_read(ptr=1, addr=0, rows=4, cols=4, transpose=0)
    g.tick(9)
    g.issue_read(ptr=0, addr=dz_addr, rows=4, cols=4, transpose=1,
                 pathway=0b0000000)
    g.switch_pulse()
    g.tick(3)
    g.issue_read(ptr=6, addr=16, rows=4, cols=4)
    g.tick(60)


def generate_program_train2():
    """286 words: 13 host-write beats + trailing idle (= 14), step 1
    (136) reading dZ @52, step 2 (136) reading dZ @84. Step 2 needs NO
    host writes — it trains on step 1's in-place W'/B'."""
    g = ProgGen()
    g.idle()
    words = ([to_fixed(v) for v in X.flatten()]
             + [to_fixed(v) for v in W.flatten()]
             + [to_fixed(v) for v in B]
             + [to_fixed(v) for v in Y.flatten()])
    assert len(words) % N == 0
    for b in range(len(words) // N):
        g.write_beat([words[N * b + (N - 1 - i)] for i in range(N)])
    g.tick()  # trailing idle cycle after the last beat

    training_step(g, dz_addr=52)
    training_step(g, dz_addr=84)
    return g.prog


async def tick(dut, cycles=1):
    """Edge + 1ns settle: reads see post-edge values, drives land
    mid-cycle."""
    for _ in range(cycles):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")


@cocotb.test()
async def test_tpu_nxn_prog_train2_n4(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    prog = generate_program_train2()
    assert len(prog) == 286, f"program is {len(prog)} words, expected 286"
    nxn = dut.tpu_nxn_prog_inst.tpu_nxn_ic_inst.tpu_nxn_inst

    # Reset (the program load happens with the chip held in reset).
    dut.rst.value = 1
    dut.prog_wr_en.value = 0
    dut.prog_wr_data.value = 0
    dut.run.value = 0
    dut.learning_rate_in.value = to_fixed(LR)
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

    # Release reset and run the program — ONE pulse, then the host is
    # silent for both training steps.
    dut.rst.value = 0
    await tick(dut)
    dut.run.value = 1
    await tick(dut)
    dut.run.value = 0

    # The program self-paces: wait it out plus margin.
    await tick(dut, len(prog) + 20)
    collector.kill()

    # ---- per-lane streams: dZ1, G1, dZ2, G2 (16 beats per lane) ----
    for j in range(N):
        for k, (name, mat) in enumerate([("dZ1", dZ1), ("G1", G1),
                                         ("dZ2", dZ2), ("G2", G2)]):
            got = lanes[j][4 * k:4 * k + 4]
            expected = [to_fixed(mat[r][j]) for r in range(4)]
            assert got == expected, (
                f"VPU lane {j} {name}: got "
                f"{[f'{from_fixed(w):+.4f}' for w in got]}, expected "
                f"{[f'{from_fixed(w):+.4f}' for w in expected]} "
                f"(column {j})")
        assert len(lanes[j]) == 16, (
            f"VPU lane {j}: expected exactly 16 beats "
            f"(dZ1,G1,dZ2,G2), got {len(lanes[j])}")

    # ---- final UB image: both steps' regions, exact everywhere ----
    for base, count, name in [(0, 16, "X"), (16, 16, "W''"),
                              (32, 4, "B''"), (36, 16, "Y"),
                              (52, 16, "dZ1"), (68, 16, "G1"),
                              (84, 16, "dZ2"), (100, 16, "G2")]:
        for a in range(base, base + count):
            got = nxn.ub_inst.ub_memory[a].value.integer & 0xFFFF
            want = to_fixed(GOLD[a])
            assert got == want, (
                f"{name} region: mem[{a}] = {from_fixed(got):+.4f}, "
                f"expected {GOLD[a]:+.4f}")

    print(f"tpu_nxn_prog N={N} program-driven TWO training steps OK "
          f"({len(prog)}-word loaded program, one run pulse, zero host "
          f"traffic; same golden as test_tpu_nxn_ic_train2_n4)")
