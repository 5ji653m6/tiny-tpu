"""Gate test for roadmap item 15 (top half): the item-14 multi-head
attention B PHASES (per-head S = Q~ @ K~^T with the softmax pathway)
executed as ONE 52-word body looped TWICE by a single INDEXED LOOP
(LOOPI) control word — the body's read addresses advance by the
per-iteration strides, so head 2's weights/activations are read with
no new program words. Written by the harness author, not the agent —
per tinytpu-loop README "the gate grows with the design". Red-first:
FAILS until the item-15 sequencer RTL lands (BASELINE_EXCLUDE'd until
then).

Item 12's LOOP replays the body verbatim — every iteration reads the
SAME regions, which is why the item-14 program spelled out B1/B2 (and
A1..A6) explicitly. With LOOPI, one B body serves both heads: the
ptr-1 (weight) read starts at K~1 @128 with stride_w = 40 -> K~2
@168 on iteration 2; the ptr-0 (activation) read starts at Q~1 @112
with stride_a = 40 -> Q~2 @152. P appends at the UB write pointer as
always: P1 @192, P2 @208.

The program is 394 words: 28 host-write beats + trailing idle (29 —
the FULL item-14 image including W_O, so the whole 0..223 image check
stays contiguous) + the six A phases spelled out (6 x 52 = 312 —
per-head weight regions are not uniformly strided across Q/K/V, so
indexing them is item 16's tiled-matmul territory) + ONE LOOPI word
(count=2, len=52, stride_a=40, stride_w=40) + the 52-word B body.
Straight-line this would be 29 + 8 x 52 = 445 words.

Golden: the item-14 hardware-exact integer model (same stimulus, same
Q~/K~/V/P raws — imported from test_tpu_nxn_prog_mh_attn_n4).

Checks (LIVE asserts — PYTHONOPTIMIZE is empty):
  - per-lane VPU streams beat-exact through P2: lanes 0/1 capture 32
    beats (Q~1, K~1, V1, Q~2, K~2, V2, P1, P2 columns), lanes 2/3
    capture 24 (no V streams);
  - UB image words 0..223 exact: the 112-word host image, all six A
    streams, and P1 @192 / P2 @208 — P2's presence proves iteration 2
    read the ADVANCED addresses (a verbatim LOOP would re-emit P1).
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# ProgGen + Q8.8 helpers from the 9b test; softmax pathway encoding
# from the item-11 test; the whole item-14 golden (stimulus, padded
# projections, P raws, UB image) from the item-14 test.
from test_tpu_nxn_prog_n4 import ProgGen, to_fixed, from_fixed
from test_tpu_nxn_prog_mh_attn_n4 import (
    X, Wq1, Wk1, Wv1, Wq2, Wk2, Wv2, W_O, pad42,
    Qt1, Kt1, V1, Qt2, Kt2, V2, P1, P2,
    ADDR, GOLD, PATHWAY_SOFTMAX,
)

N = int(os.environ.get("TPU_NXN_PROG_N", "4"))
WORD_W = 134 + 17 * (N - 2)   # instruction width (item 13: SiLU bit)
CTRL = 1 << WORD_W            # control-word escape bit (prog-word MSB)
IDX_FLAG = 1 << 16

LOOP_COUNT = 2
BODY_LEN = 52                 # 1 + 9 + 1 + 1 + 40
STRIDE = 40                   # head h+1's Q~/K~ sit 40 words past head h's


def loopi_word(count, length, stride_a, stride_w):
    return (CTRL | (count << 8) | length | IDX_FLAG
            | (stride_a << 24) | (stride_w << 40))


def generate_program_mh_attn_loopi():
    """394 words: 29-word host prefix + 6 x 52 A phases + LOOPI(2, 52,
    sa=40, sw=40) + the 52-word B body (S = Q~1 @ K~1^T, softmax
    pathway). Iteration 2 reads K~2 @168 / Q~2 @152 via the strides."""
    g = ProgGen()
    g.idle()
    host_words = []
    for M in (X, pad42(Wq1), pad42(Wk1), Wv1,
              pad42(Wq2), pad42(Wk2), Wv2, W_O):
        host_words += [to_fixed(v) for row in M for v in row]
    assert len(host_words) == 112 and len(host_words) % N == 0
    for b in range(len(host_words) // N):
        g.write_beat([host_words[N * b + (N - 1 - i)] for i in range(N)])
    g.tick()  # trailing idle cycle after the last beat
    assert len(g.prog) == 29, f"prefix is {len(g.prog)} words, expected 29"

    def matmul_phase(w_addr, w_rows, w_cols, w_T, a_addr, a_rows, a_cols,
                     a_T, pathway):
        g.issue_read(ptr=1, addr=w_addr, rows=w_rows, cols=w_cols,
                     transpose=w_T)
        g.tick(9)
        g.issue_read(ptr=0, addr=a_addr, rows=a_rows, cols=a_cols,
                     transpose=a_T, pathway=pathway)
        g.switch_pulse()
        g.tick(40)

    # A: projections (bypass pathway) — spelled out (see docstring)
    matmul_phase(ADDR["Wq1p"], 4, 4, 0, ADDR["X"], 4, 4, 0, 0)
    matmul_phase(ADDR["Wk1p"], 4, 4, 0, ADDR["X"], 4, 4, 0, 0)
    matmul_phase(ADDR["Wv1"], 4, 2, 0, ADDR["X"], 4, 4, 0, 0)
    matmul_phase(ADDR["Wq2p"], 4, 4, 0, ADDR["X"], 4, 4, 0, 0)
    matmul_phase(ADDR["Wk2p"], 4, 4, 0, ADDR["X"], 4, 4, 0, 0)
    matmul_phase(ADDR["Wv2"], 4, 2, 0, ADDR["X"], 4, 4, 0, 0)

    # B: ONE indexed body serves both heads — ptr-1 read starts at
    # K~1 @128 (stride_w=40 -> K~2 @168), ptr-0 read at Q~1 @112
    # (stride_a=40 -> Q~2 @152). P1 @192 / P2 @208 append at wr_ptr.
    g.prog.append(loopi_word(LOOP_COUNT, BODY_LEN, STRIDE, STRIDE))
    g.issue_read(ptr=1, addr=ADDR["Kt1"], rows=4, cols=4, transpose=1)
    g.tick(9)
    g.issue_read(ptr=0, addr=ADDR["Qt1"], rows=4, cols=4,
                 pathway=PATHWAY_SOFTMAX)
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
async def test_tpu_nxn_prog_mh_attn_loopi_n4(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    prog = generate_program_mh_attn_loopi()
    assert len(prog) == 394, f"program is {len(prog)} words, expected 394"
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

    # Release reset and run — ONE pulse; the LOOPI word does the rest.
    dut.rst.value = 0
    await tick(dut)
    dut.run.value = 1
    await tick(dut)
    dut.run.value = 0

    # Emission: 394 loaded words + 1 extra body pass (52) = 446 cycles;
    # the last body's 40 idles cover the final P stream.
    await tick(dut, 446 + 20)
    collector.kill()

    # ---- per-lane streams, beat-exact through P2 ----
    def col(M, j, rows):
        return [M[r][j] & 0xFFFF for r in range(rows)]

    for j in range(N):
        expected = []
        for M in (Qt1, Kt1):
            expected += col(M, j, 4)
        if j < 2:
            expected += col(V1, j, 4)
        for M in (Qt2, Kt2):
            expected += col(M, j, 4)
        if j < 2:
            expected += col(V2, j, 4)
        for M in (P1, P2):
            expected += col(M, j, 4)
        assert lanes[j] == expected, (
            f"VPU lane {j}: {len(lanes[j])} beats, expected "
            f"{len(expected)}; got "
            f"{[f'{from_fixed(w):+.4f}' for w in lanes[j][:8]]}..., "
            f"expected "
            f"{[f'{from_fixed(w):+.4f}' for w in expected[:8]]}...")

    # ---- UB image words 0..223 exact (host + A streams + P1/P2) ----
    for a in range(224):
        got = nxn.ub_inst.ub_memory[a].value.integer & 0xFFFF
        want = GOLD[a] & 0xFFFF
        assert got == want, (
            f"mem[{a}] = {from_fixed(got):+.4f}, expected "
            f"{from_fixed(want):+.4f}")

    print(f"tpu_nxn_prog N={N} INDEXED-LOOP multi-head attention OK "
          f"(394-word program: 6 spelled-out A phases + ONE 52-word B "
          f"body looped 2x with strides 40/40 — head 2's reads advance "
          f"with no new program words; P1 @192 / P2 @208 exact; "
          f"straight-line would take 445 words)")
