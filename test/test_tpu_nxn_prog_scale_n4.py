"""Gate test for roadmap item 18a (chip half, agent RTL): the
per-ELEMENT SCALE stage — the adaLN per-element multiply half of DiT
timestep conditioning (the shift half is the item-17a residual add;
item 18b composes them into the full adaLN-modulated denoiser block).
Written by the harness author, not the agent — per tinytpu-loop README
"the gate grows with the design". Red-first: FAILS until the agent's
RTL lands (BASELINE_EXCLUDE'd until then).

Contract under test: a UB read command (bit 1 set) with **ptr = 8**
is a SCALE READ. The UB streams rows x cols values ELEMENTWISE into
the SAME bias operand stream the ptr-2 bias and ptr-7 residual reads
use: lane i's r-th active beat carries
ub_memory[addr + r*col_size + i], with the SAME per-lane skew and
active window (the ptr-7 linear walk). The consumer is a NEW
per-element multiply stage at the HEAD of the VPU chain (before bias):

    C = (A @ W) . S        (elementwise fxp_mul, clamp16 saturating)

Mechanics (all mirroring item 17a):
  - the UB gains an rd_bias_scale flag: SET by a ptr-8 read, CLEARED
    by a ptr-2 (bias) or ptr-7 (residual) read; while set, the operand
    stream's window is ALSO driven on a new per-lane scale-valid
    output (data + valid ride the existing channel, registered once
    in tpu_nxn like the other operand streams);
  - the VPU gains a scale stage at the chain head, routed in when the
    (registered) scale-arm flag is set and bypassed combinationally
    when clear — pre-item-18a programs contain no ptr-8 reads, the
    flag stays clear, and the whole chain is bit- and latency-
    identical;
  - the scale stage multiplies its input beat by the operand beat
    ONLY on scale-valid beats and passes the input through otherwise —
    a STALE-armed phase (a ptr-8 read in an earlier phase, no operand
    read in this one) is an exact passthrough;
  - no new pathway bit (all 8 are taken); the scale phase's pathway
    is 0 (bypass) — the operand read alone arms the multiply;
  - under LOOPI a ptr-8 body read advances by i*stride_a (scale
    matrices are activation-like per-iteration data) — the sequencer
    half is gated by test_instr_seq_nxn_scale.

Three phases pin the contract:
  P1 (armed):       C1 = (X @ W1) . S1   @80  — the multiply itself
  P2 (stale-armed): C2 = X @ W2          @96  — bypass, NO scale read:
                    proves the stale flag's passthrough is exact
  P3 (scale-of-chip-region): C3 = (X @ W3) . C1  @112 — the scale
                    matrix is a region a previous phase PRODUCED (the
                    adaLN pattern: modulation data computed on chip)

Host image: X @0, W1 @16, W2 @32, W3 @48, S1 @64 = 80 words; streams
append @80/@96/@112 -> the image is exactly UB_WIDTH=128 words.
Program: 21-word prefix + 55 + 52 + 55 = 183 words (a scale phase
carries one extra read word + 3 alignment cycles vs a plain phase,
identical to an radd phase).

Golden: matmul_g (per-step fxp rounding) + fxp_mul elementwise, on
signed-raw stimulus (to_raw — to_fixed returns the two's-complement
ENCODING; program host words keep the unsigned encoding).

Checks (LIVE asserts — PYTHONOPTIMIZE is empty):
  - per-lane VPU streams beat-exact: C1, C2, C3 columns (12 beats per
    lane);
  - UB image words 0..127 exact.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# ProgGen + Q8.8 helpers from the 9b test; matmul golden + signed-raw
# converter from the item-14 test; exact multiply from the item-11
# test.
from test_tpu_nxn_prog_n4 import ProgGen, to_fixed, from_fixed
from test_tpu_nxn_prog_mh_attn_n4 import matmul_g, to_raw
from test_tpu_nxn_ic_attn_n4 import fxp_mul

N = int(os.environ.get("TPU_NXN_PROG_N", "4"))

PTR_SCALE = 8
# No pathway bit: the operand read alone arms the multiply; the phase
# pathway is bypass.
PATHWAY_BYPASS = 0b0000000

# Stimulus: quarter/eighth multiples. Scales cluster around 1.0 (the
# adaLN 1+s shape) with a sign flip for coverage; |matmul| * |scale|
# stays far below the Q8.8 clamp.
X = [[ 0.50, -0.375,  1.00,  0.75],
     [-0.625,  1.25, -0.50,  0.25],
     [ 0.25,  0.75, -0.875,  0.50],
     [ 1.00, -0.50,  0.25, -0.125]]
W1 = [[ 0.25,  0.50, -0.625,  1.00],
      [ 1.00, -0.25,  0.50,  0.75],
      [-0.50,  0.75,  1.25, -0.25],
      [ 0.75,  1.00, -0.50,  0.25]]
W2 = [[-0.25,  1.00,  0.75, -0.50],
      [ 0.50, -0.875, 1.00,  0.25],
      [ 1.25,  0.25, -0.25,  0.75],
      [ 0.75, -0.50,  0.50,  1.00]]
W3 = [[ 1.00,  0.25, -0.50,  0.75],
      [-0.75,  0.50,  1.00, -0.25],
      [ 0.50,  1.25,  0.25, -0.375],
      [ 0.125, -0.75, 0.75,  0.50]]
S1 = [[ 1.25,   0.75,  1.00, -0.50],
      [ 0.875,  1.50,  1.125, 1.00],
      [ 1.00,  -1.25,  0.625, 1.375],
      [ 0.50,   1.00,  1.25,  0.75]]

Xr = [[to_raw(v) for v in row] for row in X]
W1r = [[to_raw(v) for v in row] for row in W1]
W2r = [[to_raw(v) for v in row] for row in W2]
W3r = [[to_raw(v) for v in row] for row in W3]
S1r = [[to_raw(v) for v in row] for row in S1]


def ew_mul(A, B):
    """Elementwise fxp_mul (saturating, per-step rounding) over
    same-shape matrices."""
    return [[fxp_mul(A[r][c], B[r][c]) for c in range(len(A[0]))]
            for r in range(len(A))]


C1 = ew_mul(matmul_g(Xr, W1r), S1r)   # @80
C2 = matmul_g(Xr, W2r)                # @96 (stale-arm passthrough)
C3 = ew_mul(matmul_g(Xr, W3r), C1)    # @112 (chip-produced scale)

# UB image: host 80 words then C1/C2/C3 appended @80/@96/@112.
GOLD = ([v for row in Xr for v in row]
        + [v for row in W1r for v in row]
        + [v for row in W2r for v in row]
        + [v for row in W3r for v in row]
        + [v for row in S1r for v in row])
assert len(GOLD) == 80
for C in (C1, C2, C3):
    GOLD += [v for row in C for v in row]
assert len(GOLD) == 128


def generate_program_scale():
    """183 words: 21-word host prefix + P1 (scale, 55) + P2 (plain, 52)
    + P3 (scale on C1, 55). A scale phase = weight read -> tick(9) ->
    activation read (pathway 0 — bypass) -> switch -> tick(2) -> SCALE
    read (ptr=8, issued mid-flight like the ptr-7 residual read) ->
    tick(40)."""
    g = ProgGen()
    g.idle()
    host_words = ([to_fixed(v) for row in X for v in row]
                  + [to_fixed(v) for row in W1 for v in row]
                  + [to_fixed(v) for row in W2 for v in row]
                  + [to_fixed(v) for row in W3 for v in row]
                  + [to_fixed(v) for row in S1 for v in row])
    assert len(host_words) == 80 and len(host_words) % N == 0
    for b in range(len(host_words) // N):
        g.write_beat([host_words[N * b + (N - 1 - i)] for i in range(N)])
    g.tick()  # trailing idle cycle after the last beat
    assert len(g.prog) == 21, f"prefix is {len(g.prog)} words, expected 21"

    def scale_phase(w_addr, s_addr):
        g.issue_read(ptr=1, addr=w_addr, rows=4, cols=4, transpose=0)
        g.tick(9)
        g.issue_read(ptr=0, addr=0, rows=4, cols=4, transpose=0,
                     pathway=PATHWAY_BYPASS)
        g.switch_pulse()
        g.tick(2)
        g.issue_read(ptr=PTR_SCALE, addr=s_addr, rows=4, cols=4)
        g.tick(40)

    def plain_phase(w_addr):
        g.issue_read(ptr=1, addr=w_addr, rows=4, cols=4, transpose=0)
        g.tick(9)
        g.issue_read(ptr=0, addr=0, rows=4, cols=4, transpose=0,
                     pathway=PATHWAY_BYPASS)
        g.switch_pulse()
        g.tick(40)

    scale_phase(16, 64)   # P1: C1 = (X@W1) . S1 @80
    plain_phase(32)       # P2: C2 = X@W2        @96 (stale-arm check)
    scale_phase(48, 80)   # P3: C3 = (X@W3) . C1 @112

    return g.prog


async def tick(dut, cycles=1):
    """Edge + 1ns settle: reads see post-edge values, drives land
    mid-cycle."""
    for _ in range(cycles):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")


@cocotb.test()
async def test_tpu_nxn_prog_scale_n4(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    prog = generate_program_scale()
    assert len(prog) == 183, f"program is {len(prog)} words, expected 183"
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

    # Release reset and run — ONE pulse.
    dut.rst.value = 0
    await tick(dut)
    dut.run.value = 1
    await tick(dut)
    dut.run.value = 0

    # Emission: 183 loaded words; the last phase's 40 idles cover the
    # final stream.
    await tick(dut, 183 + 20)
    collector.kill()

    # ---- per-lane streams, beat-exact through C3 ----
    def col(M, j, rows):
        return [M[r][j] & 0xFFFF for r in range(rows)]

    for j in range(N):
        expected = []
        for C in (C1, C2, C3):
            expected += col(C, j, 4)
        assert lanes[j] == expected, (
            f"VPU lane {j}: {len(lanes[j])} beats, expected "
            f"{len(expected)}; got "
            f"{[f'{from_fixed(w):+.4f}' for w in lanes[j][:8]]}..., "
            f"expected "
            f"{[f'{from_fixed(w):+.4f}' for w in expected[:8]]}...")

    # ---- UB image words 0..127 exact (host + C1/C2/C3) ----
    for a in range(128):
        got = nxn.ub_inst.ub_memory[a].value.integer & 0xFFFF
        want = GOLD[a] & 0xFFFF
        assert got == want, (
            f"mem[{a}] = {from_fixed(got):+.4f}, expected "
            f"{from_fixed(want):+.4f}")

    print(f"tpu_nxn_prog N={N} SCALE STAGE OK (183-word program: "
          f"C1 = (X@W1).S1 via a ptr-8 scale read + the head-of-chain "
          f"multiply stage, C2 = X@W2 exact through the stale-armed "
          f"passthrough, C3 = (X@W3).C1 with a chip-produced scale — "
          f"the adaLN per-element multiply half)")
