"""Gate test for roadmap item 17a (chip half, agent RTL): the
RESIDUAL ADD. Written by the harness author, not the agent — per
tinytpu-loop README "the gate grows with the design". Red-first:
FAILS until the agent's RTL lands (BASELINE_EXCLUDE'd until then).

A DiT block is two residual connections around attention and the MLP:
x2 = x + attn(...), x3 = x2 + mlp(...). tinyTPU has no elementwise
matrix add exposed as an instruction — but it HAS the full operand
path for one: the ptr-2 BIAS read streams UB values into the VPU's
bias stream, and the bias stage (pathway bit 3, HEAD of the VPU
chain) adds them to the systolic output beat-by-beat with fxp_add.
The bias read is per-COLUMN (lane i's value held for all rows) — a
residual needs per-ELEMENT.

Contract under test: a UB read command (bit 1 set) with **ptr = 7**
is a RESIDUAL READ. The UB streams rows x cols values ELEMENTWISE
into the same bias operand stream: lane i's r-th active beat carries
ub_memory[addr + r*col_size + i], with the SAME per-lane skew and
active window as the ptr-2 bias schedule. Issued mid-phase (like the
legacy bias read: after the activation read + switch, inside the
output wait), it arms the bias stage for that phase's output stream;
the phase's pathway is 0b1000 (bias stage only), so the writeback is

    C = A @ W + R        (elementwise fxp_add, clamp16 saturating)

No instruction width changes; pre-item-17a programs contain no ptr-7
reads and replay bit-identically.

Three phases pin the contract:
  P1 (armed):   C1 = X @ W1 + R1           @80  — the add itself
  P2 (disarmed): C2 = X @ W2               @96  — bypass, NO residual
                read: proves the arming does not leak across phases
  P3 (residual-of-chip-region): C3 = X @ W3 + C1  @112 — the residual
                is a region a previous phase PRODUCED (the DiT
                pattern: x2 = x + attn feeds x3 = x2 + mlp)

Host image: X @0, W1 @16, W2 @32, W3 @48, R1 @64 = 80 words; streams
append @80/@96/@112 -> the image is exactly UB_WIDTH=128 words.
Program: 21-word prefix + 55 + 52 + 55 = 183 words (an radd phase
carries one extra read word + 3 alignment cycles vs a plain phase).

Golden: matmul_g (per-step fxp rounding) + fxp_add elementwise, on
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
# converter from the item-14 test; exact add from the item-11 test.
from test_tpu_nxn_prog_n4 import ProgGen, to_fixed, from_fixed
from test_tpu_nxn_prog_mh_attn_n4 import matmul_g, to_raw
from test_tpu_nxn_ic_attn_n4 import fxp_add

N = int(os.environ.get("TPU_NXN_PROG_N", "4"))

PTR_RESIDUAL = 7
PATHWAY_BIAS = 0b1000       # bias stage only: out = sys + residual

# Stimulus: quarter/eighth multiples, |matmul| + |R| << clamp.
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
R1 = [[ 0.125, -0.25,  0.50, -0.375],
      [ 0.75,   0.25, -0.625, 0.50],
      [-0.50,   1.00,  0.25, -0.75],
      [ 0.25,  -0.875, 0.75,  0.125]]

Xr = [[to_raw(v) for v in row] for row in X]
W1r = [[to_raw(v) for v in row] for row in W1]
W2r = [[to_raw(v) for v in row] for row in W2]
W3r = [[to_raw(v) for v in row] for row in W3]
R1r = [[to_raw(v) for v in row] for row in R1]


def ew_add(A, B):
    """Elementwise fxp_add (saturating) over same-shape matrices."""
    return [[fxp_add(A[r][c], B[r][c]) for c in range(len(A[0]))]
            for r in range(len(A))]


C1 = ew_add(matmul_g(Xr, W1r), R1r)   # @80
C2 = matmul_g(Xr, W2r)                # @96 (no residual read issued)
C3 = ew_add(matmul_g(Xr, W3r), C1)    # @112 (chip-produced residual)

# UB image: host 80 words then C1/C2/C3 appended @80/@96/@112.
GOLD = ([v for row in Xr for v in row]
        + [v for row in W1r for v in row]
        + [v for row in W2r for v in row]
        + [v for row in W3r for v in row]
        + [v for row in R1r for v in row])
assert len(GOLD) == 80
for C in (C1, C2, C3):
    GOLD += [v for row in C for v in row]
assert len(GOLD) == 128


def generate_program_radd():
    """183 words: 21-word host prefix + P1 (radd, 55) + P2 (plain, 52)
    + P3 (radd on C1, 55). An radd phase = weight read -> tick(9) ->
    activation read (pathway 0b1000) -> switch -> tick(2) -> RESIDUAL
    read (ptr=7, issued mid-flight like the legacy ptr-2 bias read) ->
    tick(40)."""
    g = ProgGen()
    g.idle()
    host_words = ([to_fixed(v) for row in X for v in row]
                  + [to_fixed(v) for row in W1 for v in row]
                  + [to_fixed(v) for row in W2 for v in row]
                  + [to_fixed(v) for row in W3 for v in row]
                  + [to_fixed(v) for row in R1 for v in row])
    assert len(host_words) == 80 and len(host_words) % N == 0
    for b in range(len(host_words) // N):
        g.write_beat([host_words[N * b + (N - 1 - i)] for i in range(N)])
    g.tick()  # trailing idle cycle after the last beat
    assert len(g.prog) == 21, f"prefix is {len(g.prog)} words, expected 21"

    def radd_phase(w_addr, r_addr):
        g.issue_read(ptr=1, addr=w_addr, rows=4, cols=4, transpose=0)
        g.tick(9)
        g.issue_read(ptr=0, addr=0, rows=4, cols=4, transpose=0,
                     pathway=PATHWAY_BIAS)
        g.switch_pulse()
        g.tick(2)
        g.issue_read(ptr=PTR_RESIDUAL, addr=r_addr, rows=4, cols=4)
        g.tick(40)

    def plain_phase(w_addr):
        g.issue_read(ptr=1, addr=w_addr, rows=4, cols=4, transpose=0)
        g.tick(9)
        g.issue_read(ptr=0, addr=0, rows=4, cols=4, transpose=0,
                     pathway=0)
        g.switch_pulse()
        g.tick(40)

    radd_phase(16, 64)    # P1: C1 = X@W1 + R1 @80
    plain_phase(32)       # P2: C2 = X@W2      @96 (disarm check)
    radd_phase(48, 80)    # P3: C3 = X@W3 + C1 @112

    return g.prog


async def tick(dut, cycles=1):
    """Edge + 1ns settle: reads see post-edge values, drives land
    mid-cycle."""
    for _ in range(cycles):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")


@cocotb.test()
async def test_tpu_nxn_prog_radd_n4(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    prog = generate_program_radd()
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
        got = nxn.ub_inst.ub_sram.mem[a].value.integer & 0xFFFF
        want = GOLD[a] & 0xFFFF
        assert got == want, (
            f"mem[{a}] = {from_fixed(got):+.4f}, expected "
            f"{from_fixed(want):+.4f}")

    print(f"tpu_nxn_prog N={N} RESIDUAL ADD OK (183-word program: "
          f"C1 = X@W1 + R1 via a ptr-7 residual read + the bias stage, "
          f"C2 = X@W2 clean (no leak), C3 = X@W3 + C1 with a "
          f"chip-produced residual — the DiT residual pattern)")
