"""Gate test for roadmap item 14: MULTI-HEAD ATTENTION as ONE loaded
program on tpu_nxn_prog — H=2 heads, d_model=4, d_head=2, R=4 tokens,
with per-head Q/K/V input projections and the merged output
projection O = [O1|O2] @ W_O. Written by the harness author, not the
agent — per tinytpu-loop README "the gate grows with the design".

No new RTL: this reuses src/tpu_nxn_prog.sv (items 9b/12/13) with
dump-parameter passthroughs only (PROG_DEPTH=1024 for the 601-word
program, UB_WIDTH=256 for the 256-word image — the legacy 128-word UB
cannot hold a multi-head choreography).

The merge is the interesting part. tinyTPU has no elementwise matrix
add, and VPU streams only append at the UB write pointer — so the
textbook column-concat [O1|O2] cannot be assembled by placement. The
trick: emit each head's output TRANSPOSED,

    O_h^T = V_h^T @ P_h^T      (2x4, all reads gated shapes)

so the two 2x4 streams land CONTIGUOUS and the 16-word region
[O1^T; O2^T] read row-major IS the 4x4 matrix ([O1|O2])^T. A
transposed activation read then presents [O1|O2] to the array, and
ONE more matmul applies W_O. No adder, no placement control, no new
hardware — multi-head merge from the transpose-read alone.

Q/K projections use zero-PADDED 4x4 weights [W|0] so the S = Q@K^T
phase keeps the fully-gated 4x4 shape at K-dim=4 (the zero columns
contribute exactly 0 through the PE's per-step rounding). V stays
unpadded (4x2) so O_h^T is 2x4 and the contiguous-stack trick works.
This test thereby also gates two output writeback shapes no earlier
test exercised: 4x2 (V_h) and 2x4 (O_h^T) VPU streams.

Choreography (11 phases, one run pulse, zero host rewrites; outputs
append at the UB write pointer):

    host image (112 words):
      X @0 (4x4), Wq1p @16 / Wk1p @32 (4x4 padded), Wv1 @48 (4x2),
      Wq2p @56 / Wk2p @72 (4x4 padded), Wv2 @88 (4x2), W_O @96 (4x4)
    streams:
      A1..A6  Q~1 @112, K~1 @128 (4x4 padded), V1 @144 (4x2),
              Q~2 @152, K~2 @168, V2 @184
      B1, B2  P_h = softmax(Q~_h @ K~_h^T) @192 / @208  (softmax pathway)
      C1, C2  O_h^T = V_h^T @ P_h^T @224 / @232 (2x4, bypass pathway)
      D       O = [O1|O2] @ W_O @240 (act @224 read T=1)

Golden: the hardware-exact integer model from test_tpu_nxn_ic_attn_n4
(fxp_mul/fxp_add per-PE-step rounding, exact exp LUT, fxp_div strict
round-half-down), generalized to (R x K) @ (K x C).

Checks (LIVE asserts -- PYTHONOPTIMIZE is empty):
  - per-lane VPU streams beat-exact against the integer golden:
    lanes 0/1 capture 40 beats, lanes 2/3 capture 32 (the 4x2 V_h
    streams occupy lanes 0/1 only; padded Q~/K~ columns stream zeros);
  - the full 256-word UB image exact everywhere (host regions intact,
    every stream's append-at-wr_ptr placement itself part of the check).
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# ProgGen + Q8.8 helpers from the 9b test; exact arithmetic models +
# softmax golden from the item-11 test.
from test_tpu_nxn_prog_n4 import ProgGen, to_fixed, from_fixed
from test_tpu_nxn_ic_attn_n4 import (
    fxp_mul, fxp_add, softmax_row_exact,
    PATHWAY_SOFTMAX, PATHWAY_BYPASS,
)

N = int(os.environ.get("TPU_NXN_PROG_N", "4"))

# ---- stimulus: quarter multiples, exact in Q8.8 ----
X = [[0.50, -0.25, 0.75, 0.00],
     [-0.50, 0.25, 0.00, 0.50],
     [0.25, 0.50, -0.75, 0.25],
     [0.00, -0.50, 0.25, -0.25]]

# Per-head 4x2 projection weights (d_model=4 -> d_head=2).
Wq1 = [[0.50, -0.25], [0.25, 0.50], [-0.50, 0.00], [0.75, -0.50]]
Wk1 = [[-0.25, 0.50], [0.50, 0.25], [0.00, -0.75], [0.25, 0.50]]
Wv1 = [[0.25, 0.50], [-0.50, 0.25], [0.75, 0.00], [0.00, -0.25]]
Wq2 = [[-0.50, 0.25], [0.00, -0.50], [0.75, 0.50], [-0.25, 0.00]]
Wk2 = [[0.50, -0.50], [-0.25, 0.75], [0.25, 0.00], [0.00, 0.25]]
Wv2 = [[-0.25, 0.00], [0.50, -0.25], [0.00, 0.50], [0.25, 0.75]]

# Merged output projection (4x4: [O1|O2] -> O).
W_O = [[0.50, 0.25, -0.50, 0.00],
       [-0.25, 0.75, 0.00, 0.50],
       [0.00, -0.25, 0.50, -0.75],
       [0.75, 0.00, -0.25, 0.25]]


def pad42(W):
    """Zero-pad a 4x2 weight to 4x4 ([W|0]): Q~/K~ keep the gated 4x4
    shape and the zero columns contribute exactly 0 to S."""
    return [row + [0.0, 0.0] for row in W]


def to_raw(v):
    """Signed raw Q8.8 integer (to_fixed returns the 16-bit two's
    complement encoding)."""
    w = to_fixed(v)
    return w - 0x10000 if w & 0x8000 else w


def matmul_g(A, B):
    """(R x K) @ (K x C) with the PE's per-step rounding: each product
    is fxp_mul-rounded and each accumulation step is an fxp_add
    (saturating), folded in k order."""
    R, K, C = len(A), len(B), len(B[0])
    return [[__import__("functools").reduce(
        lambda acc, k: fxp_add(acc, fxp_mul(A[i][k], B[k][j])),
        range(K), 0) for j in range(C)] for i in range(R)]


def transpose(M):
    return [[M[r][c] for r in range(len(M))] for c in range(len(M[0]))]


# ---- golden: the whole 11-phase choreography in raw Q8.8 ----
Xr = [[to_raw(v) for v in row] for row in X]


def head_golden(Wq, Wk, Wv):
    """Q~/K~ (padded 4x4), V (4x2), S, P, O^T (2x4) for one head."""
    Wqpr = [[to_raw(v) for v in row] for row in pad42(Wq)]
    Wkpr = [[to_raw(v) for v in row] for row in pad42(Wk)]
    Wvr = [[to_raw(v) for v in row] for row in Wv]
    Qt = matmul_g(Xr, Wqpr)                 # [Q|0] 4x4
    Kt = matmul_g(Xr, Wkpr)                 # [K|0] 4x4
    V = matmul_g(Xr, Wvr)                   # 4x2
    S = matmul_g(Qt, transpose(Kt))         # 4x4 (zeros exact)
    P = [softmax_row_exact(row) for row in S]
    OT = matmul_g(transpose(V), transpose(P))  # 2x4
    return Wqpr, Wkpr, Wvr, Qt, Kt, V, S, P, OT


(Wq1pr, Wk1pr, Wv1r, Qt1, Kt1, V1, S1, P1, O1T) = head_golden(Wq1, Wk1, Wv1)
(Wq2pr, Wk2pr, Wv2r, Qt2, Kt2, V2, S2, P2, O2T) = head_golden(Wq2, Wk2, Wv2)

# Merge: C = [O1|O2] from the transposed outputs, then O = C @ W_O.
C_MERGE = [[O1T[0][r], O1T[1][r], O2T[0][r], O2T[1][r]] for r in range(4)]
W_Or = [[to_raw(v) for v in row] for row in W_O]
O_RAW = matmul_g(C_MERGE, W_Or)

# ---- UB image (256 words) ----
ADDR = {"X": 0, "Wq1p": 16, "Wk1p": 32, "Wv1": 48,
        "Wq2p": 56, "Wk2p": 72, "Wv2": 88, "W_O": 96,
        "Qt1": 112, "Kt1": 128, "V1": 144,
        "Qt2": 152, "Kt2": 168, "V2": 184,
        "P1": 192, "P2": 208, "O1T": 224, "O2T": 232, "O": 240}
GOLD = {}


def put(base, M, rows, cols):
    for r in range(rows):
        for c in range(cols):
            GOLD[base + cols * r + c] = M[r][c]


put(ADDR["X"], Xr, 4, 4)
put(ADDR["Wq1p"], Wq1pr, 4, 4)
put(ADDR["Wk1p"], Wk1pr, 4, 4)
put(ADDR["Wv1"], Wv1r, 4, 2)
put(ADDR["Wq2p"], Wq2pr, 4, 4)
put(ADDR["Wk2p"], Wk2pr, 4, 4)
put(ADDR["Wv2"], Wv2r, 4, 2)
put(ADDR["W_O"], W_Or, 4, 4)
put(ADDR["Qt1"], Qt1, 4, 4)
put(ADDR["Kt1"], Kt1, 4, 4)
put(ADDR["V1"], V1, 4, 2)
put(ADDR["Qt2"], Qt2, 4, 4)
put(ADDR["Kt2"], Kt2, 4, 4)
put(ADDR["V2"], V2, 4, 2)
put(ADDR["P1"], P1, 4, 4)
put(ADDR["P2"], P2, 4, 4)
put(ADDR["O1T"], O1T, 2, 4)
put(ADDR["O2T"], O2T, 2, 4)
put(ADDR["O"], O_RAW, 4, 4)
assert len(GOLD) == 256, f"image is {len(GOLD)} words, expected 256"


def generate_program_mh_attn():
    """601 words: 28 host-write beats + trailing idle (29) + 11 phases
    x 52 words (1 weight-read + 9 walk + 1 activation-read + 1 switch
    + 40 drain). Straight-line (no LOOP): every phase's addresses
    differ; the item-12 LOOP without indexed addressing re-reads the
    same regions, which is exactly item 15's motivation."""
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

    # A: projections (bypass pathway)
    matmul_phase(ADDR["Wq1p"], 4, 4, 0, ADDR["X"], 4, 4, 0, PATHWAY_BYPASS)
    matmul_phase(ADDR["Wk1p"], 4, 4, 0, ADDR["X"], 4, 4, 0, PATHWAY_BYPASS)
    matmul_phase(ADDR["Wv1"], 4, 2, 0, ADDR["X"], 4, 4, 0, PATHWAY_BYPASS)
    matmul_phase(ADDR["Wq2p"], 4, 4, 0, ADDR["X"], 4, 4, 0, PATHWAY_BYPASS)
    matmul_phase(ADDR["Wk2p"], 4, 4, 0, ADDR["X"], 4, 4, 0, PATHWAY_BYPASS)
    matmul_phase(ADDR["Wv2"], 4, 2, 0, ADDR["X"], 4, 4, 0, PATHWAY_BYPASS)
    # B: S = Q~ @ K~^T with the softmax pathway -> P stays on chip
    matmul_phase(ADDR["Kt1"], 4, 4, 1, ADDR["Qt1"], 4, 4, 0,
                 PATHWAY_SOFTMAX)
    matmul_phase(ADDR["Kt2"], 4, 4, 1, ADDR["Qt2"], 4, 4, 0,
                 PATHWAY_SOFTMAX)
    # C: O_h^T = V_h^T @ P_h^T (2x4, bypass) — transposed emission so
    # the two streams stack into ([O1|O2])^T contiguously
    matmul_phase(ADDR["P1"], 4, 4, 1, ADDR["V1"], 4, 2, 1, PATHWAY_BYPASS)
    matmul_phase(ADDR["P2"], 4, 4, 1, ADDR["V2"], 4, 2, 1, PATHWAY_BYPASS)
    # D: O = [O1|O2] @ W_O — the stacked region read T=1 presents the
    # column-concat; one matmul applies the output projection
    matmul_phase(ADDR["W_O"], 4, 4, 0, ADDR["O1T"], 4, 4, 1,
                 PATHWAY_BYPASS)

    return g.prog


async def tick(dut, cycles=1):
    """Edge + 1ns settle: reads see post-edge values, drives land
    mid-cycle."""
    for _ in range(cycles):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")


@cocotb.test()
async def test_tpu_nxn_prog_mh_attn_n4(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    prog = generate_program_mh_attn()
    assert len(prog) == 601, f"program is {len(prog)} words, expected 601"
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

    # Release reset and run — ONE pulse; 601 loaded words self-pace.
    dut.rst.value = 0
    await tick(dut)
    dut.run.value = 1
    await tick(dut)
    dut.run.value = 0

    await tick(dut, len(prog) + 20)
    collector.kill()

    # ---- per-lane streams, beat-exact ----
    # Stream order per lane: Q~1, K~1, [V1 lanes 0/1 only], Q~2, K~2,
    # [V2 lanes 0/1 only], P1, P2, O1^T, O2^T (2 beats each), O.
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
        expected += col(O1T, j, 2) + col(O2T, j, 2)
        expected += col(O_RAW, j, 4)
        div = next((k for k in range(min(len(lanes[j]), len(expected)))
                    if lanes[j][k] != expected[k]), None)
        assert lanes[j] == expected, (
            f"VPU lane {j}: {len(lanes[j])} beats, expected "
            f"{len(expected)}; first divergence at {div}: got "
            f"{[f'{from_fixed(w):+.4f}' for w in lanes[j][:8]]}..., "
            f"expected {[f'{from_fixed(w):+.4f}' for w in expected[:8]]}"
            f"...")

    # ---- final UB image: all 256 words exact ----
    for a in range(256):
        got = nxn.ub_inst.ub_memory[a].value.integer & 0xFFFF
        want = GOLD[a] & 0xFFFF
        assert got == want, (
            f"mem[{a}] = {from_fixed(got):+.4f}, expected "
            f"{from_fixed(want):+.4f}")

    print(f"tpu_nxn_prog N={N} multi-head attention OK (H=2, d_model=4, "
          f"d_head=2: 601-word program, 11 phases, one run pulse; "
          f"O = [O1|O2] @ W_O via transposed-emission merge — no adder, "
          f"no new RTL; 256-word UB image exact)")
