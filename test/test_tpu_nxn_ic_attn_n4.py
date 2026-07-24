"""Gate test for roadmap item 11: an ATTENTION-HEAD COMPOSITE through
the instruction port of tpu_nxn_ic — matmul -> softmax -> matmul as one
choreography, no host rewrites between phases. Written by the harness
author, not the agent — per tinytpu-loop README "the gate grows with
the design".

No new RTL: this reuses src/tpu_nxn_ic.sv (item 8b). All three ops
already exist as gated leaves (array matmul, item 5; VPU softmax group
stage, item 7a) — the composite is the new thing: the softmax output
P stays ON CHIP (it lands in the UB at the write pointer) and is read
straight back as the activation stream of the second matmul.

    phase 1:  S = Q @ K^T   (weight walk K@16 T=1, activations Q@0 T=0)
              P = softmax(S) per row   (pathway bit 6 only)
              -> P stream lands @48-63 (48-word host image)
    phase 2:  O = P @ V     (weight walk V_T@32 T=1 -> array holds V,
                             activations P@48 T=0, bypass pathway)
              -> O stream lands @64-79

The golden is HARDWARE-EXACT integer arithmetic, not float-plus-
tolerance (derived from src/fixedpoint.sv, verified against the RTL
semantics line by line):
  fxp_mul Q8.8: clamp16((a*b + 0x80) >> 8)   -- fxp_zoom ROUND=1 rounds
    on the top dropped bit; the all-ones overflow guard coincides with
    clamping, so the plain formula is exact for every input.
  fxp_add Q8.8: clamp16(a + b)               -- WOF==WIF: no fractional
    rounding, only saturation.
  PE MAC: every product AND every accumulation step rounds/saturates
    (pe.sv: fxp_mul + fxp_add per step), so C[i][j] folds k = 0..3
    through both, in order.
  exp LUT: imported from test_softmax_group_nxn (exact integer model).
  fxp_div ROUND=1 (positive operands): q = floor(256*e/t); round up
    iff 2*rem > t -- note STRICT > (round-half-DOWN on exact ties),
    from the RTL's `acct - divd < divd - acc` comparison.

Stimulus: quarter-multiple Q/K/V (exact in Q8.8) sized so S spans
[-0.63, +0.81] -- the exp LUT interpolator and the divider rounding
are both exercised, and P@V products hit sub-LSB values that force
the PE's per-step rounding to bite in phase 2.

Checks (LIVE asserts -- PYTHONOPTIMIZE is empty):
  - per-lane VPU streams capture exactly 8 beats: P columns then O
    columns, beat-exact against the integer golden;
  - the full 80-word UB image: Q/K/V_T untouched, P @48-63 and O
    @64-79 exact everywhere (the append-at-wr_ptr placement itself is
    part of the check).
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

# Exact exp-LUT model from the 7a softmax gate test; Host + Q8.8
# helpers from the 9a test (leak/inv2b ride along in the hold word but
# are inert on softmax/bypass pathways).
from test_softmax_group_nxn import exp_lut_exact
from test_tpu_nxn_ic_train2_n4 import Host, to_fixed, from_fixed, tick

N = int(os.environ.get("TPU_NXN_IC_N", "4"))
FRAC = 8

# ---- stimulus: quarter multiples, exact in Q8.8 ----
Q = [[0.50, -0.25, 0.75, 0.00],
     [-0.50, 0.25, 0.00, 0.50],
     [0.25, 0.50, -0.75, 0.25],
     [0.00, -0.50, 0.25, -0.25]]
K = [[0.25, 0.50, -0.25, 0.75],
     [-0.75, 0.00, 0.50, -0.25],
     [0.50, -0.50, 0.25, 0.00],
     [0.00, 0.75, -0.50, 0.25]]
V = [[0.50, 0.25, -0.50, 0.00],
     [-0.25, 0.75, 0.00, 0.50],
     [0.00, -0.25, 0.50, -0.75],
     [0.75, 0.00, -0.25, 0.25]]

# softmax-only pathway: bit 6 of |sm(6)|ln(5)|gelu(4)|bias(3)|lr(2)|
# loss(1)|lr_d(0)|; phase 2 bypasses every stage.
PATHWAY_SOFTMAX = 0b1000000
PATHWAY_BYPASS = 0b0000000


# ---- hardware-exact fixed-point models (src/fixedpoint.sv) ----
def clamp16(v):
    return max(-32768, min(32767, v))


def fxp_mul(a, b):
    """Q8.8 x Q8.8 -> Q8.8, fxp_zoom ROUND=1: round on the top dropped
    bit; the increment guard and the saturation together are exactly
    clamp16 of the naive formula."""
    return clamp16((a * b + 0x80) >> 8)


def fxp_add(a, b):
    """Q8.8 + Q8.8 -> Q8.8: WOF==WIF so no fractional rounding, only
    the WOI<WII saturation."""
    return clamp16(a + b)


def fxp_div_pos(e, t):
    """fxp_div ROUND=1 on positive raws: q = floor(256*e/t), round up
    iff 2*rem > t (strict -- round-half-down on exact ties)."""
    q = (e * 256) // t
    rem = e * 256 - q * t
    if q != 0xFFFF and 2 * rem > t:
        q += 1
    return q


def matmul_raw(A, B):
    """C[i][j] = sum_k A[i][k]*B[k][j] with the PE's per-step rounding:
    each product is fxp_mul-rounded and each accumulation step is an
    fxp_add (saturating), folded in k order."""
    C = []
    for i in range(4):
        row = []
        for j in range(4):
            acc = 0
            for k in range(4):
                acc = fxp_add(acc, fxp_mul(A[i][k], B[k][j]))
            row.append(acc)
        C.append(row)
    return C


def softmax_row_exact(row):
    """One group-softmax beat: max, exact exp LUT, exact division."""
    m = max(row)
    exps = [exp_lut_exact(m - r) for r in row]
    total = sum(exps)
    return [fxp_div_pos(e, total) for e in exps]


def to_raw(v):
    """Signed raw Q8.8 integer (to_fixed returns the 16-bit two's
    complement encoding)."""
    w = to_fixed(v)
    return w - 0x10000 if w & 0x8000 else w


# ---- golden: S = Q@K^T, P = softmax(S), O = P@V (all raw Q8.8) ----
Qr = [[to_raw(v) for v in row] for row in Q]
Kr = [[to_raw(v) for v in row] for row in K]
Vr = [[to_raw(v) for v in row] for row in V]

KTr = [[Kr[j][k] for j in range(4)] for k in range(4)]  # S = Q @ K^T
S_RAW = matmul_raw(Qr, KTr)
P_RAW = [softmax_row_exact(row) for row in S_RAW]
O_RAW = matmul_raw(P_RAW, Vr)  # array holds V (V_T read T=1)

# UB image: Q@0, K@16, V_T@32 (host, 48 words), P@48, O@64 (streams).
V_T = [[V[c][r] for c in range(4)] for r in range(4)]
GOLD = {}
for r in range(4):
    for c in range(4):
        GOLD[0 + 4 * r + c] = Qr[r][c]
        GOLD[16 + 4 * r + c] = Kr[r][c]
        GOLD[32 + 4 * r + c] = to_raw(V_T[r][c])
        GOLD[48 + 4 * r + c] = P_RAW[r][c]  # streams append at the
        GOLD[64 + 4 * r + c] = O_RAW[r][c]  # UB write pointer


@cocotb.test()
async def test_tpu_nxn_ic_attn_n4(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    host = Host(dut)
    nxn = dut.tpu_nxn_ic_inst.tpu_nxn_inst

    dut.rst.value = 1
    await host.idle()
    dut.learning_rate_in.value = to_fixed(0.5)  # inert: no lr_d stage
    await tick(dut, 2)
    dut.rst.value = 0
    await tick(dut)

    lanes = [[] for _ in range(N)]

    async def collect():
        while True:
            await RisingEdge(dut.clk)
            for i in range(N):
                if nxn.ub_wr_valid_in[i].value.integer:
                    lanes[i].append(nxn.ub_wr_data_in[i].value.integer
                                    & 0xFFFF)

    collector = cocotb.start_soon(collect())

    # Host image: Q @ 0, K @ 16, V_T @ 32 (48 words) — loaded once;
    # both phases read it, nothing is rewritten by the host.
    words = ([to_fixed(v) for row in Q for v in row]
             + [to_fixed(v) for row in K for v in row]
             + [to_fixed(v) for row in V_T for v in row])
    assert len(words) % N == 0
    for b in range(len(words) // N):
        await host.write_beat([words[N * b + (N - 1 - i)]
                               for i in range(N)])
    await tick(dut)

    # ---- phase 1: S = Q@K^T, softmax -> P lands @48-63 ----
    await host.issue_read(ptr=1, addr=16, rows=4, cols=4, transpose=1)
    await tick(dut, 9)  # N=4 weight walk = R+C-1 = 7 cycles + margin

    await host.issue_read(ptr=0, addr=0, rows=4, cols=4,
                          pathway=PATHWAY_SOFTMAX)
    await host.switch_pulse()

    await tick(dut, 40)  # S computes, softmax beats stream P to UB

    # ---- phase 2: O = P@V, bypass -> O lands @64-79 ----
    # The weight walk reads V_T@32 with transpose=1, so the array
    # holds (V_T)^T = V; the activations are the on-chip P @48.
    await host.issue_read(ptr=1, addr=32, rows=4, cols=4, transpose=1)
    await tick(dut, 9)

    await host.issue_read(ptr=0, addr=48, rows=4, cols=4, transpose=0,
                          pathway=PATHWAY_BYPASS)
    await host.switch_pulse()

    await tick(dut, 40)  # O computes, bypass beats stream O to UB

    collector.kill()

    # ---- per-lane streams: P then O (8 beats per lane) ----
    for j in range(N):
        for k, (name, mat) in enumerate([("P", P_RAW), ("O", O_RAW)]):
            got = lanes[j][4 * k:4 * k + 4]
            expected = [mat[r][j] & 0xFFFF for r in range(4)]
            assert got == expected, (
                f"VPU lane {j} {name}: got "
                f"{[f'{from_fixed(w):+.4f}' for w in got]}, expected "
                f"{[f'{from_fixed(w):+.4f}' for w in expected]} "
                f"(column {j})")
        assert len(lanes[j]) == 8, (
            f"VPU lane {j}: expected exactly 8 beats (P,O), "
            f"got {len(lanes[j])}")

    # ---- final UB image: Q/K/V_T intact, P/O appended, exact ----
    for base, count, name in [(0, 16, "Q"), (16, 16, "K"),
                              (32, 16, "V_T"), (48, 16, "P"),
                              (64, 16, "O")]:
        for a in range(base, base + count):
            got = nxn.ub_inst.ub_memory[a].value.integer & 0xFFFF
            want = GOLD[a] & 0xFFFF
            assert got == want, (
                f"{name} region: mem[{a}] = {from_fixed(got):+.4f}, "
                f"expected {from_fixed(want):+.4f}")

    print(f"tpu_nxn_ic N={N} attention-head composite OK "
          f"(S=Q@K^T -> softmax -> O=P@V; P stays on chip @48, O @64; "
          f"hardware-exact integer golden)")
