"""Gate test for array scaling item 19c — the N=8 adaLN CAPSTONE: the
item-18b per-timestep-conditioned DiT denoiser block at N=8 (d_model
8, two heads x d_head 4, T=3 timesteps), run as ONE loaded program by
a single LOOPI control word. Written by the harness author, not the
agent — per tinytpu-loop README "the gate grows with the design".

This is the array-scaling evidence: the complete DiT inference
dataflow — LN, adaLN scale+shift, 2-head attention with the
transposed-emission merge, on-chip gates, residuals, SiLU MLP, head
projection, and the t-loop sampler recurrence — on an 8x8 systolic
array with the 236-bit instruction word, zero RTL changes from N=4.

Same 20-phase block as 18b, with N-scaled constants:

  - weight preload wait = 2N+1 = 17 cycles (item 19b proved it);
  - phase emission wait = 56 (N=8 stream + drain + margin; the final
    phase's 64 covers the last y stream);
  - the mid-phase operand read moves to switch -> tick(N-2=6) ->
    ptr-7/8: the bias-operand walk is SELF-TIMED from the read's
    execution (rd_bias_time_counter <= 0 at issue,
    unified_buffer_nxn.sv), so the read must be positioned where the
    systolic output stream actually arrives at the VPU chain head --
    the N=4-proven tick(2) is 4 cycles early at N=8 (the deeper array
    adds N-4 cycles of latency; measured: tick(1)/tick(2) pair sys
    beat k with operand k+5/k+4). The borrowed cycles come from the
    trailing idle, keeping the 79-word modulation phase length.

Layout (wbase=3648, E=1280, image 7488 -> UB_WIDTH=8192):

  Host (3648 words): I8 @0, W_qp1 @64, W_kp1 @128, W_v1 @192 (8x4),
  W_qp2 @224, W_kp2 @288, W_v2 @352 (8x4), W_O @384, W1m @448,
  W2m @512, W_head @576 (= 640 words); mod t0 @640..1023 (six 8x8
  matrices: s1m/sh1m/s2m/sh2m/G1M/G2M), zero gap, mod t1 @1920..2303,
  gap, mod t2 @3200..3583, x @3584 (= wbase-64). Mod copies spaced
  E=1280 apart so the striding ptr-7/ptr-8 reads advance one copy per
  timestep while ptr-1 host weights freeze below wbase.

  Per-iteration region (E=1280 words, base B_i = 3648 + i*1280):
  h1 @+0, h1s @+64, h1m @+128, Q~1 @+192, K~1 @+256, V1 @+320 (8x4),
  Q~2 @+352, K~2 @+416, V2 @+480 (8x4), P1 @+512, P2 @+576,
  O1^T @+640 (4x8), O2^T @+672 (4x8), t1g @+704, x2 @+768, h2 @+832,
  h2s @+896, h2m @+960, m1 @+1024, t2g @+1088, x3 @+1152, y @+1216
  (= E-64: phase 1's striding read of x @wbase-64 lands on iteration
  t-1's y — the sampler recurrence never leaves the UB).

Program: 457-word prefix (456 host beats + trailing tick) + the LOOPI
word + the 1696-word body (14 plain x 76 + 8 modulation/gate x 79) =
2154 words. PROG_DEPTH=4096. LOOPI(3, 1696, sa=1280, sw=1280,
wbase=3648).

Golden: the same hardware-exact integer stack as 18b — every helper
is dimension-generic Python; Q/K weight matrices are 8x4 padded to
8x8 with zero columns (junk columns are exactly zero and never affect
the softmax dot products).

Checks (LIVE asserts — PYTHONOPTIMIZE is empty):
  - per-lane VPU streams beat-exact across ALL THREE iterations:
    lanes 0-3 capture 168 beats/iteration (19 full-width streams +
    the two 8-beat V_h + the two 4-beat O_h^T), lanes 4-7 capture
    152 (no V_h columns);
  - the full 7488-word UB image exact everywhere — host regions
    intact, every iteration's 1280-word region exact.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from test_tpu_nxn_prog_n4 import ProgGen, to_fixed, from_fixed
from test_tpu_nxn_ic_attn_n4 import softmax_row_exact
from test_tpu_nxn_prog_mh_attn_n4 import matmul_g, to_raw, transpose
from test_tpu_nxn_prog_radd_n4 import ew_add
from test_tpu_nxn_prog_scale_n4 import ew_mul
from test_silu_parent import silu_exact
from test_tpu_nxn_prog_dit_n4 import (fxp_mul_rtl, fxp_sqrt_rtl,
                                      fxp_div_signed_rtl)

N = int(os.environ.get("TPU_NXN_PROG_N", "8"))
WORD_W = 134 + 17 * (N - 2)   # instruction width: 236 bits at N=8
CTRL = 1 << WORD_W
IDX_FLAG = 1 << 16

PATHWAY_LN = 0b0100000
PATHWAY_SOFTMAX = 0b1000000
PATHWAY_SILU = 0b10000000
PATHWAY_BIAS = 0b0001000
PATHWAY_BYPASS = 0b0000000
PTR_RESIDUAL = 7
PTR_SCALE = 8

T_STEPS = 3
PLAIN_LEN = 76              # 1 + 17 + 1 + 1 + 56
MOD_LEN = 79                # 1 + 17 + 1 + 1 + 6 + 1 + 52
BODY_LEN = 14 * PLAIN_LEN + 8 * MOD_LEN   # 1696
E_REGION = 1280
WBASE = 3648

# ---- stimulus: eighth/sixteenth multiples, exact in Q8.8, distinct
# per cell (coprime moduli — a column-constant matrix collapses the
# golden and hides addressing bugs) ----
X = [[((r * 4 + c * 5 + 1) % 9 - 4) * 0.25 for c in range(8)]
     for r in range(8)]

I8 = [[1.0 if c == r else 0.0 for c in range(8)] for r in range(8)]


def w84(off):
    """A full-rank 8x4 weight matrix (Q/K/V projections)."""
    return [[((r * 3 + c * 5 + off) % 7 - 3) * 0.125 for c in range(4)]
            for r in range(8)]


def w88(off):
    """A full-rank 8x8 weight matrix (W_O / MLP / head). Multipliers
    coprime with the modulus (gcd(5,9)=gcd(4,9)=1) — a shared factor
    makes the columns periodic and hides addressing bugs."""
    return [[((r * 5 + c * 4 + off) % 9 - 4) * 0.125 for c in range(8)]
            for r in range(8)]


def pad84(M):
    """Pad an 8x4 matrix to 8x8 with zero columns (junk columns are
    exactly zero and never affect dot products)."""
    return [row + [0.0] * 4 for row in M]


Wq1, Wk1, Wv1 = w84(1), w84(3), w84(5)
Wq2, Wk2, Wv2 = w84(2), w84(4), w84(6)
W_O, W1m, W2m, W_head = w88(0), w88(2), w88(4), w88(6)

# Per-timestep adaLN modulation — DISTINCT per t: full 8x8 elementwise
# scale/shift matrices plus row-broadcast column gates. Every formula
# uses multipliers coprime with its modulus in BOTH dimensions (a
# shared factor makes the matrix row- or column-constant — exactly
# the failure the elementwise stages must be able to catch).
MOD = []
for t in range(T_STEPS):
    s1 = [[1.0 + ((r * 3 + c * 2 + t * 2) % 5 - 2) * 0.125
           for c in range(8)] for r in range(8)]
    sh1 = [[((r * 2 + c * 3 + t * 3) % 7 - 3) * 0.0625
            for c in range(8)] for r in range(8)]
    s2 = [[1.0 + ((r * 2 + c * 3 + t * 3) % 5 - 2) * 0.125
           for c in range(8)] for r in range(8)]
    sh2 = [[((r * 3 + c * 2 + t * 5) % 7 - 3) * 0.0625
            for c in range(8)] for r in range(8)]
    g1_cols = [0.5 + ((c * 3 + t) % 5) * 0.25 for c in range(8)]
    g2_cols = [0.5 + ((c * 3 + t * 2) % 5) * 0.25 for c in range(8)]
    G1 = [list(g1_cols) for _ in range(8)]   # row-broadcast
    G2 = [list(g2_cols) for _ in range(8)]
    MOD.append((s1, sh1, s2, sh2, G1, G2))

# ---- raw (signed) forms ----
Xr = [[to_raw(v) for v in row] for row in X]
I8r = [[to_raw(v) for v in row] for row in I8]
Wq1pr = [[to_raw(v) for v in row] for row in pad84(Wq1)]
Wk1pr = [[to_raw(v) for v in row] for row in pad84(Wk1)]
Wv1r = [[to_raw(v) for v in row] for row in Wv1]
Wq2pr = [[to_raw(v) for v in row] for row in pad84(Wq2)]
Wk2pr = [[to_raw(v) for v in row] for row in pad84(Wk2)]
Wv2r = [[to_raw(v) for v in row] for row in Wv2]
W_Or = [[to_raw(v) for v in row] for row in W_O]
W1mr = [[to_raw(v) for v in row] for row in W1m]
W2mr = [[to_raw(v) for v in row] for row in W2m]
W_headr = [[to_raw(v) for v in row] for row in W_head]
MODr = [tuple([[to_raw(v) for v in row] for row in M] for M in mod_t)
        for mod_t in MOD]


# ---- exact LayerNorm model at N=8: same layernorm_group_nxn.sv RTL
# as dit_n4's ln_row_exact, but the mean/variance truncating shifts
# are by log2(N) = 3, not 2 (the N=4 derivation hardcodes >> 2; the
# RTL is N-generic — the reduction is a flat sum, so only the shift
# amount changes). softmax_row_exact needs no change: it divides by
# the exp sum, never shifts by log2(N). ----
def ln_row_exact(row):
    """One N=8 group-LN beat, bit-exact vs layernorm_group_nxn.sv.
    row: 8 signed Q8.8 raws. Returns 8 signed Q8.8 raws."""
    mean = sum(row) >> 3                     # sum_ext[18:3], truncating
    devs = [x - mean for x in row]           # 17-bit Q9.8
    sqs = [fxp_mul_rtl(d, d, 9, 8, 9, 8, 18, 8)[0] for d in devs]
    var_eps = (sum(sqs) >> 3) + 16           # LN_EPS = 1/16
    std, _ = fxp_sqrt_rtl(var_eps, 18, 8, 8, 8)
    return [fxp_div_signed_rtl(d, std) for d in devs]


# ---- golden: the whole 20-phase block x T_STEPS, raw Q8.8 ----
def adaln_iteration(x_in, mod_t):
    """One adaLN-conditioned denoiser block on an 8x8 raw matrix.
    mod_t = (s1m, sh1m, s2m, sh2m, G1M, G2M) raw for this timestep.
    Returns the phase outputs in stream order."""
    (s1m, sh1m, s2m, sh2m, G1M, G2M) = mod_t
    h1 = [ln_row_exact(row) for row in x_in]
    h1s = ew_mul(h1, s1m)                # adaLN scale (p1a)
    h1m = ew_add(h1s, sh1m)              # adaLN shift (p1b)
    Qt1 = matmul_g(h1m, Wq1pr)
    Kt1 = matmul_g(h1m, Wk1pr)
    V1 = matmul_g(h1m, Wv1r)             # 8x4
    Qt2 = matmul_g(h1m, Wq2pr)
    Kt2 = matmul_g(h1m, Wk2pr)
    V2 = matmul_g(h1m, Wv2r)             # 8x4
    P1 = [softmax_row_exact(row)
          for row in matmul_g(Qt1, transpose(Kt1))]
    P2 = [softmax_row_exact(row)
          for row in matmul_g(Qt2, transpose(Kt2))]
    O1T = matmul_g(transpose(V1), transpose(P1))   # 4x8
    O2T = matmul_g(transpose(V2), transpose(P2))   # 4x8
    Cm = [[O1T[0][r], O1T[1][r], O1T[2][r], O1T[3][r],
           O2T[0][r], O2T[1][r], O2T[2][r], O2T[3][r]] for r in range(8)]
    t1g = ew_mul(matmul_g(Cm, W_Or), G1M)          # attention gate
    x2 = ew_add(t1g, x_in)               # residual: the phase input x
    h2 = [ln_row_exact(row) for row in x2]
    h2s = ew_mul(h2, s2m)                # adaLN scale (p13a)
    h2m = ew_add(h2s, sh2m)              # adaLN shift (p13b)
    m1 = [[silu_exact(v) for v in row] for row in matmul_g(h2m, W1mr)]
    t2g = ew_mul(matmul_g(m1, W2mr), G2M)          # MLP gate
    x3 = ew_add(t2g, x2)                 # residual: x2
    y = matmul_g(x3, W_headr)
    return (h1, h1s, h1m, Qt1, Kt1, V1, Qt2, Kt2, V2, P1, P2,
            O1T, O2T, t1g, x2, h2, h2s, h2m, m1, t2g, x3, y)


ITERS = []
_x = Xr
for _t in range(T_STEPS):
    _out = adaln_iteration(_x, MODr[_t])
    ITERS.append(_out)
    _x = _out[-1]                        # y_{t} is t+1's input

# ---- UB image (7488 words) ----
ADDR = {"I8": 0, "Wqp1": 64, "Wkp1": 128, "Wv1": 192,
        "Wqp2": 224, "Wkp2": 288, "Wv2": 352, "W_O": 384,
        "W1m": 448, "W2m": 512, "W_head": 576, "x": WBASE - 64}
MOD_BASE = [640 + t * E_REGION for t in range(T_STEPS)]  # 640/1920/3200
MOFF = {"s1m": 0, "sh1m": 64, "s2m": 128, "sh2m": 192,
        "G1M": 256, "G2M": 320}
OFF = {"h1": 0, "h1s": 64, "h1m": 128, "Qt1": 192, "Kt1": 256,
       "V1": 320, "Qt2": 352, "Kt2": 416, "V2": 480,
       "P1": 512, "P2": 576, "O1T": 640, "O2T": 672,
       "t1g": 704, "x2": 768, "h2": 832, "h2s": 896, "h2m": 960,
       "m1": 1024, "t2g": 1088, "x3": 1152, "y": 1216}
GOLD = {a: 0 for a in range(WBASE + T_STEPS * E_REGION)}


def put(base, M, rows, cols):
    for r in range(rows):
        for c in range(cols):
            GOLD[base + cols * r + c] = M[r][c]


put(ADDR["I8"], I8r, 8, 8)
put(ADDR["Wqp1"], Wq1pr, 8, 8)
put(ADDR["Wkp1"], Wk1pr, 8, 8)
put(ADDR["Wv1"], Wv1r, 8, 4)
put(ADDR["Wqp2"], Wq2pr, 8, 8)
put(ADDR["Wkp2"], Wk2pr, 8, 8)
put(ADDR["Wv2"], Wv2r, 8, 4)
put(ADDR["W_O"], W_Or, 8, 8)
put(ADDR["W1m"], W1mr, 8, 8)
put(ADDR["W2m"], W2mr, 8, 8)
put(ADDR["W_head"], W_headr, 8, 8)
put(ADDR["x"], Xr, 8, 8)
for _t in range(T_STEPS):
    (_s1m, _sh1m, _s2m, _sh2m, _G1M, _G2M) = MODr[_t]
    for _name, _M in (("s1m", _s1m), ("sh1m", _sh1m), ("s2m", _s2m),
                      ("sh2m", _sh2m), ("G1M", _G1M), ("G2M", _G2M)):
        put(MOD_BASE[_t] + MOFF[_name], _M, 8, 8)
for _t in range(T_STEPS):
    _base = WBASE + _t * E_REGION
    (_h1, _h1s, _h1m, _Qt1, _Kt1, _V1, _Qt2, _Kt2, _V2, _P1, _P2,
     _O1T, _O2T, _t1g, _x2, _h2, _h2s, _h2m, _m1, _t2g, _x3,
     _y) = ITERS[_t]
    for _name, _M, _rows, _cols in (
            ("h1", _h1, 8, 8), ("h1s", _h1s, 8, 8), ("h1m", _h1m, 8, 8),
            ("Qt1", _Qt1, 8, 8), ("Kt1", _Kt1, 8, 8), ("V1", _V1, 8, 4),
            ("Qt2", _Qt2, 8, 8), ("Kt2", _Kt2, 8, 8), ("V2", _V2, 8, 4),
            ("P1", _P1, 8, 8), ("P2", _P2, 8, 8), ("O1T", _O1T, 4, 8),
            ("O2T", _O2T, 4, 8), ("t1g", _t1g, 8, 8), ("x2", _x2, 8, 8),
            ("h2", _h2, 8, 8), ("h2s", _h2s, 8, 8), ("h2m", _h2m, 8, 8),
            ("m1", _m1, 8, 8), ("t2g", _t2g, 8, 8), ("x3", _x3, 8, 8),
            ("y", _y, 8, 8)):
        put(_base + OFF[_name], _M, _rows, _cols)
IMG_WORDS = WBASE + T_STEPS * E_REGION
assert len(GOLD) == IMG_WORDS, \
    f"image is {len(GOLD)} words, expected {IMG_WORDS}"


def loopxl_word(count, length, stride_a, stride_w, wbase):
    """LOOPI control word with the 17b1 15-bit length (low 8 bits at
    [7:0], high 7 at [23:17]) and the 17a wbase field [71:56]."""
    assert 0 < length < (1 << 15)
    return (CTRL | (count << 8) | (length & 0xFF) | IDX_FLAG
            | ((length >> 8) << 17)
            | (stride_a << 24) | (stride_w << 40) | (wbase << 56))


def generate_program_adaln_n8():
    """2154 words: 457-word host prefix + LOOPI(3, 1696, sa=1280,
    sw=1280, wbase=3648) + the 1696-word body (20 phases: 14 plain
    x 76 + 8 modulation/gate x 79). Body addresses are the iteration-0
    absolute addresses; the strides advance every region read AND
    every modulation-block read by E=1280 per iteration while host
    weight reads (< wbase, ptr-1) freeze."""
    g = ProgGen()
    g.idle()
    # Sparse fill: the mod blocks and x land at their absolute
    # addresses; gap words are zeros (written, never read).
    sparse = {a: 0 for a in range(WBASE)}
    pos = 0
    for M in (I8, pad84(Wq1), pad84(Wk1), Wv1,
              pad84(Wq2), pad84(Wk2), Wv2, W_O, W1m, W2m, W_head):
        for row in M:
            for v in row:
                sparse[pos] = to_fixed(v)
                pos += 1
    assert pos == 640
    for t in range(T_STEPS):
        for name, M in zip(("s1m", "sh1m", "s2m", "sh2m", "G1M", "G2M"),
                           MOD[t]):
            ws = [to_fixed(v) for row in M for v in row]
            for k, w in enumerate(ws):
                sparse[MOD_BASE[t] + MOFF[name] + k] = w
    xs = [to_fixed(v) for row in X for v in row]
    for k, w in enumerate(xs):
        sparse[ADDR["x"] + k] = w
    host_words = [sparse[a] for a in range(WBASE)]
    assert len(host_words) == WBASE and len(host_words) % N == 0
    for b in range(len(host_words) // N):
        g.write_beat([host_words[N * b + (N - 1 - i)] for i in range(N)])
    g.tick()  # trailing idle cycle after the last beat
    assert len(g.prog) == 457, \
        f"prefix is {len(g.prog)} words, expected 457"

    def matmul_phase(w_addr, w_rows, w_cols, w_T, a_addr, a_rows, a_cols,
                     a_T, pathway):
        g.issue_read(ptr=1, addr=w_addr, rows=w_rows, cols=w_cols,
                     transpose=w_T)
        g.tick(17)                       # weight preload: 2N+1
        g.issue_read(ptr=0, addr=a_addr, rows=a_rows, cols=a_cols,
                     transpose=a_T, pathway=pathway)
        g.switch_pulse()
        g.tick(56)

    def scale_phase(w_addr, w_T, a_addr, a_T, s_addr):
        """(A @ W) . S: matmul + a mid-phase ptr-8 scale read (the
        item-18a contract). The bias-operand walk is SELF-TIMED from
        the read's execution (rd_bias_time_counter <= 0 at issue, per
        unified_buffer_nxn.sv), so the operand read sits at
        switch + (N-2): the systolic stream reaches the VPU chain head
        N-4 cycles later than at N=4, and the operand window must
        coincide with it (tick(1) pairs sys beat k with operand k+5)."""
        g.issue_read(ptr=1, addr=w_addr, rows=8, cols=8, transpose=w_T)
        g.tick(17)
        g.issue_read(ptr=0, addr=a_addr, rows=8, cols=8, transpose=a_T,
                     pathway=PATHWAY_BYPASS)
        g.switch_pulse()
        g.tick(6)
        g.issue_read(ptr=PTR_SCALE, addr=s_addr, rows=8, cols=8)
        g.tick(52)

    def radd_phase(w_addr, w_T, a_addr, a_T, r_addr):
        g.issue_read(ptr=1, addr=w_addr, rows=8, cols=8, transpose=w_T)
        g.tick(17)
        g.issue_read(ptr=0, addr=a_addr, rows=8, cols=8, transpose=a_T,
                     pathway=PATHWAY_BIAS)
        g.switch_pulse()
        g.tick(6)
        g.issue_read(ptr=PTR_RESIDUAL, addr=r_addr, rows=8, cols=8)
        g.tick(52)

    g.prog.append(loopxl_word(T_STEPS, BODY_LEN, E_REGION, E_REGION,
                              WBASE))
    _body_at = len(g.prog)
    B = WBASE  # iteration-0 region base; strides do the rest
    M0 = MOD_BASE[0]

    # p1: h1 = LN(x) — identity matmul + the LN group stage
    matmul_phase(ADDR["I8"], 8, 8, 0, ADDR["x"], 8, 8, 0, PATHWAY_LN)
    # p1a/p1b: the adaLN modulation of h1 (scale then shift)
    scale_phase(ADDR["I8"], 0, B + OFF["h1"], 0, M0 + MOFF["s1m"])
    radd_phase(ADDR["I8"], 0, B + OFF["h1s"], 0, M0 + MOFF["sh1m"])
    # p2-7: per-head Q/K/V projections of h1m (bypass)
    matmul_phase(ADDR["Wqp1"], 8, 8, 0, B + OFF["h1m"], 8, 8, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wkp1"], 8, 8, 0, B + OFF["h1m"], 8, 8, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wv1"], 8, 4, 0, B + OFF["h1m"], 8, 8, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wqp2"], 8, 8, 0, B + OFF["h1m"], 8, 8, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wkp2"], 8, 8, 0, B + OFF["h1m"], 8, 8, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wv2"], 8, 4, 0, B + OFF["h1m"], 8, 8, 0,
                 PATHWAY_BYPASS)
    # p8/9: P_h = softmax(Q~_h @ K~_h^T)
    matmul_phase(B + OFF["Kt1"], 8, 8, 1, B + OFF["Qt1"], 8, 8, 0,
                 PATHWAY_SOFTMAX)
    matmul_phase(B + OFF["Kt2"], 8, 8, 1, B + OFF["Qt2"], 8, 8, 0,
                 PATHWAY_SOFTMAX)
    # p10/11: O_h^T = V_h^T @ P_h^T (4x8; the item-14 merge)
    matmul_phase(B + OFF["P1"], 8, 8, 1, B + OFF["V1"], 8, 4, 1,
                 PATHWAY_BYPASS)
    matmul_phase(B + OFF["P2"], 8, 8, 1, B + OFF["V2"], 8, 4, 1,
                 PATHWAY_BYPASS)
    # p12a: t1g = ([O1|O2] @ W_O) . G1M — the T-read of the contiguous
    # O-stack presents the concat; the gate rides the ptr-8 channel
    scale_phase(ADDR["W_O"], 0, B + OFF["O1T"], 1, M0 + MOFF["G1M"])
    # p12b: x2 = t1g + x — residual read (ptr-7) of the phase input x
    radd_phase(ADDR["I8"], 0, B + OFF["t1g"], 0, ADDR["x"])
    # p13: h2 = LN(x2)
    matmul_phase(ADDR["I8"], 8, 8, 0, B + OFF["x2"], 8, 8, 0, PATHWAY_LN)
    # p13a/p13b: the adaLN modulation of h2
    scale_phase(ADDR["I8"], 0, B + OFF["h2"], 0, M0 + MOFF["s2m"])
    radd_phase(ADDR["I8"], 0, B + OFF["h2s"], 0, M0 + MOFF["sh2m"])
    # p14: m1 = SiLU(h2m @ W1m)
    matmul_phase(ADDR["W1m"], 8, 8, 0, B + OFF["h2m"], 8, 8, 0,
                 PATHWAY_SILU)
    # p15a: t2g = (m1 @ W2m) . G2M
    scale_phase(ADDR["W2m"], 0, B + OFF["m1"], 0, M0 + MOFF["G2M"])
    # p15b: x3 = t2g + x2
    radd_phase(ADDR["I8"], 0, B + OFF["t2g"], 0, B + OFF["x2"])
    # p16: y = x3 @ W_head (the final stream of the iteration)
    matmul_phase(ADDR["W_head"], 8, 8, 0, B + OFF["x3"], 8, 8, 0,
                 PATHWAY_BYPASS)

    assert len(g.prog) - _body_at == BODY_LEN, (
        f"body is {len(g.prog) - _body_at} words, expected {BODY_LEN}")
    return g.prog


async def tick(dut, cycles=1):
    """Edge + 1ns settle: reads see post-edge values, drives land
    mid-cycle."""
    for _ in range(cycles):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")


@cocotb.test()
async def test_tpu_nxn_prog_adaln_n8(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    prog = generate_program_adaln_n8()
    assert len(prog) == 2154, \
        f"program is {len(prog)} words, expected 2154"
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

    # Release reset and run — ONE pulse; the LOOPI word iterates the
    # 1696-word body T=3 times.
    dut.rst.value = 0
    await tick(dut)
    dut.run.value = 1
    await tick(dut)
    dut.run.value = 0

    # Emission: 2154 loaded words + 2 extra body passes (2 x 1696);
    # the last body's 56-cycle idle + 40 margin covers the final y
    # stream.
    await tick(dut, 2154 + 2 * BODY_LEN + 40)
    collector.kill()

    # ---- per-lane streams, beat-exact across all three iterations ----
    def col(M, j, rows):
        return [M[r][j] & 0xFFFF for r in range(rows)]

    for j in range(N):
        expected = []
        for t in range(T_STEPS):
            (h1, h1s, h1m, Qt1, Kt1, V1, Qt2, Kt2, V2, P1, P2,
             O1T, O2T, t1g, x2, h2, h2s, h2m, m1, t2g, x3, y) = ITERS[t]
            for M in (h1, h1s, h1m, Qt1, Kt1):
                expected += col(M, j, 8)
            if j < 4:
                expected += col(V1, j, 8)
            for M in (Qt2, Kt2):
                expected += col(M, j, 8)
            if j < 4:
                expected += col(V2, j, 8)
            for M in (P1, P2):
                expected += col(M, j, 8)
            expected += col(O1T, j, 4) + col(O2T, j, 4)
            for M in (t1g, x2, h2, h2s, h2m, m1, t2g, x3, y):
                expected += col(M, j, 8)
        divs = [k for k in range(min(len(lanes[j]), len(expected)))
                if lanes[j][k] != expected[k]]
        div = divs[0] if divs else None
        assert lanes[j] == expected, (
            f"VPU lane {j}: {len(lanes[j])} beats, expected "
            f"{len(expected)}; {len(divs)} divergent beats "
            f"{divs[:24]}; first at beat {div} "
            f"(iteration {div // 168 if j < 4 else div // 152}): got "
            f"{[f'{from_fixed(w):+.4f}' for w in lanes[j][max(0, div-2):div+3]]}"
            f", expected "
            f"{[f'{from_fixed(w):+.4f}' for w in expected[max(0, div-2):div+3]]}"
            f" (beats {max(0, div-2)}..{div+2})")

    # ---- final UB image: all 7488 words exact ----
    for a in range(IMG_WORDS):
        got = nxn.ub_inst.ub_memory[a].value.integer & 0xFFFF
        want = GOLD[a] & 0xFFFF
        assert got == want, (
            f"mem[{a}] = {from_fixed(got):+.4f}, expected "
            f"{from_fixed(want):+.4f}")

    print(f"tpu_nxn_prog N={N} adaLN CAPSTONE at scale OK (2154-word "
          f"program: ONE 1696-word adaLN-conditioned denoiser body — "
          f"LN, scale+shift, 2-head attention, gate+residual, LN, "
          f"scale+shift, SiLU MLP, gate+residual, head projection — "
          f"looped T={T_STEPS} by LOOPI(3, 1696, 1280, 1280, "
          f"wbase=3648) with DISTINCT per-timestep modulation "
          f"striding through the host image; {IMG_WORDS}-word UB "
          f"image exact)")
