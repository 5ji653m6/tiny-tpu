"""Gate test for array scaling item 21c — the N=16 adaLN CAPSTONE: the
item-19c per-timestep-conditioned DiT denoiser block at N=16 (d_model
16, two heads x d_head 8, T=3 timesteps), run as ONE loaded program by
a single LOOPI control word. Written by the harness author, not the
agent — per tinytpu-loop README "the gate grows with the design".

This is the array-scaling evidence at N=16: the complete DiT inference
dataflow — LN, adaLN scale+shift, 2-head attention with the
transposed-emission merge, on-chip gates, residuals, SiLU MLP, head
projection, and the t-loop sampler recurrence — on a 16x16 systolic
array with the 372-bit instruction word, zero RTL changes from N=4.

Same 20-phase block as 19c (N=8) and 18b (N=4), with N-scaled constants:

  - weight preload wait = 2N+1 = 33 cycles (item 21b proved it);
  - phase emission wait = 7N = 112 (N=16 stream + drain + margin;
    at N=8 this was 56 = 7*8, proportional to the systolic + VPU
    latency which scales linearly with N);
  - mid-phase operand read at switch + (N-2) = switch + tick(14): the
    self-timed bias walk must coincide with the systolic stream at the
    VPU chain head;
  - mod drain = 13N/2 = 104 (proportional to the VPU scale/radd
    processing of NxN outputs through the N-wide VPU);
  - the group-LN golden uses >>log2(N) = >>4 for the mean/variance
    (the N=4 derivation uses >>2, N=8 uses >>3; the RTL is N-generic).

Layout (wbase=14592, E=5120, image 29952 -> UB_WIDTH=32768):

  Host (14592 words): I16 @0, Wqp1 @256, Wkp1 @512, Wv1 @768 (16x8),
  Wqp2 @896, Wkp2 @1152, Wv2 @1408 (16x8), W_O @1536, W1m @1792,
  W2m @2048, W_head @2304 (= 2560 words); mod t0 @2560..4095 (six
  16x16 matrices: s1m/sh1m/s2m/sh2m/G1M/G2M), zero gap, mod t1
  @7680..9215, gap, mod t2 @12800..14335, x @14336 (= wbase-256).
  Mod copies spaced E=5120 apart so the striding ptr-7/ptr-8 reads
  advance one copy per timestep while ptr-1 host weights freeze below
  wbase.

  Per-iteration region (E=5120 words, base B_i = 14592 + i*5120):
  h1 @+0, h1s @+256, h1m @+512, Q~1 @+768, K~1 @+1024, V1 @+1280
  (16x8), Q~2 @+1408, K~2 @+1664, V2 @+1792 (16x8), P1 @+2048,
  P2 @+2304, O1^T @+2560 (8x16), O2^T @+2688 (8x16), t1g @+2816,
  x2 @+3072, h2 @+3328, h2s @+3584, h2m @+3840, m1 @+4096,
  t2g @+4352, x3 @+4608, y @+4864 (= E-256: phase 1's striding read
  of x @wbase-256 lands on iteration t-1's y — the sampler recurrence
  never leaves the UB).

Program: 913-word prefix (912 host beats + trailing tick) + the LOOPI
word + the 3312-word body (14 plain x 148 + 8 modulation/gate x 155) =
4226 words. PROG_DEPTH=8192. LOOPI(3, 3312, sa=5120, sw=5120,
wbase=14592).

Golden: the same hardware-exact integer stack as 19c — every helper
is dimension-generic Python; Q/K weight matrices are 16x8 padded to
16x16 with zero columns (junk columns are exactly zero and never affect
the softmax dot products).

Checks (LIVE asserts — PYTHONOPTIMIZE is empty):
  - per-lane VPU streams beat-exact across ALL THREE iterations:
    lanes 0-7 capture 336 beats/iteration (18 full-width streams +
    the two 16-beat V_h + the two 8-beat O_h^T), lanes 8-15 capture
    304 (no V_h columns);
  - the full 29952-word UB image exact everywhere — host regions
    intact, every iteration's 5120-word region exact.
"""

import math
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

N = int(os.environ.get("TPU_NXN_PROG_N", "16"))
WORD_W = 134 + 17 * (N - 2)   # instruction width: 372 bits at N=16
CTRL = 1 << WORD_W
IDX_FLAG = 1 << 16
LOG2N = int(math.log2(N))     # 4 at N=16

PATHWAY_LN = 0b0100000
PATHWAY_SOFTMAX = 0b1000000
PATHWAY_SILU = 0b10000000
PATHWAY_BIAS = 0b0001000
PATHWAY_BYPASS = 0b0000000
PTR_RESIDUAL = 7
PTR_SCALE = 8

T_STEPS = 3
DH = N // 2                    # d_head = 8 at N=16
PLAIN_LEN = 1 + (2 * N + 1) + 1 + 1 + 7 * N      # 148
MOD_LEN = 1 + (2 * N + 1) + 1 + 1 + (N - 2) + 1 + 13 * N // 2   # 155
BODY_LEN = 14 * PLAIN_LEN + 8 * MOD_LEN           # 3312

# ---- UB layout: weight host region ----
# 9 full NxN + 2 Nx(N/2) V matrices: I_N, Wqp1, Wkp1, Wqp2, Wkp2,
# W_O, W1m, W2m, W_head (9 full) + Wv1, Wv2 (2 half)
_WHOST = 9 * N * N + 2 * N * DH    # 2560 at N=16
assert _WHOST == 2560

# ---- UB layout: per-iteration region ----
# 22 phase outputs; each occupies NxN except V1/V2 (Nx(DH)) and
# O1T/O2T ((DH)xN). The offsets must tile exactly E_REGION.
_N2 = N * N                        # 256
_NDH = N * DH                      # 128
OFF = {}
_pos = 0
for _k, _s in (
        ("h1", _N2), ("h1s", _N2), ("h1m", _N2),
        ("Qt1", _N2), ("Kt1", _N2), ("V1", _NDH),
        ("Qt2", _N2), ("Kt2", _N2), ("V2", _NDH),
        ("P1", _N2), ("P2", _N2),
        ("O1T", _NDH), ("O2T", _NDH),
        ("t1g", _N2), ("x2", _N2),
        ("h2", _N2), ("h2s", _N2), ("h2m", _N2),
        ("m1", _N2), ("t2g", _N2), ("x3", _N2), ("y", _N2)):
    OFF[_k] = _pos
    _pos += _s
# 18 full tiles + 4 half tiles = 18*256 + 4*128 = 5120
E_REGION = 18 * _N2 + 4 * _NDH
assert E_REGION == 5120
assert OFF["y"] + _N2 == E_REGION, \
    f"y ends at {OFF['y'] + _N2}, expected {E_REGION}"

# ---- UB layout: modulation matrices per timestep ----
MOFF = {"s1m": 0, "sh1m": _N2, "s2m": 2 * _N2,
        "sh2m": 3 * _N2, "G1M": 4 * _N2, "G2M": 5 * _N2}
MOD_BASE = [_WHOST + t * E_REGION for t in range(T_STEPS)]
# [2560, 7680, 12800]

# x sits right after the last mod block
_ADDR_X = MOD_BASE[-1] + 6 * _N2    # 14336
WBASE = _ADDR_X + _N2               # 14592
assert WBASE == 14592

IMG_WORDS = WBASE + T_STEPS * E_REGION    # 29952
assert IMG_WORDS == 29952

# ---- stimulus: deterministic, distinct per cell, exact in Q8.8 ----
# (formulas chosen so W is NOT column-constant and no X row sums to
# zero — a rank-1 stimulus makes C rows collapse to zero and hides
# addressing bugs; coprime moduli in BOTH dims so elementwise stages
# catch any broadcast/routing fault)
X = [[((r * 4 + c * 5 + 1) % 9 - 4) * 0.25 for c in range(N)]
     for r in range(N)]

I_N = [[1.0 if c == r else 0.0 for c in range(N)] for r in range(N)]


def w_n_half(off):
    """A full-rank Nx(N/2) weight matrix (Q/K/V projections)."""
    return [[((r * 3 + c * 5 + off) % 7 - 3) * 0.125 for c in range(DH)]
            for r in range(N)]


def w_nn(off):
    """A full-rank NxN weight matrix (W_O / MLP / head). Multipliers
    coprime with the modulus (gcd(5,9)=gcd(4,9)=1) — a shared factor
    makes the columns periodic and hides addressing bugs."""
    return [[((r * 5 + c * 4 + off) % 9 - 4) * 0.125 for c in range(N)]
            for r in range(N)]


def pad_half(M):
    """Pad Nx(N/2) matrix to NxN with zero columns (junk columns are
    exactly zero and never affect dot products)."""
    return [row + [0.0] * DH for row in M]


Wq1, Wk1, Wv1 = w_n_half(1), w_n_half(3), w_n_half(5)
Wq2, Wk2, Wv2 = w_n_half(2), w_n_half(4), w_n_half(6)
W_O, W1m, W2m, W_head = w_nn(0), w_nn(2), w_nn(4), w_nn(6)

# Per-timestep adaLN modulation — DISTINCT per t: full NxN elementwise
# scale/shift matrices plus row-broadcast column gates. Every formula
# uses multipliers coprime with its modulus in BOTH dimensions.
MOD = []
for t in range(T_STEPS):
    s1 = [[1.0 + ((r * 3 + c * 2 + t * 2) % 5 - 2) * 0.125
           for c in range(N)] for r in range(N)]
    sh1 = [[((r * 2 + c * 3 + t * 3) % 7 - 3) * 0.0625
            for c in range(N)] for r in range(N)]
    s2 = [[1.0 + ((r * 2 + c * 3 + t * 3) % 5 - 2) * 0.125
           for c in range(N)] for r in range(N)]
    sh2 = [[((r * 3 + c * 2 + t * 5) % 7 - 3) * 0.0625
            for c in range(N)] for r in range(N)]
    g1_cols = [0.5 + ((c * 3 + t) % 5) * 0.25 for c in range(N)]
    g2_cols = [0.5 + ((c * 3 + t * 2) % 5) * 0.25 for c in range(N)]
    G1 = [list(g1_cols) for _ in range(N)]   # row-broadcast
    G2 = [list(g2_cols) for _ in range(N)]
    MOD.append((s1, sh1, s2, sh2, G1, G2))

# ---- raw (signed) forms ----
Xr = [[to_raw(v) for v in row] for row in X]
I_Nr = [[to_raw(v) for v in row] for row in I_N]
Wq1pr = [[to_raw(v) for v in row] for row in pad_half(Wq1)]
Wk1pr = [[to_raw(v) for v in row] for row in pad_half(Wk1)]
Wv1r = [[to_raw(v) for v in row] for row in Wv1]
Wq2pr = [[to_raw(v) for v in row] for row in pad_half(Wq2)]
Wk2pr = [[to_raw(v) for v in row] for row in pad_half(Wk2)]
Wv2r = [[to_raw(v) for v in row] for row in Wv2]
W_Or = [[to_raw(v) for v in row] for row in W_O]
W1mr = [[to_raw(v) for v in row] for row in W1m]
W2mr = [[to_raw(v) for v in row] for row in W2m]
W_headr = [[to_raw(v) for v in row] for row in W_head]
MODr = [tuple([[to_raw(v) for v in row] for row in M] for M in mod_t)
        for mod_t in MOD]


# ---- exact LayerNorm model at N=16: same layernorm_group_nxn.sv RTL
# as dit_n4's ln_row_exact, but the mean/variance truncating shifts
# are by log2(N) = 4, not 3 (N=8) or 2 (N=4). The RTL is N-generic —
# only the shift amount changes. ----
def ln_row_exact(row):
    """One N=16 group-LN beat, bit-exact vs layernorm_group_nxn.sv.
    row: N signed Q8.8 raws. Returns N signed Q8.8 raws."""
    mean = sum(row) >> LOG2N              # truncating shift by log2(N)
    devs = [x - mean for x in row]
    sqs = [fxp_mul_rtl(d, d, 9, 8, 9, 8, 18, 8)[0] for d in devs]
    var_eps = (sum(sqs) >> LOG2N) + 16    # LN_EPS = 1/16
    std, _ = fxp_sqrt_rtl(var_eps, 18, 8, 8, 8)
    return [fxp_div_signed_rtl(d, std) for d in devs]


# ---- golden: the whole 20-phase block x T_STEPS, raw Q8.8 ----
def adaln_iteration(x_in, mod_t):
    """One adaLN-conditioned denoiser block on an NxN raw matrix.
    mod_t = (s1m, sh1m, s2m, sh2m, G1M, G2M) raw for this timestep.
    Returns the phase outputs in stream order."""
    (s1m, sh1m, s2m, sh2m, G1M, G2M) = mod_t
    h1 = [ln_row_exact(row) for row in x_in]
    h1s = ew_mul(h1, s1m)                # adaLN scale (p1a)
    h1m = ew_add(h1s, sh1m)              # adaLN shift (p1b)
    Qt1 = matmul_g(h1m, Wq1pr)
    Kt1 = matmul_g(h1m, Wk1pr)
    V1 = matmul_g(h1m, Wv1r)             # Nx(N/2)
    Qt2 = matmul_g(h1m, Wq2pr)
    Kt2 = matmul_g(h1m, Wk2pr)
    V2 = matmul_g(h1m, Wv2r)             # Nx(N/2)
    P1 = [softmax_row_exact(row)
          for row in matmul_g(Qt1, transpose(Kt1))]
    P2 = [softmax_row_exact(row)
          for row in matmul_g(Qt2, transpose(Kt2))]
    O1T = matmul_g(transpose(V1), transpose(P1))   # (N/2)xN
    O2T = matmul_g(transpose(V2), transpose(P2))   # (N/2)xN
    # The transposed-emission merge: Cm[r] = [O1T[0][r]..O1T[DH-1][r],
    # O2T[0][r]..O2T[DH-1][r]] — the concat presented by the contiguous
    # O-stack's T-read
    Cm = [[O1T[h][r] for h in range(DH)]
          + [O2T[h][r] for h in range(DH)] for r in range(N)]
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

# ---- UB image (29952 words) ----
ADDR = {"I_N": 0, "Wqp1": _N2, "Wkp1": 2 * _N2, "Wv1": 3 * _N2,
        "Wqp2": 3 * _N2 + _NDH, "Wkp2": 4 * _N2 + _NDH,
        "Wv2": 5 * _N2 + _NDH,
        "W_O": 5 * _N2 + 2 * _NDH,
        "W1m": 6 * _N2 + 2 * _NDH,
        "W2m": 7 * _N2 + 2 * _NDH,
        "W_head": 8 * _N2 + 2 * _NDH,
        "x": _ADDR_X}
# Verify weight address arithmetic
assert ADDR["W_head"] + _N2 == _WHOST
GOLD = {a: 0 for a in range(IMG_WORDS)}


def put(base, M, rows, cols):
    for r in range(rows):
        for c in range(cols):
            GOLD[base + cols * r + c] = M[r][c]


put(ADDR["I_N"], I_Nr, N, N)
put(ADDR["Wqp1"], Wq1pr, N, N)
put(ADDR["Wkp1"], Wk1pr, N, N)
put(ADDR["Wv1"], Wv1r, N, DH)
put(ADDR["Wqp2"], Wq2pr, N, N)
put(ADDR["Wkp2"], Wk2pr, N, N)
put(ADDR["Wv2"], Wv2r, N, DH)
put(ADDR["W_O"], W_Or, N, N)
put(ADDR["W1m"], W1mr, N, N)
put(ADDR["W2m"], W2mr, N, N)
put(ADDR["W_head"], W_headr, N, N)
put(ADDR["x"], Xr, N, N)
for _t in range(T_STEPS):
    (_s1m, _sh1m, _s2m, _sh2m, _G1M, _G2M) = MODr[_t]
    for _name, _M in (("s1m", _s1m), ("sh1m", _sh1m), ("s2m", _s2m),
                      ("sh2m", _sh2m), ("G1M", _G1M), ("G2M", _G2M)):
        put(MOD_BASE[_t] + MOFF[_name], _M, N, N)
for _t in range(T_STEPS):
    _base = WBASE + _t * E_REGION
    (_h1, _h1s, _h1m, _Qt1, _Kt1, _V1, _Qt2, _Kt2, _V2, _P1, _P2,
     _O1T, _O2T, _t1g, _x2, _h2, _h2s, _h2m, _m1, _t2g, _x3,
     _y) = ITERS[_t]
    for _name, _M, _rows, _cols in (
            ("h1", _h1, N, N), ("h1s", _h1s, N, N),
            ("h1m", _h1m, N, N),
            ("Qt1", _Qt1, N, N), ("Kt1", _Kt1, N, N),
            ("V1", _V1, N, DH),
            ("Qt2", _Qt2, N, N), ("Kt2", _Kt2, N, N),
            ("V2", _V2, N, DH),
            ("P1", _P1, N, N), ("P2", _P2, N, N),
            ("O1T", _O1T, DH, N), ("O2T", _O2T, DH, N),
            ("t1g", _t1g, N, N), ("x2", _x2, N, N),
            ("h2", _h2, N, N), ("h2s", _h2s, N, N),
            ("h2m", _h2m, N, N),
            ("m1", _m1, N, N), ("t2g", _t2g, N, N),
            ("x3", _x3, N, N), ("y", _y, N, N)):
        put(_base + OFF[_name], _M, _rows, _cols)
assert len(GOLD) == IMG_WORDS, \
    f"image is {len(GOLD)} words, expected {IMG_WORDS}"


def loopxl_word(count, length, stride_a, stride_w, wbase):
    """LOOPI control word with the 17b1 15-bit length (low 8 bits at
    [7:0], high 7 at [23:17]) and the 17a wbase field [71:56]."""
    assert 0 < length < (1 << 15)
    return (CTRL | (count << 8) | (length & 0xFF) | IDX_FLAG
            | ((length >> 8) << 17)
            | (stride_a << 24) | (stride_w << 40) | (wbase << 56))


def generate_program_adaln_n16():
    """4226 words: 913-word host prefix + LOOPI(3, 3312, sa=5120,
    sw=5120, wbase=14592) + the 3312-word body (20 phases: 14 plain
    x 148 + 8 modulation/gate x 155). Body addresses are the
    iteration-0 absolute addresses; the strides advance every region
    read AND every modulation-block read by E=5120 per iteration while
    host weight reads (< wbase, ptr-1) freeze."""
    g = ProgGen()
    g.idle()
    # Sparse fill: the mod blocks and x land at their absolute
    # addresses; gap words are zeros (written, never read).
    sparse = {a: 0 for a in range(WBASE)}
    pos = 0
    for M in (I_N, pad_half(Wq1), pad_half(Wk1), Wv1,
              pad_half(Wq2), pad_half(Wk2), Wv2, W_O, W1m, W2m, W_head):
        for row in M:
            for v in row:
                sparse[pos] = to_fixed(v)
                pos += 1
    assert pos == _WHOST, f"weight pos {pos}, expected {_WHOST}"
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
    _prefix_len = WBASE // N + 1           # 913
    assert len(g.prog) == _prefix_len, \
        f"prefix is {len(g.prog)} words, expected {_prefix_len}"

    def matmul_phase(w_addr, w_rows, w_cols, w_T, a_addr, a_rows, a_cols,
                     a_T, pathway):
        g.issue_read(ptr=1, addr=w_addr, rows=w_rows, cols=w_cols,
                     transpose=w_T)
        g.tick(2 * N + 1)                  # weight preload: 2N+1
        g.issue_read(ptr=0, addr=a_addr, rows=a_rows, cols=a_cols,
                     transpose=a_T, pathway=pathway)
        g.switch_pulse()
        g.tick(7 * N)                      # emission wait

    def scale_phase(w_addr, w_T, a_addr, a_T, s_addr):
        """(A @ W) . S: matmul + a mid-phase ptr-8 scale read (the
        item-18a contract). The bias-operand walk is SELF-TIMED from
        the read's execution (rd_bias_time_counter <= 0 at issue, per
        unified_buffer_nxn.sv), so the operand read sits at
        switch + (N-2): the systolic stream reaches the VPU chain head
        N-4 cycles later than at N=4, and the operand window must
        coincide with it."""
        g.issue_read(ptr=1, addr=w_addr, rows=N, cols=N, transpose=w_T)
        g.tick(2 * N + 1)
        g.issue_read(ptr=0, addr=a_addr, rows=N, cols=N, transpose=a_T,
                     pathway=PATHWAY_BYPASS)
        g.switch_pulse()
        g.tick(N - 2)
        g.issue_read(ptr=PTR_SCALE, addr=s_addr, rows=N, cols=N)
        g.tick(13 * N // 2)

    def radd_phase(w_addr, w_T, a_addr, a_T, r_addr):
        g.issue_read(ptr=1, addr=w_addr, rows=N, cols=N, transpose=w_T)
        g.tick(2 * N + 1)
        g.issue_read(ptr=0, addr=a_addr, rows=N, cols=N, transpose=a_T,
                     pathway=PATHWAY_BIAS)
        g.switch_pulse()
        g.tick(N - 2)
        g.issue_read(ptr=PTR_RESIDUAL, addr=r_addr, rows=N, cols=N)
        g.tick(13 * N // 2)

    g.prog.append(loopxl_word(T_STEPS, BODY_LEN, E_REGION, E_REGION,
                              WBASE))
    _body_at = len(g.prog)
    B = WBASE  # iteration-0 region base; strides do the rest
    M0 = MOD_BASE[0]

    # p1: h1 = LN(x) — identity matmul + the LN group stage
    matmul_phase(ADDR["I_N"], N, N, 0, ADDR["x"], N, N, 0, PATHWAY_LN)
    # p1a/p1b: the adaLN modulation of h1 (scale then shift)
    scale_phase(ADDR["I_N"], 0, B + OFF["h1"], 0, M0 + MOFF["s1m"])
    radd_phase(ADDR["I_N"], 0, B + OFF["h1s"], 0, M0 + MOFF["sh1m"])
    # p2-7: per-head Q/K/V projections of h1m (bypass)
    matmul_phase(ADDR["Wqp1"], N, N, 0, B + OFF["h1m"], N, N, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wkp1"], N, N, 0, B + OFF["h1m"], N, N, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wv1"], N, DH, 0, B + OFF["h1m"], N, N, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wqp2"], N, N, 0, B + OFF["h1m"], N, N, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wkp2"], N, N, 0, B + OFF["h1m"], N, N, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wv2"], N, DH, 0, B + OFF["h1m"], N, N, 0,
                 PATHWAY_BYPASS)
    # p8/9: P_h = softmax(Q~_h @ K~_h^T)
    matmul_phase(B + OFF["Kt1"], N, N, 1, B + OFF["Qt1"], N, N, 0,
                 PATHWAY_SOFTMAX)
    matmul_phase(B + OFF["Kt2"], N, N, 1, B + OFF["Qt2"], N, N, 0,
                 PATHWAY_SOFTMAX)
    # p10/11: O_h^T = V_h^T @ P_h^T ((N/2)xN; the item-14 merge)
    matmul_phase(B + OFF["P1"], N, N, 1, B + OFF["V1"], N, DH, 1,
                 PATHWAY_BYPASS)
    matmul_phase(B + OFF["P2"], N, N, 1, B + OFF["V2"], N, DH, 1,
                 PATHWAY_BYPASS)
    # p12a: t1g = ([O1|O2] @ W_O) . G1M — the T-read of the contiguous
    # O-stack presents the concat; the gate rides the ptr-8 channel
    scale_phase(ADDR["W_O"], 0, B + OFF["O1T"], 1, M0 + MOFF["G1M"])
    # p12b: x2 = t1g + x — residual read (ptr-7) of the phase input x
    radd_phase(ADDR["I_N"], 0, B + OFF["t1g"], 0, ADDR["x"])
    # p13: h2 = LN(x2)
    matmul_phase(ADDR["I_N"], N, N, 0, B + OFF["x2"], N, N, 0,
                 PATHWAY_LN)
    # p13a/p13b: the adaLN modulation of h2
    scale_phase(ADDR["I_N"], 0, B + OFF["h2"], 0, M0 + MOFF["s2m"])
    radd_phase(ADDR["I_N"], 0, B + OFF["h2s"], 0, M0 + MOFF["sh2m"])
    # p14: m1 = SiLU(h2m @ W1m)
    matmul_phase(ADDR["W1m"], N, N, 0, B + OFF["h2m"], N, N, 0,
                 PATHWAY_SILU)
    # p15a: t2g = (m1 @ W2m) . G2M
    scale_phase(ADDR["W2m"], 0, B + OFF["m1"], 0, M0 + MOFF["G2M"])
    # p15b: x3 = t2g + x2
    radd_phase(ADDR["I_N"], 0, B + OFF["t2g"], 0, B + OFF["x2"])
    # p16: y = x3 @ W_head (the final stream of the iteration)
    matmul_phase(ADDR["W_head"], N, N, 0, B + OFF["x3"], N, N, 0,
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


_PROG_LEN = WBASE // N + 1 + 1 + BODY_LEN    # 4226
_SIM_WAIT = _PROG_LEN + 2 * BODY_LEN + 40     # 10890


@cocotb.test()
async def test_tpu_nxn_prog_adaln_n16(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    prog = generate_program_adaln_n16()
    assert len(prog) == _PROG_LEN, \
        f"program is {len(prog)} words, expected {_PROG_LEN}"
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
    # 3312-word body T=3 times.
    dut.rst.value = 0
    await tick(dut)
    dut.run.value = 1
    await tick(dut)
    dut.run.value = 0

    # Emission: loaded words + 2 extra body passes (the first iteration
    # executes during load); the last body's 112-cycle idle + 40 margin
    # covers the final y stream.
    await tick(dut, _SIM_WAIT)
    collector.kill()

    # ---- per-lane streams, beat-exact across all three iterations ----
    # Lane j < DH: 18 full-width (N-beat) streams + 2 V_h (N-beat) +
    #   2 O_h^T (DH-beat) = 18*N + 2*N + 2*DH = 336 at N=16
    # Lane j >= DH: 18 full-width + 2 O_h^T = 18*N + 2*DH = 304
    def col(M, j, rows):
        return [M[r][j] & 0xFFFF for r in range(rows)]

    _BEATS_LO = 18 * N + 2 * N + 2 * DH     # 336
    _BEATS_HI = 18 * N + 2 * DH              # 304

    for j in range(N):
        expected = []
        for t in range(T_STEPS):
            (h1, h1s, h1m, Qt1, Kt1, V1, Qt2, Kt2, V2, P1, P2,
             O1T, O2T, t1g, x2, h2, h2s, h2m, m1, t2g, x3, y) = ITERS[t]
            for M in (h1, h1s, h1m, Qt1, Kt1):
                expected += col(M, j, N)
            if j < DH:
                expected += col(V1, j, N)
            for M in (Qt2, Kt2):
                expected += col(M, j, N)
            if j < DH:
                expected += col(V2, j, N)
            for M in (P1, P2):
                expected += col(M, j, N)
            expected += col(O1T, j, DH) + col(O2T, j, DH)
            for M in (t1g, x2, h2, h2s, h2m, m1, t2g, x3, y):
                expected += col(M, j, N)
        _exp_len = _BEATS_LO if j < DH else _BEATS_HI
        assert len(lanes[j]) == _exp_len * T_STEPS, (
            f"lane {j}: {len(lanes[j])} beats, expected "
            f"{_exp_len * T_STEPS}")
        divs = [k for k in range(len(expected))
                if lanes[j][k] != expected[k]]
        div = divs[0] if divs else None
        _per = _BEATS_LO if j < DH else _BEATS_HI
        assert lanes[j] == expected, (
            f"VPU lane {j}: {len(lanes[j])} beats, expected "
            f"{len(expected)}; {len(divs)} divergent beats "
            f"{divs[:24]}; first at beat {div} "
            f"(iteration {div // _per}): got "
            f"{[f'{from_fixed(w):+.4f}' for w in lanes[j][max(0, div-2):div+3]]}"
            f", expected "
            f"{[f'{from_fixed(w):+.4f}' for w in expected[max(0, div-2):div+3]]}"
            f" (beats {max(0, div-2)}..{div+2})")

    # ---- final UB image: all 29952 words exact ----
    for a in range(IMG_WORDS):
        got = nxn.ub_inst.ub_memory[a].value.integer & 0xFFFF
        want = GOLD[a] & 0xFFFF
        assert got == want, (
            f"mem[{a}] = {from_fixed(got):+.4f}, expected "
            f"{from_fixed(want):+.4f}")

    print(f"tpu_nxn_prog N={N} adaLN CAPSTONE at scale OK "
          f"({_PROG_LEN}-word program: ONE {BODY_LEN}-word "
          f"adaLN-conditioned denoiser body — LN, scale+shift, "
          f"2-head attention, gate+residual, LN, scale+shift, "
          f"SiLU MLP, gate+residual, head projection — looped "
          f"T={T_STEPS} by LOOPI(3, {BODY_LEN}, {E_REGION}, "
          f"{E_REGION}, wbase={WBASE}) with DISTINCT per-timestep "
          f"modulation striding through the host image; "
          f"{IMG_WORDS}-word UB image exact)")
