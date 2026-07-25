"""DRAFT (harness-side, /tmp — copy into the repo only after the
chip_scale loop finishes and the 18a scale RTL is committed).

Gate test for roadmap item 18b — THE adaLN CAPSTONE: the item-17b2
DiT denoiser block WITH per-timestep adaptive LayerNorm conditioning
(adaLN scale + shift on both LNs, gates on the attention and MLP
outputs), iterated T=3 timesteps as ONE loaded program — the weight-
shared, modulation-reading body executed by a single LOOPI control
word. Written by the harness author, not the agent — per tinytpu-loop
README "the gate grows with the design".

New vs 17b2: composes the item-18a scale stage into the block. adaLN
(DiT's conditioning): given per-timestep modulation (s1, sh1, s2, sh2,
g1, g2 from the t embedding — host-precomputed here, quantized Q8.8):

    h1  = LN(x);  h1m = h1 . s1m + sh1m        (scale then shift: the
    Q/K/V = h1m @ W_...                          18a stage is BEFORE
    attn = merge(heads) @ W_O                    bias in the chain)
    x2  = attn . G1M + x                       (gate = per-column scale
    h2  = LN(x2); h2m = h2 . s2m + sh2m          = elementwise with a
    mlp  = SiLU(h2m @ W1m) @ W2m                 row-broadcast matrix)
    x3  = mlp . G2M + x2
    y   = x3 @ W_head

PER-TIMESTEP MODULATION (the whole point of adaLN — each t gets
DISTINCT values): the six 4x4 modulation matrices (s1m, sh1m, s2m,
sh2m, G1M, G2M) live in the HOST image as three copies spaced exactly
E=320 words apart (@176, @496, @816). This rides the t-loop rule
confirmed live by 17b2: ptr-0/ptr-7 (and item-18a's ptr-8) reads
ALWAYS stride by stride_a, even below wbase — only ptr-1 reads below
wbase freeze. So the modulation reads (ptr-8 scale / ptr-7 shift)
advance one copy per iteration while every host WEIGHT (ptr-1) stays
frozen and shared. (The pre-18a sketch folded the gates into
per-timestep weight copies W_O_t/W2m_t — impossible: ptr-1 reads
below wbase freeze, and region-resident host data can't be distinct
per iteration when body write beats replay verbatim. The gates move
ON CHIP through the scale stage instead — exactly what it is for.)

The block (20 phases; p1a/p1b/p13a/p13b are the adaLN modulation
pairs, p12a/p15a the on-chip gates):

    h1  = LN(x)                 p1   (identity matmul + LN pathway)
    h1s = h1 . s1m              p1a  (identity matmul + ptr-8 scale)
    h1m = h1s + sh1m            p1b  (identity matmul + ptr-7 + bias)
    Q~/K~/V per head            p2-7 (read h1m, bypass)
    P_h = softmax(Q~ K~^T)      p8/9
    O_h^T = V_h^T @ P_h^T       p10/11 (transposed-emission merge)
    t1g = ([O1|O2] @ W_O) . G1M p12a (T-read of the O-stack + ptr-8)
    x2  = t1g + x               p12b (identity matmul + ptr-7 of x)
    h2  = LN(x2)                p13
    h2s = h2 . s2m              p13a
    h2m = h2s + sh2m            p13b
    m1  = SiLU(h2m @ W1m)       p14
    t2g = (m1 @ W2m) . G2M      p15a
    x3  = t2g + x2              p15b
    y   = x3 @ W_head           p16  (final stream of the iteration)

The t-loop (the sampler): all 20 phases form ONE 1168-word LOOPI body
(14 plain x 52 + 8 modulation/gate phases x 55) run by
LOOPI(3, 1168, stride_a=320, stride_w=320, wbase=928). Host weights
(160 words) freeze below wbase; host modulation copies (@176/496/816)
stride one block per iteration; host x sits at wbase-16 (@912) so
phase 1's striding act read lands on iteration t-1's y (at B+E-16 =
B+304) — the sampler recurrence never leaves the UB.

Host image (928 words = wbase):
  I4 @0, W_qp1 @16 / W_kp1 @32 (4x4 padded), W_v1 @48 (4x2),
  W_qp2 @56 / W_kp2 @72 (4x4 padded), W_v2 @88 (4x2), W_O @96,
  W1m @112, W2m @128, W_head @144 (= 160 words), gap @160..175,
  mod t0 @176..271 (s1m/sh1m/s2m/sh2m/G1M/G2M, 16 each), gap,
  mod t1 @496..591, gap, mod t2 @816..911, x @912 (= wbase-16).

Per-iteration append region (E=320 words, base B_i = 928 + i*320):
  h1 @+0, h1s @+16, h1m @+32, Q~1 @+48, K~1 @+64, V1 @+80 (8w),
  Q~2 @+88, K~2 @+104, V2 @+120 (8w), P1 @+128, P2 @+144,
  O1^T @+160 (8w), O2^T @+168 (8w), t1g @+176, x2 @+192, h2 @+208,
  h2s @+224, h2m @+240, m1 @+256, t2g @+272, x3 @+288, y @+304
  (final stream, E-16 — the t-loop linchpin).

Program: 233-word prefix (232 host beats + trailing tick; the leading
idle primes but never commits) + the LOOPI word + the 1168-word body
= 1402 words. PROG_DEPTH=2048, UB_WIDTH=2048; the final image is
928 + 3*320 = 1888 words, final y @1872..1887.

Golden: hardware-exact integer arithmetic throughout — the 17b2 exact
LayerNorm model + ew_mul (item-18a fxp_mul elementwise) / ew_add
(item-17a) composed with matmul_g / softmax_row_exact / silu_exact.
Modulation values are DISTINCT per timestep (the host-image copies
differ — the iteration-t region proves the mod reads advanced).

Checks (LIVE asserts — PYTHONOPTIMIZE is empty):
  - per-lane VPU streams beat-exact across ALL THREE iterations:
    lanes 0/1 capture 252 beats (84/iteration: 18 full-width streams +
    the 4x2 V_h pair + the two 2-beat O_h^T), lanes 2/3 capture 228;
  - the full 1888-word UB image exact everywhere — host regions
    intact, every iteration's 320-word region exact.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# Same model stack as 17b2 plus the 18a elementwise multiply.
from test_tpu_nxn_prog_n4 import ProgGen, to_fixed, from_fixed
from test_tpu_nxn_ic_attn_n4 import softmax_row_exact, fxp_add
from test_tpu_nxn_prog_mh_attn_n4 import (
    matmul_g, to_raw, transpose, pad42,
)
from test_tpu_nxn_prog_radd_n4 import ew_add
from test_tpu_nxn_prog_scale_n4 import ew_mul
from test_silu_parent import silu_exact
from test_tpu_nxn_prog_dit_n4 import ln_row_exact

N = int(os.environ.get("TPU_NXN_PROG_N", "4"))
WORD_W = 134 + 17 * (N - 2)   # instruction width (item 13: SiLU bit)
CTRL = 1 << WORD_W            # control-word escape bit (prog-word MSB)
IDX_FLAG = 1 << 16

PATHWAY_LN = 0b0100000
PATHWAY_SOFTMAX = 0b1000000
PATHWAY_SILU = 0b10000000
PATHWAY_BIAS = 0b0001000
PATHWAY_BYPASS = 0b0000000
PTR_RESIDUAL = 7
PTR_SCALE = 8

T_STEPS = 3
BODY_LEN = 1168             # 14 x 52 + 8 x 55
E_REGION = 320              # per-iteration append size (= stride)
WBASE = 928                 # host image size; x @ WBASE-16

# ---- stimulus: quarter/eighth multiples, exact in Q8.8, distinct per
# matrix so an addressing mixup reads as a value mismatch ----
X = [[0.50, -0.25, 0.75, 0.00],
     [-0.50, 0.25, 0.00, 0.50],
     [0.25, 0.50, -0.75, 0.25],
     [0.00, -0.50, 0.25, -0.25]]

I4 = [[1.0 if k == j else 0.0 for j in range(4)] for k in range(4)]

Wq1 = [[0.50, -0.25], [0.25, 0.50], [-0.50, 0.00], [0.75, -0.50]]
Wk1 = [[-0.25, 0.50], [0.50, 0.25], [0.00, -0.75], [0.25, 0.50]]
Wv1 = [[0.25, 0.50], [-0.50, 0.25], [0.75, 0.00], [0.00, -0.25]]
Wq2 = [[-0.50, 0.25], [0.00, -0.50], [0.75, 0.50], [-0.25, 0.00]]
Wk2 = [[0.50, -0.50], [-0.25, 0.75], [0.25, 0.00], [0.00, 0.25]]
Wv2 = [[-0.25, 0.00], [0.50, -0.25], [0.00, 0.50], [0.25, 0.75]]

W_O = [[0.50, 0.25, -0.50, 0.00],
       [-0.25, 0.75, 0.00, 0.50],
       [0.00, -0.25, 0.50, -0.75],
       [0.75, 0.00, -0.25, 0.25]]
W1m = [[0.25, -0.50, 0.75, 0.00],
       [0.50, 0.25, -0.25, 0.50],
       [-0.75, 0.00, 0.50, -0.25],
       [0.00, 0.50, -0.50, 0.25]]
W2m = [[-0.50, 0.25, 0.00, 0.75],
       [0.25, 0.50, -0.75, 0.00],
       [0.00, -0.25, 0.50, 0.25],
       [0.75, 0.00, -0.50, -0.25]]
W_head = [[0.50, 0.00, -0.25, 0.75],
          [0.00, -0.50, 0.25, 0.00],
          [-0.25, 0.75, 0.50, -0.50],
          [0.25, 0.00, -0.75, 0.50]]

# Per-timestep adaLN modulation — DISTINCT per t (this is what makes
# the test prove the modulation reads advance): scales cluster around
# 1.0 (the adaLN 1+s shape), shifts are small, gates vary per column.
# s1m_t/sh1m_t/s2m_t/sh2m_t are full 4x4 elementwise matrices; G1M_t /
# G2M_t are row-broadcast column gates (gate k scales output column k).
MOD = []
for t in range(T_STEPS):
    s1 = [[1.00 + 0.25 * t, 1.00, 0.75 + 0.25 * t, 1.25 - 0.25 * t],
          [0.75 + 0.125 * t, 1.25, 1.00 - 0.125 * t, 0.875],
          [1.125, 0.875 + 0.125 * t, 1.00, 1.00 + 0.125 * t],
          [1.00 - 0.125 * t, 1.125, 1.25 - 0.25 * t, 0.75 + 0.25 * t]]
    sh1 = [[0.125 * (t + 1), -0.125, 0.00, 0.25 - 0.125 * t],
           [-0.25, 0.125 * t, 0.125, -0.125 * (t + 1)],
           [0.00, 0.25 - 0.125 * t, -0.125, 0.125],
           [0.125, -0.125 * t, 0.25, -0.25 + 0.125 * t]]
    s2 = [[1.25 - 0.25 * t, 0.875, 1.00 + 0.125 * t, 1.00],
          [1.00, 1.125 + 0.125 * t, 0.75 + 0.25 * t, 1.25 - 0.125 * t],
          [0.875 + 0.125 * t, 1.00, 1.25, 1.00 - 0.125 * t],
          [1.00, 1.00 - 0.25 * t, 1.125, 0.75 + 0.125 * t]]
    sh2 = [[-0.125 * (t + 1), 0.25, 0.125, -0.125],
           [0.125 * t, -0.25 + 0.125 * t, 0.00, 0.125],
           [0.25 - 0.125 * t, 0.125, -0.125 * t, -0.25],
           [0.00, -0.125, 0.125 * (t + 1), 0.125 - 0.125 * t]]
    g1_cols = [0.50 + 0.25 * t, 1.00, 1.25 - 0.25 * t, 0.75 + 0.125 * t]
    g2_cols = [1.00 - 0.125 * t, 0.75 + 0.25 * t, 1.00, 1.25 - 0.25 * t]
    G1 = [list(g1_cols) for _ in range(4)]   # row-broadcast
    G2 = [list(g2_cols) for _ in range(4)]
    MOD.append((s1, sh1, s2, sh2, G1, G2))

# ---- raw (signed) forms ----
Xr = [[to_raw(v) for v in row] for row in X]
I4r = [[to_raw(v) for v in row] for row in I4]
Wq1pr = [[to_raw(v) for v in row] for row in pad42(Wq1)]
Wk1pr = [[to_raw(v) for v in row] for row in pad42(Wk1)]
Wv1r = [[to_raw(v) for v in row] for row in Wv1]
Wq2pr = [[to_raw(v) for v in row] for row in pad42(Wq2)]
Wk2pr = [[to_raw(v) for v in row] for row in pad42(Wk2)]
Wv2r = [[to_raw(v) for v in row] for row in Wv2]
W_Or = [[to_raw(v) for v in row] for row in W_O]
W1mr = [[to_raw(v) for v in row] for row in W1m]
W2mr = [[to_raw(v) for v in row] for row in W2m]
W_headr = [[to_raw(v) for v in row] for row in W_head]
MODr = [tuple([[to_raw(v) for v in row] for row in M] for M in mod_t)
        for mod_t in MOD]


# ---- golden: the whole 20-phase block x T_STEPS, raw Q8.8 ----
def adaln_iteration(x_in, mod_t):
    """One adaLN-conditioned denoiser block on a 4x4 raw matrix.
    mod_t = (s1m, sh1m, s2m, sh2m, G1M, G2M) raw for this timestep.
    Returns the phase outputs in stream order."""
    (s1m, sh1m, s2m, sh2m, G1M, G2M) = mod_t
    h1 = [ln_row_exact(row) for row in x_in]
    h1s = ew_mul(h1, s1m)                # adaLN scale (p1a)
    h1m = ew_add(h1s, sh1m)              # adaLN shift (p1b)
    Qt1 = matmul_g(h1m, Wq1pr)
    Kt1 = matmul_g(h1m, Wk1pr)
    V1 = matmul_g(h1m, Wv1r)             # 4x2
    Qt2 = matmul_g(h1m, Wq2pr)
    Kt2 = matmul_g(h1m, Wk2pr)
    V2 = matmul_g(h1m, Wv2r)             # 4x2
    P1 = [softmax_row_exact(row)
          for row in matmul_g(Qt1, transpose(Kt1))]
    P2 = [softmax_row_exact(row)
          for row in matmul_g(Qt2, transpose(Kt2))]
    O1T = matmul_g(transpose(V1), transpose(P1))   # 2x4
    O2T = matmul_g(transpose(V2), transpose(P2))   # 2x4
    Cm = [[O1T[0][r], O1T[1][r], O2T[0][r], O2T[1][r]] for r in range(4)]
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

# ---- UB image (1888 words) ----
ADDR = {"I4": 0, "Wqp1": 16, "Wkp1": 32, "Wv1": 48,
        "Wqp2": 56, "Wkp2": 72, "Wv2": 88, "W_O": 96,
        "W1m": 112, "W2m": 128, "W_head": 144, "x": WBASE - 16}
MOD_BASE = [176 + t * E_REGION for t in range(T_STEPS)]  # 176/496/816
MOFF = {"s1m": 0, "sh1m": 16, "s2m": 32, "sh2m": 48,
        "G1M": 64, "G2M": 80}
OFF = {"h1": 0, "h1s": 16, "h1m": 32, "Qt1": 48, "Kt1": 64, "V1": 80,
       "Qt2": 88, "Kt2": 104, "V2": 120, "P1": 128, "P2": 144,
       "O1T": 160, "O2T": 168, "t1g": 176, "x2": 192, "h2": 208,
       "h2s": 224, "h2m": 240, "m1": 256, "t2g": 272, "x3": 288,
       "y": 304}
GOLD = {a: 0 for a in range(WBASE + T_STEPS * E_REGION)}


def put(base, M, rows, cols):
    for r in range(rows):
        for c in range(cols):
            GOLD[base + cols * r + c] = M[r][c]


put(ADDR["I4"], I4r, 4, 4)
put(ADDR["Wqp1"], Wq1pr, 4, 4)
put(ADDR["Wkp1"], Wk1pr, 4, 4)
put(ADDR["Wv1"], Wv1r, 4, 2)
put(ADDR["Wqp2"], Wq2pr, 4, 4)
put(ADDR["Wkp2"], Wk2pr, 4, 4)
put(ADDR["Wv2"], Wv2r, 4, 2)
put(ADDR["W_O"], W_Or, 4, 4)
put(ADDR["W1m"], W1mr, 4, 4)
put(ADDR["W2m"], W2mr, 4, 4)
put(ADDR["W_head"], W_headr, 4, 4)
put(ADDR["x"], Xr, 4, 4)
for _t in range(T_STEPS):
    (_s1m, _sh1m, _s2m, _sh2m, _G1M, _G2M) = MODr[_t]
    for _name, _M in (("s1m", _s1m), ("sh1m", _sh1m), ("s2m", _s2m),
                      ("sh2m", _sh2m), ("G1M", _G1M), ("G2M", _G2M)):
        put(MOD_BASE[_t] + MOFF[_name], _M, 4, 4)
for _t in range(T_STEPS):
    _base = WBASE + _t * E_REGION
    (_h1, _h1s, _h1m, _Qt1, _Kt1, _V1, _Qt2, _Kt2, _V2, _P1, _P2,
     _O1T, _O2T, _t1g, _x2, _h2, _h2s, _h2m, _m1, _t2g, _x3,
     _y) = ITERS[_t]
    for _name, _M, _rows, _cols in (
            ("h1", _h1, 4, 4), ("h1s", _h1s, 4, 4), ("h1m", _h1m, 4, 4),
            ("Qt1", _Qt1, 4, 4), ("Kt1", _Kt1, 4, 4), ("V1", _V1, 4, 2),
            ("Qt2", _Qt2, 4, 4), ("Kt2", _Kt2, 4, 4), ("V2", _V2, 4, 2),
            ("P1", _P1, 4, 4), ("P2", _P2, 4, 4), ("O1T", _O1T, 2, 4),
            ("O2T", _O2T, 2, 4), ("t1g", _t1g, 4, 4), ("x2", _x2, 4, 4),
            ("h2", _h2, 4, 4), ("h2s", _h2s, 4, 4), ("h2m", _h2m, 4, 4),
            ("m1", _m1, 4, 4), ("t2g", _t2g, 4, 4), ("x3", _x3, 4, 4),
            ("y", _y, 4, 4)):
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


def generate_program_adaln():
    """1402 words: 233-word host prefix + LOOPI(3, 1168, sa=320,
    sw=320, wbase=928) + the 1168-word body (20 phases: 14 plain x 52
    + 8 modulation/gate x 55). Body addresses are the iteration-0
    absolute addresses; the strides advance every region read AND
    every modulation-block read by E=320 per iteration while host
    weight reads (< wbase, ptr-1) freeze."""
    g = ProgGen()
    g.idle()
    # Sparse fill: the modulation blocks and x land at their absolute
    # addresses; gap words are zeros (written, never read).
    sparse = {a: 0 for a in range(WBASE)}
    pos = 0
    for M in (I4, pad42(Wq1), pad42(Wk1), Wv1,
              pad42(Wq2), pad42(Wk2), Wv2, W_O, W1m, W2m, W_head):
        for row in M:
            for v in row:
                sparse[pos] = to_fixed(v)
                pos += 1
    assert pos == 160
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
    assert len(g.prog) == 233, \
        f"prefix is {len(g.prog)} words, expected 233"

    def matmul_phase(w_addr, w_rows, w_cols, w_T, a_addr, a_rows, a_cols,
                     a_T, pathway):
        g.issue_read(ptr=1, addr=w_addr, rows=w_rows, cols=w_cols,
                     transpose=w_T)
        g.tick(9)
        g.issue_read(ptr=0, addr=a_addr, rows=a_rows, cols=a_cols,
                     transpose=a_T, pathway=pathway)
        g.switch_pulse()
        g.tick(40)

    def scale_phase(w_addr, w_T, a_addr, a_T, s_addr):
        """(A @ W) . S: identity-or-weight matmul + a mid-phase ptr-8
        scale read (the item-18a contract)."""
        g.issue_read(ptr=1, addr=w_addr, rows=4, cols=4, transpose=w_T)
        g.tick(9)
        g.issue_read(ptr=0, addr=a_addr, rows=4, cols=4, transpose=a_T,
                     pathway=PATHWAY_BYPASS)
        g.switch_pulse()
        g.tick(2)
        g.issue_read(ptr=PTR_SCALE, addr=s_addr, rows=4, cols=4)
        g.tick(40)

    def radd_phase(w_addr, w_T, a_addr, a_T, r_addr):
        g.issue_read(ptr=1, addr=w_addr, rows=4, cols=4, transpose=w_T)
        g.tick(9)
        g.issue_read(ptr=0, addr=a_addr, rows=4, cols=4, transpose=a_T,
                     pathway=PATHWAY_BIAS)
        g.switch_pulse()
        g.tick(2)
        g.issue_read(ptr=PTR_RESIDUAL, addr=r_addr, rows=4, cols=4)
        g.tick(40)

    g.prog.append(loopxl_word(T_STEPS, BODY_LEN, E_REGION, E_REGION,
                              WBASE))
    _body_at = len(g.prog)
    B = WBASE  # iteration-0 region base; strides do the rest
    M0 = MOD_BASE[0]

    # p1: h1 = LN(x) — identity matmul + the LN group stage
    matmul_phase(ADDR["I4"], 4, 4, 0, ADDR["x"], 4, 4, 0, PATHWAY_LN)
    # p1a/p1b: the adaLN modulation of h1 (scale then shift)
    scale_phase(ADDR["I4"], 0, B + OFF["h1"], 0, M0 + MOFF["s1m"])
    radd_phase(ADDR["I4"], 0, B + OFF["h1s"], 0, M0 + MOFF["sh1m"])
    # p2-7: per-head Q/K/V projections of h1m (bypass)
    matmul_phase(ADDR["Wqp1"], 4, 4, 0, B + OFF["h1m"], 4, 4, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wkp1"], 4, 4, 0, B + OFF["h1m"], 4, 4, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wv1"], 4, 2, 0, B + OFF["h1m"], 4, 4, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wqp2"], 4, 4, 0, B + OFF["h1m"], 4, 4, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wkp2"], 4, 4, 0, B + OFF["h1m"], 4, 4, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wv2"], 4, 2, 0, B + OFF["h1m"], 4, 4, 0,
                 PATHWAY_BYPASS)
    # p8/9: P_h = softmax(Q~_h @ K~_h^T)
    matmul_phase(B + OFF["Kt1"], 4, 4, 1, B + OFF["Qt1"], 4, 4, 0,
                 PATHWAY_SOFTMAX)
    matmul_phase(B + OFF["Kt2"], 4, 4, 1, B + OFF["Qt2"], 4, 4, 0,
                 PATHWAY_SOFTMAX)
    # p10/11: O_h^T = V_h^T @ P_h^T (2x4; the item-14 merge)
    matmul_phase(B + OFF["P1"], 4, 4, 1, B + OFF["V1"], 4, 2, 1,
                 PATHWAY_BYPASS)
    matmul_phase(B + OFF["P2"], 4, 4, 1, B + OFF["V2"], 4, 2, 1,
                 PATHWAY_BYPASS)
    # p12a: t1g = ([O1|O2] @ W_O) . G1M — the T-read presents the
    # concat; the gate rides the ptr-8 scale channel
    scale_phase(ADDR["W_O"], 0, B + OFF["O1T"], 1, M0 + MOFF["G1M"])
    # p12b: x2 = t1g + x — residual read (ptr-7) of the phase input x
    radd_phase(ADDR["I4"], 0, B + OFF["t1g"], 0, ADDR["x"])
    # p13: h2 = LN(x2)
    matmul_phase(ADDR["I4"], 4, 4, 0, B + OFF["x2"], 4, 4, 0, PATHWAY_LN)
    # p13a/p13b: the adaLN modulation of h2
    scale_phase(ADDR["I4"], 0, B + OFF["h2"], 0, M0 + MOFF["s2m"])
    radd_phase(ADDR["I4"], 0, B + OFF["h2s"], 0, M0 + MOFF["sh2m"])
    # p14: m1 = SiLU(h2m @ W1m)
    matmul_phase(ADDR["W1m"], 4, 4, 0, B + OFF["h2m"], 4, 4, 0,
                 PATHWAY_SILU)
    # p15a: t2g = (m1 @ W2m) . G2M
    scale_phase(ADDR["W2m"], 0, B + OFF["m1"], 0, M0 + MOFF["G2M"])
    # p15b: x3 = t2g + x2
    radd_phase(ADDR["I4"], 0, B + OFF["t2g"], 0, B + OFF["x2"])
    # p16: y = x3 @ W_head (the final stream of the iteration)
    matmul_phase(ADDR["W_head"], 4, 4, 0, B + OFF["x3"], 4, 4, 0,
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
async def test_tpu_nxn_prog_adaln_n4(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    prog = generate_program_adaln()
    assert len(prog) == 1402, \
        f"program is {len(prog)} words, expected 1402"
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
    # 1168-word body T=3 times.
    dut.rst.value = 0
    await tick(dut)
    dut.run.value = 1
    await tick(dut)
    dut.run.value = 0

    # Emission: 1402 loaded words + 2 extra body passes (2 x 1168);
    # the last body's 40 idles cover the final y stream.
    await tick(dut, 1402 + 2 * BODY_LEN + 40)
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
            for M in (t1g, x2, h2, h2s, h2m, m1, t2g, x3, y):
                expected += col(M, j, 4)
        div = next((k for k in range(min(len(lanes[j]), len(expected)))
                    if lanes[j][k] != expected[k]), None)
        assert lanes[j] == expected, (
            f"VPU lane {j}: {len(lanes[j])} beats, expected "
            f"{len(expected)}; first divergence at beat {div} "
            f"(iteration {div // 84 if j < 2 else div // 76}): got "
            f"{[f'{from_fixed(w):+.4f}' for w in lanes[j][:8]]}..., "
            f"expected {[f'{from_fixed(w):+.4f}' for w in expected[:8]]}"
            f"...")

    # ---- final UB image: all 1888 words exact ----
    for a in range(IMG_WORDS):
        got = nxn.ub_inst.ub_memory[a].value.integer & 0xFFFF
        want = GOLD[a] & 0xFFFF
        assert got == want, (
            f"mem[{a}] = {from_fixed(got):+.4f}, expected "
            f"{from_fixed(want):+.4f}")

    print(f"tpu_nxn_prog N={N} adaLN CAPSTONE OK (1402-word program: "
          f"ONE 1168-word adaLN-conditioned denoiser body — LN, "
          f"scale+shift, 2-head attention, gate+residual, LN, "
          f"scale+shift, SiLU MLP, gate+residual, head projection — "
          f"looped T={T_STEPS} by LOOPI(3, 1168, 320, 320, wbase=928) "
          f"with DISTINCT per-timestep modulation striding through the "
          f"host image; {IMG_WORDS}-word UB image exact)")
