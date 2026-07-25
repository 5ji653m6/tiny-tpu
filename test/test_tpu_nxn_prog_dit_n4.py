"""Gate test for roadmap item 17b2 — THE DiT CAPSTONE: one full
diffusion-transformer denoiser block (pre-LN two-head attention with a
residual, pre-LN SiLU MLP with a residual, final head projection)
iterated T=3 timesteps as ONE loaded program on tpu_nxn_prog, the
weight-shared body executed by a single LOOPI control word with
iteration t reading iteration t-1's output entirely on chip. Written
by the harness author, not the agent — per tinytpu-loop README "the
gate grows with the design".

No new RTL: this composes every gated capability of the machine —
items 5 (N=4 array), 7 (LN/softmax group stages), 9b (loadable
sequencer), 11 (attention composite + exact goldens), 13 (SiLU stage),
14 (multi-head transposed-emission merge), 15 (indexed LOOP), 17a
(residual add via the bias path + wbase) and 17b1 (the 15-bit loop
length). Dims: R=4 tokens, d_model=4, H=2 heads, d_head=2, d_ff=4.

The block (pre-LN, the DiT shape):

    h1 = LN(x)                       p1   (identity matmul + LN pathway)
    Q~_h/K~_h = h1 @ [W_qh|0] / ...  p2-7 (padded 4x4 projections, 4x2 V)
    P_h = softmax(Q~_h @ K~_h^T)     p8/9 (softmax pathway)
    O_h^T = V_h^T @ P_h^T            p10/11 (transposed emission; the
                                     two 2x4 streams stack into
                                     ([O1|O2])^T contiguously)
    x2 = [O1|O2] @ W_O + x           p12  (T-read of the O-stack + bias
                                     pathway + ptr-7 residual read of x)
    h2 = LN(x2)                      p13
    m1 = SiLU(h2 @ W1m)              p14  (SiLU pathway, bit 7)
    x3 = m1 @ W2m + x2               p15  (residual read of x2)
    y  = x3 @ W_head                 p16  (bypass; final stream)

The t-loop (the sampler): all 16 phases form ONE 838-word LOOPI body
(14 plain phases x 52 + 2 residual phases x 55) run by
LOOPI(3, 838, stride_a=224, stride_w=224, wbase=176). Host weights
(176-word image) sit BELOW wbase and are re-read verbatim every
timestep (weight sharing); every intermediate lands at/above wbase in
a 224-word per-iteration append region and the strides advance every
body read by one region per iteration. The linchpin: host x sits at
wbase-16 (@160) and y is the FINAL stream appended each iteration
(B_i + 208), so phase 1's body address 160 advances to
160 + i*224 = exactly iteration (i-1)'s y region — the chip denoises
three timesteps autonomously, recurrence never leaving the UB.

Scoping note (honest): this is the DiT INFERENCE DATAFLOW — the
weight-shared block iterated T times with full on-chip recurrence.
Per-timestep adaLN conditioning (scale/shift from the t embedding) is
NOT in this test: the shift half exists (a bias-pathway add) but the
per-element scale half needs a multiply stage (gap list, post-17).

Host image (176 words = wbase):
  I4 @0 (identity for the LN matmuls), W_qp1 @16 / W_kp1 @32 (4x4
  padded), W_v1 @48 (4x2), W_qp2 @56 / W_kp2 @72 (4x4 padded),
  W_v2 @88 (4x2), W_O @96, W1m @112, W2m @128, W_head @144 (4x4),
  x @160 (4x4 = wbase-16).

Per-iteration append region (E=224 words, base B_i = 176 + i*224):
  h1 @+0, Q~1 @+16, K~1 @+32, V1 @+48 (8w), Q~2 @+56, K~2 @+72,
  V2 @+88 (8w), P1 @+96, P2 @+112, O1^T @+128 (8w), O2^T @+136 (8w),
  x2 @+144, h2 @+160, m1 @+176, x3 @+192, y @+208 (final stream).

Program: 45-word prefix (44 host beats + trailing tick) + the LOOPI
word + the 838-word body = 884 words. PROG_DEPTH=1024, UB_WIDTH=1024;
the final image is 176 + 3*224 = 848 words, final y @832..847.

Golden: hardware-exact integer arithmetic throughout, composed from
the validated models — matmul_g (per-PE-step fxp rounding, item 11/14),
softmax_row_exact (item 11), silu_exact (item 13), fxp_add elementwise
(item 17a) — plus the NEW exact LayerNorm model derived line-by-line
from layernorm_group_nxn.sv + fixedpoint.sv (the item-7a test is
tolerance-based and unusable here where errors compound across 16
phases x 3 timesteps):

  mean    = sum(x) >> 2              (truncating arithmetic shift)
  dev_i   = x_i - mean               (17-bit Q9.8)
  sq_i    = fxp_mul(dev_i, dev_i)    (9.8 x 9.8 -> 18.8, ROUND=1)
  var_eps = (sum(sq) >> 2) + 16      (LN_EPS = 1/16)
  std     = fxp_sqrt(var_eps)        (18.8 -> 8.8, ROUND=1; the RTL's
                                      bit-by-digit loop emulated
                                      mask-exact, no closed form)
  y_i     = fxp_div(dev_i, std)      (SIGNED 9.8 / 8.8 -> 8.8, ROUND=1;
                                      magnitudes, floor quotient, round
                                      up iff 2*rem > divisor (strict),
                                      sign/clamp branches)

Checks (LIVE asserts — PYTHONOPTIMIZE is empty):
  - per-lane VPU streams beat-exact across ALL THREE iterations:
    lanes 0/1 capture 180 beats (60/iteration: 12 full-width streams +
    the 4x2 V_h pair + the two 2-beat O_h^T), lanes 2/3 capture 156;
  - the full 848-word UB image exact everywhere — host regions intact,
    every iteration's 224-word region exact (each iteration's presence
    at its OWN addresses proves the strides advanced: a verbatim LOOP
    would re-emit iteration 0 three times).
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# ProgGen + Q8.8 helpers from the 9b test; exact arithmetic models +
# softmax golden from the item-11 test; matmul/transpose/pad helpers
# from the item-14 test; the residual elementwise add from the 17a
# test; the exact SiLU model from the item-13 test.
from test_tpu_nxn_prog_n4 import ProgGen, to_fixed, from_fixed
from test_tpu_nxn_ic_attn_n4 import softmax_row_exact, fxp_add
from test_tpu_nxn_prog_mh_attn_n4 import (
    matmul_g, to_raw, transpose, pad42,
)
from test_tpu_nxn_prog_radd_n4 import ew_add
from test_silu_parent import silu_exact

N = int(os.environ.get("TPU_NXN_PROG_N", "4"))
WORD_W = 134 + 17 * (N - 2)   # instruction width (item 13: SiLU bit)
CTRL = 1 << WORD_W            # control-word escape bit (prog-word MSB)
IDX_FLAG = 1 << 16

# Pathways (|sm(6)|ln(5)|gelu(4)|bias(3)|lr(2)|loss(1)|lr_d(0)|, silu
# bit 7 = item-13 MSB).
PATHWAY_LN = 0b0100000
PATHWAY_SOFTMAX = 0b1000000
PATHWAY_SILU = 0b10000000
PATHWAY_BIAS = 0b0001000
PATHWAY_BYPASS = 0b0000000
PTR_RESIDUAL = 7

T_STEPS = 3
BODY_LEN = 838              # 14 x 52 + 2 x 55
E_REGION = 224              # per-iteration append size
WBASE = 176                 # host image size; x @ WBASE-16

# ---- stimulus: quarter/eighth multiples, exact in Q8.8, distinct per
# matrix so an addressing mixup reads as a value mismatch ----
X = [[0.50, -0.25, 0.75, 0.00],
     [-0.50, 0.25, 0.00, 0.50],
     [0.25, 0.50, -0.75, 0.25],
     [0.00, -0.50, 0.25, -0.25]]

I4 = [[1.0 if k == j else 0.0 for j in range(4)] for k in range(4)]

# Per-head 4x2 projection weights (d_model=4 -> d_head=2); Q/K padded
# to 4x4 in the image (zero columns exact through per-step rounding).
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

# MLP (d_model -> d_ff=4 -> d_model) and the final head projection.
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


# ---- exact LayerNorm model (layernorm_group_nxn.sv + fixedpoint.sv,
# derived line by line; see the module docstring) ----
def fxp_zoom_rtl(v, WII, WIF, WOI, WOF, ROUND=1):
    """Bit-exact fxp_zoom: v is the WII+WIF-bit unsigned raw input
    (two's-complement pattern). Returns (out, overflow) with out the
    WOI+WOF-bit unsigned raw."""
    assert 0 <= v < (1 << (WII + WIF))
    if WOF < WIF:
        inr = v >> (WIF - WOF)
        if ROUND:
            top_dropped = (v >> (WIF - WOF - 1)) & 1
            msb = (inr >> (WII + WOF - 1)) & 1
            lower_ones = ((1 << (WII + WOF - 1)) - 1)
            if top_dropped and not (msb == 0 and
                                    (inr & lower_ones) == lower_ones):
                inr += 1
            inr &= (1 << (WII + WOF)) - 1
    elif WOF == WIF:
        inr = v
    else:
        inr = v << (WOF - WIF)
    ini = inr >> WOF
    outf = inr & ((1 << WOF) - 1)
    overflow = 0
    if WOI < WII:
        sign = (ini >> (WII - 1)) & 1
        mid = (ini >> (WOI - 1)) & ((1 << (WII - WOI)) - 1)
        if sign == 0 and mid != 0:           # positive overflow
            overflow = 1
            outi = (1 << (WOI - 1)) - 1
            outf = (1 << WOF) - 1
        elif sign == 1 and mid != (1 << (WII - WOI)) - 1:  # negative
            overflow = 1
            outi = 1 << (WOI - 1)
            outf = 0
        else:
            outi = ini & ((1 << WOI) - 1)
        return ((outi << WOF) | outf), overflow
    # WOI >= WII: sign-extend
    outi = ini & ((1 << WOI) - 1)
    if (ini >> (WII - 1)) & 1:
        outi |= ((1 << WOI) - (1 << WII))
    return ((outi << WOF) | outf), 0


def fxp_mul_rtl(a, b, WIIA, WIFA, WIIB, WIFB, WOI, WOF):
    """fxp_mul on signed ints, exact: res = a*b (WRI+WRF two's-
    complement pattern) through res_zoom ROUND=1."""
    WRI, WRF = WIIA + WIIB, WIFA + WIFB
    res = (a * b) & ((1 << (WRI + WRF)) - 1)
    return fxp_zoom_rtl(res, WRI, WRF, WOI, WOF, 1)


def fxp_sqrt_rtl(v, WII=18, WIF=8, WOI=8, WOF=8):
    """fxp_sqrt bit-exact for non-negative v (WII+WIF raw): the RTL's
    digit-by-digit loop emulated mask-exact (no closed form), then the
    res_zoom (ROUND=1). Returns (out_raw, overflow)."""
    WTI = WII + 1 if (WII % 2) == 1 else WII
    WRI = WTI // 2
    inu = v                                  # sign = 0 path
    resu = 0
    resu2 = 0
    for ii in range(WRI - 1, -WIF - 1, -1):
        resu2tmp = resu2 + ((resu << (1 + ii)) if ii >= 0
                            else (resu >> (-1 - ii)))
        if 2 * ii + WIF >= 0:
            resu2tmp += 1 << (2 * ii + WIF)
        if resu2tmp <= inu and inu != 0:
            resu |= 1 << (ii + WIF)
            resu2 = resu2tmp
    resushort = resu & ((1 << (WRI + WIF + 1)) - 1)
    return fxp_zoom_rtl(resushort, WRI + 1, WIF, WOI, WOF, 1)


def fxp_div_signed_rtl(dev, std):
    """fxp_div(9.8 / 8.8 -> 8.8, ROUND=1) bit-exact. dev a signed int
    (17-bit Q9.8), std a positive int (16-bit Q8.8 raw). Returns the
    signed 16-bit result. The greedy bit loop over nested weights is
    exactly the floor quotient; the rounding adds one output LSB and
    flips iff the remainder beats half a divisor (strict)."""
    assert 0 <= abs(dev) < (1 << 15), "dev outside the clean zoom domain"
    assert std > 0
    sign = dev < 0
    a = -dev if dev < 0 else dev
    q = (a << 8) // std
    if q > 0xFFFF:
        q = 0xFFFF                           # loop accepts every bit
    else:
        rem = (a << 8) - q * std
        if 2 * rem > std:                    # ~(&out) holds: q < 0xFFFF
            q += 1
    if sign:
        out = 0x8000 if (q & 0x8000) else ((-q) & 0xFFFF)
    else:
        out = 0x7FFF if (q & 0x8000) else q
    return out - 0x10000 if out & 0x8000 else out


def ln_row_exact(row):
    """One N=4 group-LN beat, bit-exact vs layernorm_group_nxn.sv.
    row: 4 signed Q8.8 raws (one output row = the N lanes of one
    beat). Returns 4 signed Q8.8 raws."""
    mean = sum(row) >> 2                     # sum_ext[17:2], truncating
    devs = [x - mean for x in row]           # 17-bit Q9.8
    sqs = [fxp_mul_rtl(d, d, 9, 8, 9, 8, 18, 8)[0] for d in devs]
    var_eps = (sum(sqs) >> 2) + 16           # LN_EPS = 1/16
    std, _ = fxp_sqrt_rtl(var_eps, 18, 8, 8, 8)
    return [fxp_div_signed_rtl(d, std) for d in devs]


# ---- golden: the whole 16-phase block x T_STEPS, raw Q8.8 ----
def dit_iteration(x_in):
    """One denoiser block on a 4x4 raw matrix. Returns the phase
    outputs in stream order (h1, Qt1, Kt1, V1, Qt2, Kt2, V2, P1, P2,
    O1T, O2T, x2, h2, m1, x3, y)."""
    h1 = [ln_row_exact(row) for row in x_in]
    Qt1 = matmul_g(h1, Wq1pr)
    Kt1 = matmul_g(h1, Wk1pr)
    V1 = matmul_g(h1, Wv1r)                  # 4x2
    Qt2 = matmul_g(h1, Wq2pr)
    Kt2 = matmul_g(h1, Wk2pr)
    V2 = matmul_g(h1, Wv2r)                  # 4x2
    P1 = [softmax_row_exact(row)
          for row in matmul_g(Qt1, transpose(Kt1))]
    P2 = [softmax_row_exact(row)
          for row in matmul_g(Qt2, transpose(Kt2))]
    O1T = matmul_g(transpose(V1), transpose(P1))   # 2x4
    O2T = matmul_g(transpose(V2), transpose(P2))   # 2x4
    Cm = [[O1T[0][r], O1T[1][r], O2T[0][r], O2T[1][r]] for r in range(4)]
    x2 = ew_add(matmul_g(Cm, W_Or), x_in)    # residual: the phase input
    h2 = [ln_row_exact(row) for row in x2]
    m1 = [[silu_exact(v) for v in row] for row in matmul_g(h2, W1mr)]
    x3 = ew_add(matmul_g(m1, W2mr), x2)      # residual: x2
    y = matmul_g(x3, W_headr)
    return (h1, Qt1, Kt1, V1, Qt2, Kt2, V2, P1, P2,
            O1T, O2T, x2, h2, m1, x3, y)


ITERS = []
_x = Xr
for _t in range(T_STEPS):
    _out = dit_iteration(_x)
    ITERS.append(_out)
    _x = _out[-1]                            # y_{t} is t+1's input

# ---- UB image (848 words) ----
ADDR = {"I4": 0, "Wqp1": 16, "Wkp1": 32, "Wv1": 48,
        "Wqp2": 56, "Wkp2": 72, "Wv2": 88, "W_O": 96,
        "W1m": 112, "W2m": 128, "W_head": 144, "x": 160}
OFF = {"h1": 0, "Qt1": 16, "Kt1": 32, "V1": 48,
       "Qt2": 56, "Kt2": 72, "V2": 88, "P1": 96, "P2": 112,
       "O1T": 128, "O2T": 136, "x2": 144, "h2": 160,
       "m1": 176, "x3": 192, "y": 208}
GOLD = {}


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
    _base = WBASE + _t * E_REGION
    (_h1, _Qt1, _Kt1, _V1, _Qt2, _Kt2, _V2, _P1, _P2,
     _O1T, _O2T, _x2, _h2, _m1, _x3, _y) = ITERS[_t]
    for _name, _M, _rows, _cols in (
            ("h1", _h1, 4, 4), ("Qt1", _Qt1, 4, 4), ("Kt1", _Kt1, 4, 4),
            ("V1", _V1, 4, 2), ("Qt2", _Qt2, 4, 4), ("Kt2", _Kt2, 4, 4),
            ("V2", _V2, 4, 2), ("P1", _P1, 4, 4), ("P2", _P2, 4, 4),
            ("O1T", _O1T, 2, 4), ("O2T", _O2T, 2, 4), ("x2", _x2, 4, 4),
            ("h2", _h2, 4, 4), ("m1", _m1, 4, 4), ("x3", _x3, 4, 4),
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


def generate_program_dit():
    """884 words: 45-word host prefix + LOOPI(3, 838, sa=224, sw=224,
    wbase=176) + the 838-word body (16 phases: 14 plain x 52 + 2
    residual x 55). Body addresses are the iteration-0 absolute
    addresses; the strides advance every intermediate-region read by
    E=224 per iteration while host-weight reads (< wbase) freeze."""
    g = ProgGen()
    g.idle()
    host_words = []
    for M in (I4, pad42(Wq1), pad42(Wk1), Wv1,
              pad42(Wq2), pad42(Wk2), Wv2, W_O, W1m, W2m, W_head, X):
        host_words += [to_fixed(v) for row in M for v in row]
    assert len(host_words) == WBASE and len(host_words) % N == 0
    for b in range(len(host_words) // N):
        g.write_beat([host_words[N * b + (N - 1 - i)] for i in range(N)])
    g.tick()  # trailing idle cycle after the last beat
    assert len(g.prog) == 45, f"prefix is {len(g.prog)} words, expected 45"

    def matmul_phase(w_addr, w_rows, w_cols, w_T, a_addr, a_rows, a_cols,
                     a_T, pathway):
        g.issue_read(ptr=1, addr=w_addr, rows=w_rows, cols=w_cols,
                     transpose=w_T)
        g.tick(9)
        g.issue_read(ptr=0, addr=a_addr, rows=a_rows, cols=a_cols,
                     transpose=a_T, pathway=pathway)
        g.switch_pulse()
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

    # p1: h1 = LN(x) — identity matmul + the LN group stage
    matmul_phase(ADDR["I4"], 4, 4, 0, ADDR["x"], 4, 4, 0, PATHWAY_LN)
    # p2-7: per-head Q/K/V projections of h1 (bypass)
    matmul_phase(ADDR["Wqp1"], 4, 4, 0, B + OFF["h1"], 4, 4, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wkp1"], 4, 4, 0, B + OFF["h1"], 4, 4, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wv1"], 4, 2, 0, B + OFF["h1"], 4, 4, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wqp2"], 4, 4, 0, B + OFF["h1"], 4, 4, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wkp2"], 4, 4, 0, B + OFF["h1"], 4, 4, 0,
                 PATHWAY_BYPASS)
    matmul_phase(ADDR["Wv2"], 4, 2, 0, B + OFF["h1"], 4, 4, 0,
                 PATHWAY_BYPASS)
    # p8/9: P_h = softmax(Q~_h @ K~_h^T)
    matmul_phase(B + OFF["Kt1"], 4, 4, 1, B + OFF["Qt1"], 4, 4, 0,
                 PATHWAY_SOFTMAX)
    matmul_phase(B + OFF["Kt2"], 4, 4, 1, B + OFF["Qt2"], 4, 4, 0,
                 PATHWAY_SOFTMAX)
    # p10/11: O_h^T = V_h^T @ P_h^T (2x4; streams stack into
    # ([O1|O2])^T contiguously — the item-14 merge)
    matmul_phase(B + OFF["P1"], 4, 4, 1, B + OFF["V1"], 4, 2, 1,
                 PATHWAY_BYPASS)
    matmul_phase(B + OFF["P2"], 4, 4, 1, B + OFF["V2"], 4, 2, 1,
                 PATHWAY_BYPASS)
    # p12: x2 = [O1|O2] @ W_O + x — the T-read presents the concat;
    # the residual read (ptr-7) streams x elementwise into the bias
    # stage (the phase input x, NOT h1)
    radd_phase(ADDR["W_O"], 0, B + OFF["O1T"], 1, ADDR["x"])
    # p13: h2 = LN(x2)
    matmul_phase(ADDR["I4"], 4, 4, 0, B + OFF["x2"], 4, 4, 0, PATHWAY_LN)
    # p14: m1 = SiLU(h2 @ W1m)
    matmul_phase(ADDR["W1m"], 4, 4, 0, B + OFF["h2"], 4, 4, 0,
                 PATHWAY_SILU)
    # p15: x3 = m1 @ W2m + x2
    radd_phase(ADDR["W2m"], 0, B + OFF["m1"], 0, B + OFF["x2"])
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
async def test_tpu_nxn_prog_dit_n4(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    prog = generate_program_dit()
    assert len(prog) == 884, f"program is {len(prog)} words, expected 884"
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
    # 838-word body T=3 times.
    dut.rst.value = 0
    await tick(dut)
    dut.run.value = 1
    await tick(dut)
    dut.run.value = 0

    # Emission: 884 loaded words + 2 extra body passes (2 x 838) = 2520
    # cycles; the last body's 40 idles cover the final y stream.
    await tick(dut, 884 + 2 * BODY_LEN + 40)
    collector.kill()

    # ---- per-lane streams, beat-exact across all three iterations ----
    def col(M, j, rows):
        return [M[r][j] & 0xFFFF for r in range(rows)]

    for j in range(N):
        expected = []
        for t in range(T_STEPS):
            (h1, Qt1, Kt1, V1, Qt2, Kt2, V2, P1, P2,
             O1T, O2T, x2, h2, m1, x3, y) = ITERS[t]
            for M in (h1, Qt1, Kt1):
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
            for M in (x2, h2, m1, x3, y):
                expected += col(M, j, 4)
        div = next((k for k in range(min(len(lanes[j]), len(expected)))
                    if lanes[j][k] != expected[k]), None)
        assert lanes[j] == expected, (
            f"VPU lane {j}: {len(lanes[j])} beats, expected "
            f"{len(expected)}; first divergence at beat {div} "
            f"(iteration {div // 60 if j < 2 else div // 52}): got "
            f"{[f'{from_fixed(w):+.4f}' for w in lanes[j][:8]]}..., "
            f"expected {[f'{from_fixed(w):+.4f}' for w in expected[:8]]}"
            f"...")

    # ---- final UB image: all 848 words exact ----
    for a in range(IMG_WORDS):
        got = nxn.ub_inst.ub_memory[a].value.integer & 0xFFFF
        want = GOLD[a] & 0xFFFF
        assert got == want, (
            f"mem[{a}] = {from_fixed(got):+.4f}, expected "
            f"{from_fixed(want):+.4f}")

    print(f"tpu_nxn_prog N={N} DiT CAPSTONE OK (884-word program: ONE "
          f"838-word denoiser-block body — LN, 2-head attention, "
          f"residual, LN, SiLU MLP, residual, head projection — "
          f"looped T={T_STEPS} by LOOPI(3, 838, 224, 224, wbase=176), "
          f"iteration t reading t-1's y entirely on chip; "
          f"{IMG_WORDS}-word UB image exact)")
