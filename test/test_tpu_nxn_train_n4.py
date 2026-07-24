"""Gate test for src/tpu_nxn.sv (roadmap item 6): a full end-to-end
TRAINING STEP at N=4 — forward pass, loss, backprop, and in-place
gradient-descent updates of both weights and biases, checked against an
exact numpy golden. Written by the harness author, not the agent — per
tinytpu-loop README "the gate grows with the design".

Single-layer network so no tiling is needed at N=4:
  pass 1 (transition pathway 0b0001111): X streams through the array
    with W^T loaded; the VPU applies bias -> leaky_relu -> loss ->
    leaky_relu_derivative and emits dZ = (H - Y)*(2/batch)*lr'(H),
    where Z = X @ W.T and H = leaky(Z + B). A ptr-5 read pairs the old
    biases with the dZ beats so the gradient_descent modules accumulate
    B -= lr * sum_batch dZ IN PLACE.
  pass 2 (bypass pathway 0b0000000): X is loaded to the TOP of the
    array (ptr 1, untransposed) and dZ^T streams from the LEFT (ptr 0,
    transposed), so the array emits G = dZ^T @ X = dL/dW. A ptr-6 read
    walks the old weights against the G beats and the gradient_descent
    modules apply W -= lr * G IN PLACE.

UB layout (row-major; host beat b lane i = word 4b + (3-i), BUG-UB-2):
    0-15  X (4x4)   16-31 W (4x4)   32-35 B   36-51 Y (4x4)
   52-67  dZ (pass-1 VPU output)    68-83 G (pass-2 VPU output)
W and B hold their UPDATED values at the end.

Depends on BUG-UB-3 (row-major VPU-stream writes and row-major
weight-gradient writeback in unified_buffer_nxn.sv): the legacy
arrival-order writeback only coincided with row-major at N=2.

Choreography (all N-dependent offsets trace-verified at N=4):
  * weight walk wait of 9 cycles after each ptr-1 load (R+C-1 = 7 + margin)
  * B read 2 cycles after switch deassert (5d-2 forward choreography)
  * Y read 1 cycle after the B read (the loss stage's H beats pair one
    cycle earlier than the N=2 +2 spacing predicts)
  * ptr-5 bias-grad read 2 cycles after the Y read: first dZ beat lands
    at capture-window counter 1 (after the value_old load at counter i),
    last beat at counter 7 < row+col — full capture, no clip
  * ptr-6 weight-grad read 3 cycles after pass-2 switch deassert (1
    later than the N=2-relative offset): same counter-1..7 placement

Golden is exact: all stimulus is multiples of 0.5 and
lr = leak = inv_batch_size_times_two = 0.5, so every Q8.8 product stays
within 8 fractional bits (asserted by to_fixed) — fxp_mul rounding
never triggers.

Assertions (LIVE — PYTHONOPTIMIZE is empty):
  * per-lane dZ stream: lane j emits dZ[k][j] for k = 0..3 in beat order
  * per-lane G stream: lane j emits G[k][j] for k = 0..3 in beat order
  * final UB image: X and Y untouched; W, B updated in place; dZ and G
    regions row-major — all exact
"""

import os

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

N = int(os.environ.get("TPU_NXN_N", "4"))
FRAC = 8

# ---- stimulus (all values exact in Q8.8) ----
X = np.array([[1.0, 0.5, 0.0, -0.5],
              [0.5, -1.0, 0.5, 1.0],
              [-0.5, 0.5, 1.0, 0.0],
              [0.0, 1.0, -0.5, 0.5]])
W = np.array([[0.5, -0.5, 1.0, 0.5],
              [1.0, 0.5, -0.5, 0.0],
              [0.0, 1.0, 0.5, -1.0],
              [-0.5, 0.0, 1.0, 0.5]])
B = np.array([0.5, -0.5, 0.5, -0.5])
Y = np.array([[1.0, -0.5, 0.5, 0.0],
              [0.5, 1.0, -0.5, 0.5],
              [-0.5, 0.0, 1.0, 0.5],
              [0.5, -1.0, 0.0, 1.0]])
LR = 0.5
LEAK = 0.5
INV2B = 0.5  # 2/batch, batch = 4

# ---- golden model ----
Z = X @ W.T
Hpre = Z + B
H = np.where(Hpre >= 0, Hpre, Hpre * LEAK)
dZ = np.where(H >= 0, (H - Y) * INV2B, (H - Y) * INV2B * LEAK)
B_new = B - LR * dZ.sum(axis=0)
G = dZ.T @ X
W_new = W - LR * G

GOLD = {}
for r in range(4):
    for c in range(4):
        GOLD[0 + 4 * r + c] = X[r][c]
        GOLD[16 + 4 * r + c] = W_new[r][c]
        GOLD[36 + 4 * r + c] = Y[r][c]
        GOLD[52 + 4 * r + c] = dZ[r][c]
        GOLD[68 + 4 * r + c] = G[r][c]
for i in range(4):
    GOLD[32 + i] = B_new[i]


def to_fixed(val):
    """Exact Q8.8 encoding; asserts the value is representable."""
    scaled = val * (1 << FRAC)
    assert scaled == int(scaled), f"{val} not exact in Q8.8"
    return int(scaled) & 0xFFFF


def from_fixed(word):
    w = word & 0xFFFF
    v = w - 0x10000 if w & 0x8000 else w
    return v / (1 << FRAC)


async def tick(dut, cycles=1):
    for _ in range(cycles):
        await RisingEdge(dut.clk)


async def drive_idle(dut):
    for i in range(N):
        dut.ub_wr_host_data_in[i].value = 0
        dut.ub_wr_host_valid_in[i].value = 0
    dut.ub_rd_start_in.value = 0
    dut.ub_rd_transpose.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_row_size.value = 0
    dut.ub_rd_col_size.value = 0
    dut.sys_switch_in.value = 0


async def host_write_words(dut, words):
    """words: list of Q8.8 ints; beat b lane i = words[4b + (3-i)]."""
    assert len(words) % N == 0
    for b in range(len(words) // N):
        for i in range(N):
            dut.ub_wr_host_data_in[i].value = words[N * b + (N - 1 - i)]
            dut.ub_wr_host_valid_in[i].value = 1
        await tick(dut)
    for i in range(N):
        dut.ub_wr_host_data_in[i].value = 0
        dut.ub_wr_host_valid_in[i].value = 0
    await tick(dut)


async def issue_read(dut, ptr, addr, rows, cols, transpose=0, pathway=None):
    dut.ub_rd_start_in.value = 1
    dut.ub_ptr_select.value = ptr
    dut.ub_rd_addr_in.value = addr
    dut.ub_rd_row_size.value = rows
    dut.ub_rd_col_size.value = cols
    dut.ub_rd_transpose.value = transpose
    if pathway is not None:
        dut.vpu_data_pathway.value = pathway
    await tick(dut)
    dut.ub_rd_start_in.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_row_size.value = 0
    dut.ub_rd_col_size.value = 0
    dut.ub_rd_transpose.value = 0


@cocotb.test()
async def test_tpu_nxn_train_n4(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    nxn = dut.tpu_nxn_inst

    dut.rst.value = 1
    await drive_idle(dut)
    dut.learning_rate_in.value = to_fixed(LR)
    dut.vpu_data_pathway.value = 0
    dut.vpu_leak_factor_in.value = to_fixed(LEAK)
    dut.inv_batch_size_times_two_in.value = to_fixed(INV2B)
    await tick(dut, 2)
    dut.rst.value = 0
    await tick(dut)

    # Per-lane collectors on the VPU -> UB write interface. The two
    # passes are disjoint in time, so one pair of collectors covers
    # both: dZ beats arrive during pass 1, G beats during pass 2.
    lanes = [[] for _ in range(N)]

    async def collect():
        while True:
            await RisingEdge(dut.clk)
            for i in range(N):
                if nxn.ub_wr_valid_in[i].value.integer:
                    lanes[i].append(nxn.ub_wr_data_in[i].value.integer
                                    & 0xFFFF)

    collector = cocotb.start_soon(collect())

    # Host image: X @ 0, W @ 16, B @ 32, Y @ 36 (52 words).
    words = ([to_fixed(v) for v in X.flatten()]
             + [to_fixed(v) for v in W.flatten()]
             + [to_fixed(v) for v in B]
             + [to_fixed(v) for v in Y.flatten()])
    await host_write_words(dut, words)

    # ---- pass 1: forward + loss (transition pathway) ----
    await issue_read(dut, ptr=1, addr=16, rows=4, cols=4, transpose=1)
    await tick(dut, 9)  # N=4 weight walk = R+C-1 = 7 cycles + margin

    await issue_read(dut, ptr=0, addr=0, rows=4, cols=4,
                     pathway=0b0001111)
    dut.sys_switch_in.value = 1
    await tick(dut)

    # B read: 2 cycles after switch deassert (5d-2 N=4 choreography).
    dut.sys_switch_in.value = 0
    await tick(dut, 2)
    await issue_read(dut, ptr=2, addr=32, rows=4, cols=4)

    # Y read: 1 cycle after the B read (trace-verified pairing with the
    # loss stage's H beats).
    await tick(dut, 1)
    await issue_read(dut, ptr=3, addr=36, rows=4, cols=4)

    # bias-gradient read: first dZ beat at capture counter 1 (after the
    # value_old load), last beat at counter 7 < row+col — no clip.
    await tick(dut, 2)
    await issue_read(dut, ptr=5, addr=32, rows=4, cols=4)

    await tick(dut, 40)  # dZ lands @52-67, B updated @32-35

    # ---- pass 2: weight gradient G = dZ^T @ X ----
    await issue_read(dut, ptr=1, addr=0, rows=4, cols=4, transpose=0)
    await tick(dut, 9)  # X "weight" walk to the top of the array

    await issue_read(dut, ptr=0, addr=52, rows=4, cols=4, transpose=1,
                     pathway=0b0000000)
    dut.sys_switch_in.value = 1
    await tick(dut)

    # weight-gradient read: 3 cycles after switch deassert so the G
    # beats span capture counters 1..7 (full capture, no clip).
    dut.sys_switch_in.value = 0
    await tick(dut, 3)
    await issue_read(dut, ptr=6, addr=16, rows=4, cols=4)

    await tick(dut, 60)
    collector.kill()

    # ---- per-lane stream checks ----
    # Pass 1 emitted dZ (4 beats/lane), pass 2 emitted G (4 more).
    for j in range(N):
        got_dz = lanes[j][0:4]
        expected_dz = [to_fixed(dZ[k][j]) for k in range(4)]
        assert got_dz == expected_dz, (
            f"VPU lane {j} pass 1: got "
            f"{[f'{from_fixed(w):+.4f}' for w in got_dz]}, expected "
            f"{[f'{from_fixed(w):+.4f}' for w in expected_dz]} "
            f"(dZ column {j})")
        got_g = lanes[j][4:8]
        expected_g = [to_fixed(G[k][j]) for k in range(4)]
        assert got_g == expected_g, (
            f"VPU lane {j} pass 2: got "
            f"{[f'{from_fixed(w):+.4f}' for w in got_g]}, expected "
            f"{[f'{from_fixed(w):+.4f}' for w in expected_g]} "
            f"(G column {j})")

    # ---- final UB image: exact match in every region ----
    for base, count, name in [(0, 16, "X"), (16, 16, "W'"), (32, 4, "B'"),
                              (36, 16, "Y"), (52, 16, "dZ"),
                              (68, 16, "G")]:
        for a in range(base, base + count):
            got = nxn.ub_inst.ub_memory[a].value.integer & 0xFFFF
            want = to_fixed(GOLD[a])
            assert got == want, (
                f"{name} region: mem[{a}] = {from_fixed(got):+.4f}, "
                f"expected {GOLD[a]:+.4f}")

    print(f"tpu_nxn N={N} end-to-end training step OK "
          f"(forward+loss, dZ, bias update, weight update; "
          f"per-lane streams and full UB image verified)")
