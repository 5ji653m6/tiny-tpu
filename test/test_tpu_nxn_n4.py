"""Gate test for src/tpu_nxn.sv (roadmap item 5d-2), N=4 full-chip
forward-pass half. Written by the harness author, not the agent — per
tinytpu-loop README "the gate grows with the design".

One 4x4 forward matmul end to end through the parameterized top:
host writes X (4x4 row-major @ 0), W (4x4 row-major @ 16) and B (4
biases @ 32) into the UB at N=4 (beat b lane i = word 4*b + (3-i),
BUG-UB-2 decrementing write loop); W is loaded into the array
TRANSPOSED (so the array computes Z = X @ W.T, matching the 2x2
test_tpu.py precedent); X streams untransposed (lane i = column i of
X, top row first — the 5d-1b semantics); the VPU runs the forward
pathway (bias + leaky_relu, pathway = 0b0001100) and writes H back to
the UB.

All stimulus values are exact dyadic rationals (multiples of 0.5 in
the inputs; all sums/products exact in Q8.8), so the numpy golden is
exact regardless of the MAC's truncation direction. leak = 0.5.

Assertions (LIVE — PYTHONOPTIMIZE is empty):
  * per-lane VPU output streams: lane j emits H[k][j] for k = 0..3 in
    beat order (valid-qualified collect at the VPU -> UB interface)
  * the 16 H words land in the UB region immediately after the last
    host-written word (multiset check — the skew makes the exact
    addressing an anti-diagonal wavefront, covered by the stream check)
"""

import os

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

N = int(os.environ.get("TPU_NXN_N", "4"))
FRAC = 8

# Instance/wire paths inside tpu_nxn (finalized against the agent's RTL
# after the loop — see placeholder resolution in capture()).
NXN = "tpu_nxn_inst"

# ---- stimulus (all values exact in Q8.8) ----
X = np.array([[1.0, 0.5, 0.0, -1.0],
              [0.0, 1.0, -0.5, 0.5],
              [-0.5, 0.0, 1.0, 1.0],
              [0.5, -1.0, 0.5, 0.0]])
W = np.array([[0.5, -0.5, 1.0, 0.0],
              [1.0, 0.5, -0.5, 0.5],
              [0.0, 1.0, 0.5, -1.0],
              [-0.5, 0.0, 1.0, 0.5]])
B = np.array([0.25, -0.5, 0.5, -0.25])
LEAK = 0.5

# Golden: Z = X @ W.T (transposed weight load), H = leaky_relu(Z + B)
Z = X @ W.T
H = np.where(Z + B >= 0, Z + B, (Z + B) * LEAK)


def to_fixed(val):
    return int(round(val * (1 << FRAC))) & 0xFFFF


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


async def issue_read(dut, ptr, addr, rows, cols, transpose=0, pathway=None,
                     switch=None):
    dut.ub_rd_start_in.value = 1
    dut.ub_ptr_select.value = ptr
    dut.ub_rd_addr_in.value = addr
    dut.ub_rd_row_size.value = rows
    dut.ub_rd_col_size.value = cols
    dut.ub_rd_transpose.value = transpose
    if pathway is not None:
        dut.vpu_data_pathway.value = pathway
    if switch is not None:
        dut.sys_switch_in.value = switch
    await tick(dut)
    dut.ub_rd_start_in.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_row_size.value = 0
    dut.ub_rd_col_size.value = 0
    dut.ub_rd_transpose.value = 0


@cocotb.test()
async def test_tpu_nxn_forward_n4(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    nxn = dut.tpu_nxn_inst

    dut.rst.value = 1
    await drive_idle(dut)
    dut.learning_rate_in.value = to_fixed(0.75)
    dut.vpu_data_pathway.value = 0
    dut.vpu_leak_factor_in.value = to_fixed(LEAK)
    dut.inv_batch_size_times_two_in.value = to_fixed(0.5)
    await tick(dut, 2)
    dut.rst.value = 0
    await tick(dut)

    # Per-lane collector on the VPU -> UB write interface (internal
    # wires of tpu_nxn; hierarchical access).
    lanes = [[] for _ in range(N)]

    async def collect():
        while True:
            await RisingEdge(dut.clk)
            for i in range(N):
                if nxn.ub_wr_valid_in[i].value.integer:
                    lanes[i].append(nxn.ub_wr_data_in[i].value.integer
                                    & 0xFFFF)

    collector = cocotb.start_soon(collect())

    # Host writes: X @ 0..15, W @ 16..31, B @ 32..35 (36 words).
    words = ([to_fixed(v) for v in X.flatten()]
             + [to_fixed(v) for v in W.flatten()]
             + [to_fixed(v) for v in B])
    await host_write_words(dut, words)

    # Load W into the array (transposed): the N=4 weight walk lives
    # R+C-1 = 7 cycles, so sys_switch must wait for it to complete.
    await issue_read(dut, ptr=1, addr=16, rows=4, cols=4, transpose=1)
    await tick(dut, 9)

    # Stream X with the forward pathway; switch the shadow weights into
    # the active buffers one cycle after the input read starts (the
    # switch wavefront reaches every PE ahead of the first data beat).
    await issue_read(dut, ptr=0, addr=0, rows=4, cols=4,
                     pathway=0b0001100)
    dut.sys_switch_in.value = 1
    await tick(dut)

    # Read biases while the array computes. The bias walk must land its
    # per-lane windows exactly on the systolic output beats (the VPU's
    # bias_child pairs them combinationally per valid beat); at N=4 the
    # array latency is N-dependent, so the bias read goes out 2 cycles
    # later than the N=2 script's offset (verified by cycle-level trace).
    dut.sys_switch_in.value = 0
    await tick(dut, 2)
    await issue_read(dut, ptr=2, addr=32, rows=4, cols=4)

    await tick(dut, 80)
    collector.kill()

    # ---- per-lane stream check: lane j emits H[k][j], k = 0..3 ----
    for j in range(N):
        expected = [to_fixed(H[k][j]) for k in range(4)]
        assert lanes[j] == expected, (
            f"VPU lane {j}: got "
            f"{[f'{from_fixed(w):+.4f}' for w in lanes[j]]}, expected "
            f"{[f'{from_fixed(w):+.4f}' for w in expected]} "
            f"(H column {j})")

    # ---- placement check: 16 H words sit right after the host image --
    region = [nxn.ub_inst.ub_sram.mem[a].value.integer & 0xFFFF
              for a in range(36, 52)]
    assert sorted(region) == sorted(to_fixed(v) for v in H.flatten()), (
        f"UB[36:52] = {[f'{from_fixed(w):+.4f}' for w in region]}, "
        f"expected the 16 H values "
        f"{[f'{v:+.4f}' for v in sorted(H.flatten())]}")

    print(f"tpu_nxn N={N} full-chip forward pass OK "
          f"(4x4 matmul + bias + leaky_relu; per-lane streams and UB "
          f"placement verified)")
