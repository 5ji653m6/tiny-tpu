"""Gate test for the agent-authored parameterized N×N systolic array
(src/systolic_nxn.sv, roadmap item 5a). Written by the harness author,
not the agent — per tinytpu-loop README "the gate grows with the design".

Spec under test (from tpu_loop.py TASKS["nxn_systolic"]):
  * module systolic_nxn, parameter int SYSTOLIC_ARRAY_WIDTH (the Makefile
    recipe overrides it with iverilog -P and exports N via SYSTOLIC_NXN_N)
  * unpacked-array ports: sys_data_in/sys_weight_in/sys_accept_w/
    sys_data_out/sys_valid_out [SYSTOLIC_ARRAY_WIDTH]
  * weight-stationary dataflow identical to the legacy 2×2: input flows
    right, psums flow down, weights stream down BOTTOM-ROW-FIRST with
    per-column accept broadcast, sys_switch_in copies shadow->active
  * valid CHAINS through the array (right along row 1, then down each
    column) from the single sys_start_1 — so every column's bottom PE
    fires even when only input-matrix column 1 streams (BUG-SYS-1)
  * pe_enabled per column: all-enabled out of reset; col_size k enables
    exactly columns 1..k

Protocol (mirrors test/test_systolic.py's drive of the legacy array):
  * weight column c streams at cycles [c, c+N) (1-indexed, staggered),
    beats W[c-1][N-1] .. W[c-1][0] (bottom row first)
  * switch pulses with the first input beat; array row r streams input
    matrix COLUMN r-1 (X[0][r-1], X[1][r-1], ...) starting one cycle
    after row r-1
  * output lane c emits Y[r][c-1] = X[r] . W[c-1] for r = 0..rows-1
    (Y = X @ W^T), one beat/cycle, lane c leading lane c+1 by one cycle
    (systolic wavefront — the same 1-cycle lane skew documented in
    test_tpu_softmax.py); beats are collected PER LANE and paired by
    index.

All operands are small INTEGERS in Q8.8, so every product and partial
sum is exact — expected outputs are integer-equal, no tolerance. An
identity-weight pass runs first (catches wiring transpositions), then a
distinct-values pass (catches everything else).

Unlike the upstream tests, the asserts here are LIVE — PYTHONOPTIMIZE is
empty (NOASSERT is never defined), so they fire.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

N = int(os.environ.get("SYSTOLIC_NXN_N", "2"))
ROWS = 4          # input matrix rows (beats streamed per array row)

FRAC_BITS = 8


def to_fixed(val, frac_bits=FRAC_BITS):
    """convert python float to signed 16-bit fixed-point (Q8.8)."""
    scaled = int(round(val * (1 << frac_bits)))
    return scaled & 0xFFFF


def from_fixed(val, frac_bits=FRAC_BITS):
    """convert signed 16-bit fixed-point to python float."""
    if val >= 1 << 15:
        val -= 1 << 16
    return float(val) / (1 << frac_bits)


def s16(v):
    """interpret a 16-bit cocotb value as signed."""
    return v - (1 << 16) if v >= 1 << 15 else v


async def collect_beats(dut, lanes):
    """Sample every output lane every cycle (cocotb/Icarus reads right
    after `await RisingEdge` see the pre-edge state, so every cycle must
    be sampled). lanes: list of per-lane beat lists (signed raw Q8.8)."""
    while True:
        await RisingEdge(dut.clk)
        for c in range(N):
            if dut.sys_valid_out[c].value.integer:
                lanes[c].append(s16(dut.sys_data_out[c].value.integer))


async def drive_idle(dut):
    for c in range(N):
        dut.sys_data_in[c].value = 0
        dut.sys_weight_in[c].value = 0
        dut.sys_accept_w[c].value = 0
    dut.sys_start_1.value = 0
    dut.sys_switch_in.value = 0
    dut.ub_rd_col_size_in.value = 0
    dut.ub_rd_col_size_valid_in.value = 0


async def load_weights(dut, W):
    """Stream the N weight columns, bottom row first, column c staggered
    one cycle behind column c-1 (legacy protocol)."""
    for step in range(1, 2 * N):
        for c in range(1, N + 1):
            beat = step - c
            if 0 <= beat < N:
                dut.sys_weight_in[c - 1].value = to_fixed(W[c - 1][N - 1 - beat])
                dut.sys_accept_w[c - 1].value = 1
            else:
                dut.sys_accept_w[c - 1].value = 0
        await RisingEdge(dut.clk)
    for c in range(N):
        dut.sys_accept_w[c].value = 0


async def stream_inputs(dut, X, rows, col_stream_count):
    """Switch weights to active and stream the input matrix: array row r
    carries X column r-1, staggered one cycle per row. Only the first
    col_stream_count array rows stream (single/dual-column input matrices
    exercise the BUG-SYS-1 valid chain)."""
    for i in range(rows + N):
        beat_row1 = i
        if beat_row1 < rows:
            dut.sys_data_in[0].value = to_fixed(X[beat_row1][0])
            dut.sys_start_1.value = 1
        else:
            dut.sys_start_1.value = 0
        for r in range(2, col_stream_count + 1):
            beat = i - (r - 1)
            if 0 <= beat < rows:
                dut.sys_data_in[r - 1].value = to_fixed(X[beat][r - 1])
        dut.sys_switch_in.value = 1 if i == 0 else 0
        await RisingEdge(dut.clk)
    dut.sys_start_1.value = 0
    dut.sys_switch_in.value = 0
    for c in range(N):
        dut.sys_data_in[c].value = 0


async def set_col_size(dut, k):
    dut.ub_rd_col_size_in.value = k
    dut.ub_rd_col_size_valid_in.value = 1
    await RisingEdge(dut.clk)
    dut.ub_rd_col_size_valid_in.value = 0
    dut.ub_rd_col_size_in.value = 0


async def fresh_start(dut):
    """Reset, idle all drivers, start the beat collector."""
    dut.rst.value = 1
    await drive_idle(dut)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    lanes = [[] for _ in range(N)]
    collector = cocotb.start_soon(collect_beats(dut, lanes))
    return lanes, collector


async def finish(dut, collector):
    for _ in range(3 * N + 8):
        await RisingEdge(dut.clk)
    collector.kill()


def expected_y(X, W, rows, ncols):
    """Y[r][c] = X[r] . W[c] (Y = X @ W^T), exact integers."""
    return [[sum(X[r][k] * W[c][k] for k in range(N))
             for c in range(ncols)] for r in range(rows)]


@cocotb.test()
async def test_nxn_matmul(dut):
    """Two full matmuls at parameter N: identity weights, then distinct
    small integers. Exact integer equality per lane, paired by index."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    X = [[(r + k) % 3 - 1 for k in range(N)] for r in range(ROWS)]

    # Pass 1: identity weights — catches wiring transpositions outright.
    W_id = [[1 if k == c else 0 for k in range(N)] for c in range(N)]
    lanes, collector = await fresh_start(dut)
    await load_weights(dut, W_id)
    await stream_inputs(dut, X, ROWS, N)
    await finish(dut, collector)
    for c in range(N):
        got = lanes[c]
        exp = [X[r][c] * 256 for r in range(ROWS)]  # identity: Y[r][c]=X[r][c]
        assert got == exp, (
            f"identity pass lane {c}: got {got}, expected {exp}")

    # Pass 2: distinct integers — exercises real cross-terms.
    W = [[((c * 3 + k * 5) % 5) - 2 for k in range(N)] for c in range(N)]
    Y = expected_y(X, W, ROWS, N)
    lanes, collector = await fresh_start(dut)
    await load_weights(dut, W)
    await stream_inputs(dut, X, ROWS, N)
    await finish(dut, collector)
    for c in range(N):
        got = lanes[c]
        exp = [Y[r][c] * 256 for r in range(ROWS)]
        assert got == exp, (
            f"matmul pass lane {c}: got {got}, expected {exp} "
            f"(X={X}, W={W})")
    print(f"N={N} matmul passes OK (identity + distinct integers)")


@cocotb.test()
async def test_nxn_valid_chain_single_column(dut):
    """BUG-SYS-1 semantics at N: a single-column input matrix streams
    only array row 1, yet column 1's bottom PE must still raise valid
    (chained down from sys_start_1). With col_size=1, lanes 2..N stay
    silent."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    X = [[r % 3 - 1] + [0] * (N - 1) for r in range(ROWS)]
    W = [[(k % 3) - 1 for k in range(N)]] + [[0] * N for _ in range(N - 1)]

    lanes, collector = await fresh_start(dut)
    await set_col_size(dut, 1)
    await load_weights(dut, W)
    await stream_inputs(dut, X, ROWS, 1)   # only array row 1 streams
    await finish(dut, collector)

    exp0 = [X[r][0] * W[0][0] * 256 for r in range(ROWS)]
    assert lanes[0] == exp0, (
        f"single-column stream: lane 0 got {lanes[0]}, expected {exp0}")
    for c in range(1, N):
        assert lanes[c] == [], (
            f"col_size=1: lane {c} must stay silent, got {lanes[c]}")
    print(f"N={N} valid-chain single-column pass OK")


@cocotb.test()
async def test_nxn_col_enable(dut):
    """col_size=2 at N>=3: exactly columns 1..2 produce output."""
    if N < 3:
        return
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    X = [[(r + k) % 3 - 1 for k in range(N)] for r in range(ROWS)]
    W = [[((c + k) % 3) - 1 for k in range(N)] for c in range(N)]
    Y = expected_y(X, W, ROWS, 2)

    lanes, collector = await fresh_start(dut)
    await set_col_size(dut, 2)
    await load_weights(dut, W)
    await stream_inputs(dut, X, ROWS, N)
    await finish(dut, collector)

    for c in range(2):
        exp = [Y[r][c] * 256 for r in range(ROWS)]
        assert lanes[c] == exp, (
            f"col_size=2 lane {c}: got {lanes[c]}, expected {exp}")
    for c in range(2, N):
        assert lanes[c] == [], (
            f"col_size=2: lane {c} must stay silent, got {lanes[c]}")
    print(f"N={N} col_size=2 enable pass OK")
