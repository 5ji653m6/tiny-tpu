"""Gate test for src/unified_buffer_nxn.sv (roadmap item 5b), exact-value
half at parameter N (the Makefile's N=4 target overrides the width with
iverilog -Pdump.N=4 and exports UB_NXN_N). Written by the harness author,
not the agent — per tinytpu-loop README "the gate grows with the design".

The companion test_unified_buffer_nxn_equiv.py proves the nxn module is
cycle-by-cycle identical to the legacy unified_buffer at N=2 (all seven
read channels, both transpose modes, gradient descent). THIS test proves
the parameterization itself at N: known memory image (mem[a] = a+1 in
Q8.8, never zero), then exact per-lane beat sequences for the input,
weight, bias, Y and H read channels, plus the registered col_size output.

SHAPE DISCIPLINE — READ BEFORE ADDING CASES: the legacy shared-pointer
read walks deliver correct per-column streams only for TWO-column
matrices (within-cycle lane step is skip=cols+1 and the end-of-cycle
correction undoes exactly one lane's skip; with >2 active lanes the walk
diverges from per-column delivery, and at 4x4 weights it even generates
NEGATIVE addresses — the signed rd_weight_ptr is no accident). Since
this test was written, roadmap items 5d-1a (weights) and 5d-1b
(input/Y/H/grad_weight) GENERALIZED those walks in
src/unified_buffer_nxn.sv to correct N-column delivery; every
expectation here is unchanged EXCEPT transposed input — the one case
where the legacy walk was exercised outside its correct regime (the
legacy p++ scrambled the beats whenever the original matrix had >2
columns; the generalized walk delivers row i, first element first).
The remaining cols=2 shapes are bit-identical under the generalized
walks by construction (the generalized corrections reduce to the legacy
constants at C=2), and the equivalence test anchors that to silicon at
N=2. Lanes >= cols are still asserted SILENT.

Zero-filtering: bias/Y/H have no valid line and drive '0 outside their
emission window; since every written word is nonzero, collecting nonzero
beats per lane yields exactly the emission sequence. Input/weight are
valid-qualified. All compares are exact integer equality.

Unlike the upstream tests, the asserts here are LIVE — PYTHONOPTIMIZE is
empty (NOASSERT is never defined), so they fire.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

N = int(os.environ.get("UB_NXN_N", "2"))
FRAC_BITS = 8


def mem_image(nwords):
    """mem[a] = a+1 in Q8.8 (nonzero everywhere)."""
    return [(a + 1) << FRAC_BITS for a in range(nwords)]


# ---------------------------------------------------------------- model --
def model_lanes(mem, kind, addr, rows, cols, transpose=False):
    """Per-lane expected beat sequences for one read channel, following
    the legacy unified_buffer read algorithms (validated against silicon
    at N=2 by the equivalence test). See the SHAPE DISCIPLINE note in the
    module docstring for why callers pass cols=2 shapes.

    input/Y/H: lane i active while counter in [i, rows+i) and i < cols,
      walking the pointer +1 per read (loop order N-1..0, or 0..N-1 for
      transposed input — see the corrected-semantics branch below).
      input transpose swaps row/col sizes on latch.
    weight: pointer steps by -skip (untransposed) or +skip (transposed)
      per read, skip = ORIGINAL cols + 1, with a +-skip-+1 correction at
      the end of each cycle; start pointer per the legacy case block.
    bias: fixed base pointer, lane i reads mem[addr+i] during its window
      (no transpose handling, incrementing loop).
    """
    if kind == "input" and transpose:
        # Item 5d-1b corrected walk: lane i (i < original rows) streams
        # ROW i of the original matrix, FIRST element first — beat k at
        # cycle t = i+k carries mem[addr + i*cols + k]. The legacy p++
        # walk scrambled this whenever original cols > 2 (the same
        # C=2-only coincidence as the weight walk's C^2 == C+2).
        lanes = [[] for _ in range(N)]
        eff_rows, eff_cols = cols, rows      # latched: R' = orig cols, C' = orig rows
        t = 0
        while t + 1 < eff_rows + eff_cols:
            for i in range(N):               # incrementing loop
                if t >= i and t < eff_rows + i and i < eff_cols:
                    lanes[i].append(mem[addr + i * cols + (t - i)])
            t += 1
        return lanes
    eff_rows, eff_cols = rows, cols
    if kind == "weight":
        skip = cols + 1                     # original cols, per RTL
        if transpose:
            eff_rows, eff_cols = cols, rows
            ptr = addr + cols - 1           # original cols, per RTL
        else:
            ptr = addr + eff_rows * eff_cols - eff_cols
    else:
        ptr = addr

    lanes = [[] for _ in range(N)]
    t = 0
    while t + 1 < eff_rows + eff_cols:
        if kind == "bias" or (kind in ("input", "weight") and transpose):
            order = range(N)                # incrementing loop
        else:
            order = range(N - 1, -1, -1)    # decrementing loop
        for i in order:
            if not (t >= i and t < eff_rows + i and i < eff_cols):
                continue
            if kind == "bias":
                lanes[i].append(mem[addr + i])
            elif kind == "weight":
                lanes[i].append(mem[ptr])
                ptr += skip if transpose else -skip
            else:
                lanes[i].append(mem[ptr])
                ptr += 1
        if kind == "weight":
            ptr += (-skip - 1) if transpose else (skip + 1)
        t += 1
    return lanes


# --------------------------------------------------------------- driver --
async def collect(dut, lanes):
    """Sample every channel every cycle. lanes[ch][i]: input/weight are
    valid-qualified; bias/Y/H collect nonzero data (see module docstring).
    """
    ub = dut.ub_nxn
    while True:
        await RisingEdge(dut.clk)
        for i in range(N):
            if ub.ub_rd_input_valid_out[i].value.integer:
                lanes["input"][i].append(
                    ub.ub_rd_input_data_out[i].value.integer)
            if ub.ub_rd_weight_valid_out[i].value.integer:
                lanes["weight"][i].append(
                    ub.ub_rd_weight_data_out[i].value.integer)
            for ch in ("bias", "Y", "H"):
                v = getattr(ub, f"ub_rd_{ch}_data_out")[i].value.integer
                if v:
                    lanes[ch][i].append(v)


async def tick(dut, cycles=1):
    for _ in range(cycles):
        await RisingEdge(dut.clk)


async def drive_idle(dut):
    for i in range(N):
        dut.ub_wr_data_in[i].value = 0
        dut.ub_wr_valid_in[i].value = 0
        dut.ub_wr_host_data_in[i].value = 0
        dut.ub_wr_host_valid_in[i].value = 0
    dut.ub_rd_start_in.value = 0
    dut.ub_rd_transpose.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_row_size.value = 0
    dut.ub_rd_col_size.value = 0


async def host_write(dut, mem):
    """Write the memory image: beat b lane i = mem[N*b + (N-1-i)] (the
    decrementing write loop puts lane N-1 at the lower address, BUG-UB-2).
    """
    nbeats = len(mem) // N
    for b in range(nbeats):
        for i in range(N):
            dut.ub_wr_host_data_in[i].value = mem[N*b + (N - 1 - i)]
            dut.ub_wr_host_valid_in[i].value = 1
        await tick(dut)
    for i in range(N):
        dut.ub_wr_host_data_in[i].value = 0
        dut.ub_wr_host_valid_in[i].value = 0
    await tick(dut)


async def read_cmd(dut, ptr_select, addr, rows, cols, transpose=0):
    dut.ub_rd_start_in.value = 1
    dut.ub_ptr_select.value = ptr_select
    dut.ub_rd_addr_in.value = addr
    dut.ub_rd_row_size.value = rows
    dut.ub_rd_col_size.value = cols
    dut.ub_rd_transpose.value = transpose
    await tick(dut)
    dut.ub_rd_start_in.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_row_size.value = 0
    dut.ub_rd_col_size.value = 0
    dut.ub_rd_transpose.value = 0


def check(lanes, mem, kind, addr, rows, cols, transpose=False):
    exp = model_lanes(mem, kind, addr, rows, cols, transpose)
    for i in range(N):
        got = lanes[kind][i]
        assert got == exp[i], (
            f"{kind} lane {i} (addr={addr} rows={rows} cols={cols} "
            f"transpose={int(transpose)}): got {got}, expected {exp[i]}")
    # lanes accumulate across the whole test — clear after each check so
    # the next phase compares only its own beats
    for i in range(N):
        lanes[kind][i].clear()


@cocotb.test()
async def test_unified_buffer_nxn(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    lanes = {ch: [[] for _ in range(N)] for ch in ("input", "weight",
                                                   "bias", "Y", "H")}

    dut.rst.value = 1
    await drive_idle(dut)
    dut.learning_rate_in.value = 2 << FRAC_BITS
    await tick(dut, 2)
    dut.rst.value = 0
    await tick(dut)

    # Start the collector only after reset has clocked the X's away —
    # reads right after RisingEdge see the pre-edge state.
    collector = cocotb.start_soon(collect(dut, lanes))

    # 32 words (mem[a] = a+1 Q8.8) — room to spare under
    # UNIFIED_BUFFER_WIDTH=128 at both N=2 (16 beats) and N=4 (8 beats).
    mem = mem_image(32)
    await host_write(dut, mem)

    # 1x1 weight read: doubles as the col_size check (registered output
    # pulses valid for one cycle with data = col_size, untransposed) and
    # is the one weight shape whose walk is correct at ANY N.
    await read_cmd(dut, 1, addr=0, rows=1, cols=1, transpose=0)
    await tick(dut)
    assert dut.ub_nxn.ub_rd_col_size_valid_out.value.integer == 1, (
        "col_size_valid_out must pulse after a weight read command")
    assert dut.ub_nxn.ub_rd_col_size_out.value.integer == 1, (
        f"col_size_out: got "
        f"{dut.ub_nxn.ub_rd_col_size_out.value.integer}, expected 1")
    await tick(dut)
    assert dut.ub_nxn.ub_rd_col_size_valid_out.value.integer == 0, (
        "col_size_valid_out must be a single-cycle pulse")
    await tick(dut, 4)
    check(lanes, mem, "weight", addr=0, rows=1, cols=1, transpose=False)

    # Input read (untransposed), 4 rows x 2 cols — correct per-column
    # delivery at any N (see SHAPE DISCIPLINE); lanes 2..N-1 stay silent.
    await read_cmd(dut, 0, addr=0, rows=4, cols=2, transpose=0)
    await tick(dut, 4 + 2 * N + 6)
    check(lanes, mem, "input", addr=0, rows=4, cols=2, transpose=False)

    # Input read (transposed): 2x4 command latches as 4 rows x 2 cols.
    # 5d-1b: expects the CORRECTED delivery (lane i = row i, first
    # element first) — the legacy walk scrambled these beats.
    await read_cmd(dut, 0, addr=0, rows=2, cols=4, transpose=1)
    await tick(dut, 4 + 2 * N + 6)
    check(lanes, mem, "input", addr=0, rows=2, cols=4, transpose=True)

    # Weight reads, 2-column shapes (correct walk at any N).
    await read_cmd(dut, 1, addr=4, rows=4, cols=2, transpose=0)
    await tick(dut, 4 + 2 * N + 6)
    check(lanes, mem, "weight", addr=4, rows=4, cols=2, transpose=False)
    await read_cmd(dut, 1, addr=8, rows=2, cols=2, transpose=1)
    await tick(dut, 2 + 2 * N + 6)
    check(lanes, mem, "weight", addr=8, rows=2, cols=2, transpose=True)

    # Bias at full width (per-lane fixed addressing is correct at any N),
    # then Y / H at cols=2, issued back-to-back (independent counters).
    await read_cmd(dut, 2, addr=10, rows=4, cols=N)
    await read_cmd(dut, 3, addr=4, rows=3, cols=2)
    await read_cmd(dut, 4, addr=8, rows=4, cols=2)
    await tick(dut, 4 + 2 * N + 8)

    collector.kill()

    check(lanes, mem, "bias", addr=10, rows=4, cols=N)
    check(lanes, mem, "Y", addr=4, rows=3, cols=2)
    check(lanes, mem, "H", addr=8, rows=4, cols=2)

    print(f"N={N} unified_buffer_nxn exact-value passes OK "
          f"(input R/T, weight R/T + 1x1, bias@{N}, Y, H, col_size)")
