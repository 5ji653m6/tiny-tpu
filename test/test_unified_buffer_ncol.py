"""Gate test for the N-column weight read schedule in
src/unified_buffer_nxn.sv (roadmap item 5d-1a). Written by the harness
author, not the agent — per tinytpu-loop README "the gate grows with the
design".

The legacy shared-pointer weight walk delivers correct per-column streams
only for TWO-column matrices (its constant ±(skip+1) end-of-cycle
correction undoes exactly one lane's skip — right only where C² == C+2).
Item 5d-1a generalizes the correction. The three 5b tests anchor the
C<=2 behavior bit-exactly; THIS test anchors the GENERALIZED semantics
at N=4 with a model that computes the target per-column delivery
directly (not the walk internals):

  untransposed, R rows x C cols row-major at addr:
    lane i (i < C) beat k (at cycle i+k) = mem[addr + (R-1-k)*C + i]
    (column i, bottom row first)
  transposed (lane window i < R, beats = C per lane):
    lane i beat k = mem[addr + i*C + (C-1-k)]
    (row i of the stored matrix, last element first)

C=1 untransposed is included: the legacy walk re-reads the same address
every cycle there (latent upstream bug, unused by the real chip); the
generalized formula delivers the column correctly — an intended,
documented divergence.

Unlike the upstream tests, the asserts here are LIVE — PYTHONOPTIMIZE is
empty (NOASSERT is never defined), so they fire.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

N = int(os.environ.get("UB_NXN_N", "4"))
FRAC_BITS = 8


def mem_image(nwords):
    """mem[a] = a+1 in Q8.8 (nonzero everywhere)."""
    return [(a + 1) << FRAC_BITS for a in range(nwords)]


def model_weight(mem, addr, rows, cols, transpose=False):
    """Target per-column delivery semantics (see module docstring)."""
    lanes = [[] for _ in range(N)]
    if transpose:
        for i in range(min(rows, N)):      # window i < original R
            for k in range(cols):          # beats = original C
                lanes[i].append(mem[addr + i * cols + (cols - 1 - k)])
    else:
        for i in range(min(cols, N)):      # window i < C
            for k in range(rows):
                lanes[i].append(mem[addr + (rows - 1 - k) * cols + i])
    return lanes


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
    """Beat b lane i = mem[N*b + (N-1-i)] (BUG-UB-2 decrementing write
    loop puts lane N-1 at the lower address)."""
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


async def weight_read(dut, addr, rows, cols, transpose=0):
    dut.ub_rd_start_in.value = 1
    dut.ub_ptr_select.value = 1          # weight channel
    dut.ub_rd_addr_in.value = addr
    dut.ub_rd_row_size.value = rows
    dut.ub_rd_col_size.value = cols
    dut.ub_rd_transpose.value = transpose
    await tick(dut)
    await drive_idle(dut)


class Collector:
    """Valid-qualified per-lane weight-beat collector. Started only after
    reset has clocked the X's away."""

    def __init__(self, dut):
        self.dut = dut
        self.lanes = [[] for _ in range(N)]

    async def run(self):
        while True:
            await RisingEdge(self.dut.clk)
            for i in range(N):
                if self.dut.ub_nxn.ub_rd_weight_valid_out[i].value.integer:
                    self.lanes[i].append(
                        self.dut.ub_nxn.ub_rd_weight_data_out[i].value.integer)

    def check(self, mem, addr, rows, cols, transpose=False):
        exp = model_weight(mem, addr, rows, cols, transpose)
        for i in range(N):
            assert self.lanes[i] == exp[i], (
                f"weight lane {i} (addr={addr} rows={rows} cols={cols} "
                f"transpose={int(transpose)}): got {self.lanes[i]}, "
                f"expected {exp[i]}")
        self.lanes = [[] for _ in range(N)]  # clear for the next phase


@cocotb.test()
async def test_unified_buffer_ncol(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    await drive_idle(dut)
    dut.learning_rate_in.value = 2 << FRAC_BITS
    await tick(dut, 2)
    dut.rst.value = 0
    await tick(dut)

    col = Collector(dut)
    collector = cocotb.start_soon(col.run())

    mem = mem_image(32)
    await host_write(dut, mem)

    # 4x4 untransposed — the shape that made the legacy walk generate
    # negative addresses. Also spot-check col_size reports the original
    # column count.
    await weight_read(dut, addr=0, rows=4, cols=4)
    await tick(dut)
    assert dut.ub_nxn.ub_rd_col_size_out.value.integer == 4, (
        f"col_size_out: got "
        f"{dut.ub_nxn.ub_rd_col_size_out.value.integer}, expected 4")
    await tick(dut, 4 + 2 * N + 4)
    col.check(mem, addr=0, rows=4, cols=4)

    # 4x2 untransposed (legacy-correct shape — must stay bit-exact).
    await weight_read(dut, addr=16, rows=4, cols=2)
    await tick(dut, 4 + 2 * N + 4)
    col.check(mem, addr=16, rows=4, cols=2)

    # 4x3 untransposed — odd column count, three active lanes.
    await weight_read(dut, addr=0, rows=4, cols=3)
    await tick(dut, 4 + 2 * N + 4)
    col.check(mem, addr=0, rows=4, cols=3)

    # 4x1 untransposed — single column; the legacy walk re-reads one
    # address here (latent upstream bug); the generalized schedule
    # delivers the column correctly (intended divergence).
    await weight_read(dut, addr=0, rows=4, cols=1)
    await tick(dut, 4 + 2 * N + 4)
    col.check(mem, addr=0, rows=4, cols=1)

    # 4x4 transposed — lane i streams row i, last element first.
    await weight_read(dut, addr=16, rows=4, cols=4, transpose=1)
    await tick(dut, 4 + 2 * N + 4)
    col.check(mem, addr=16, rows=4, cols=4, transpose=True)

    # 2x4 transposed (stored 2 rows x 4 cols): window i < R = 2 lanes,
    # C = 4 beats each.
    await weight_read(dut, addr=0, rows=2, cols=4, transpose=1)
    await tick(dut, 4 + 2 * N + 4)
    col.check(mem, addr=0, rows=2, cols=4, transpose=True)

    collector.kill()
    print(f"N={N} N-column weight schedule passes OK "
          f"(4x4 R/T, 4x2, 4x3, 4x1, 2x4T, col_size)")
