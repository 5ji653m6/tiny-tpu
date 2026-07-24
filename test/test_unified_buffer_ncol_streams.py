"""Gate test for the N-column input/Y/H read streams in
src/unified_buffer_nxn.sv (roadmap item 5d-1b). Written by the harness
author, not the agent — per tinytpu-loop README "the gate grows with the
design".

The legacy shared-pointer walks for the input (untransposed and
transposed), Y, H and grad_weight channels step the pointer by +1 per
read with NO end-of-cycle correction — which delivers correct per-column
(or per-row, for transposed input) streams only where the within-cycle
step equals C-1, i.e. C=2 (the same family of coincidence as the weight
walk's C² == C+2, fixed in 5d-1a). Item 5d-1b generalizes: within-cycle
step C-1 plus a phase-dependent end-of-cycle correction built from the
first/last active lane indices. The three 5b tests anchor the C<=2
behavior bit-exactly; THIS test anchors the GENERALIZED semantics at
N=4 with a model that computes the target per-lane delivery directly
(not the walk internals):

  untransposed input / Y / H / grad_weight, R rows x C cols row-major:
    lane i (i < C) beat k (at cycle i+k) = mem[addr + k*C + i]
    (column i, TOP row first)
  transposed input (latched sizes swap: R' = original cols = beats per
  lane, C' = original rows = lane window):
    lane i (i < original rows) beat k = mem[addr + i*C_orig + k]
    (row i of the original matrix, FIRST element first)

Shapes exercised at N=4: 4x4 (the broken case), 2x4 (R < C phase
interleave: lanes still joining while others retire), 4x3 (odd column
count), 4x2 (legacy-correct anchor at N=4) for the descending walks;
4x4, 2x4 (R'=4/C'=2), 4x2 (R'=2/C'=4, legacy-correct at any window)
for transposed input.

Collection: input is valid-qualified; Y/H/grad_weight have no valid —
with this memory image (mem[a] = (a+1)<<8, nonzero everywhere) a
nonzero beat is a real beat and idle cycles drive '0. grad_weight is
observed on the internal value_old_in (per-lane gradient_descent input)
and is tested LAST: the gradient-descent write-back path may legitimately
write memory while a grad channel is active, so no other phase may
follow it.

Unlike the upstream tests, the asserts here are LIVE — PYTHONOPTIMIZE is
empty (NOASSERT is never defined), so they fire.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

N = int(os.environ.get("UB_NXN_N", "4"))
FRAC_BITS = 8

CH_INPUT = 0
CH_Y = 3
CH_H = 4
CH_GRAD_WEIGHT = 6


def mem_image(nwords):
    """mem[a] = a+1 in Q8.8 (nonzero everywhere)."""
    return [(a + 1) << FRAC_BITS for a in range(nwords)]


def model_column_streams(mem, addr, rows, cols):
    """Untransposed input/Y/H/grad_weight target semantics: lane i
    (i < C) beat k = mem[addr + k*C + i] — column i, top row first."""
    lanes = [[] for _ in range(N)]
    for i in range(min(cols, N)):
        for k in range(rows):
            lanes[i].append(mem[addr + k * cols + i])
    return lanes


def model_row_streams(mem, addr, rows, cols):
    """Transposed input target semantics: lane i (i < original rows)
    beat k = mem[addr + i*cols + k] — row i, first element first."""
    lanes = [[] for _ in range(N)]
    for i in range(min(rows, N)):
        for k in range(cols):
            lanes[i].append(mem[addr + i * cols + k])
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


async def issue_read(dut, channel, addr, rows, cols, transpose=0):
    dut.ub_rd_start_in.value = 1
    dut.ub_ptr_select.value = channel
    dut.ub_rd_addr_in.value = addr
    dut.ub_rd_row_size.value = rows
    dut.ub_rd_col_size.value = cols
    dut.ub_rd_transpose.value = transpose
    await tick(dut)
    await drive_idle(dut)


class Collector:
    """Per-lane beat collector for all four stream channels. Started
    only after reset has clocked the X's away. input is valid-qualified;
    Y/H/grad_weight collect nonzero data (mem image is nonzero
    everywhere; idle cycles drive '0)."""

    def __init__(self, dut):
        self.dut = dut
        self.clear()

    def clear(self):
        self.lanes = {ch: [[] for _ in range(N)]
                      for ch in ("input", "Y", "H", "grad_weight")}

    async def run(self):
        ub = self.dut.ub_nxn
        while True:
            await RisingEdge(self.dut.clk)
            for i in range(N):
                if ub.ub_rd_input_valid_out[i].value.integer:
                    self.lanes["input"][i].append(
                        ub.ub_rd_input_data_out[i].value.integer)
                y = ub.ub_rd_Y_data_out[i].value.integer
                if y:
                    self.lanes["Y"][i].append(y)
                h = ub.ub_rd_H_data_out[i].value.integer
                if h:
                    self.lanes["H"][i].append(h)
                v = ub.value_old_in[i].value.integer
                if v:
                    self.lanes["grad_weight"][i].append(v)

    def check(self, channel, expected, tag):
        got = self.lanes[channel]
        for i in range(N):
            assert got[i] == expected[i], (
                f"{channel} lane {i} ({tag}): got {got[i]}, "
                f"expected {expected[i]}")
        self.clear()


@cocotb.test()
async def test_unified_buffer_ncol_streams(dut):
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

    async def run_phase(channel, addr, rows, cols, transpose=0):
        await issue_read(dut, channel, addr, rows, cols, transpose)
        # walk lives R+C-1 cycles; data appears one cycle after the
        # start latch; generous drain like the 5d-1a test
        await tick(dut, rows + cols + 2 * N + 4)

    # ---- input, untransposed (descending walk) ----
    # 4x4 — the shape the legacy p++ walk scrambles at C=4.
    await run_phase(CH_INPUT, addr=0, rows=4, cols=4)
    col.check("input", model_column_streams(mem, 0, 4, 4), "input 4x4")

    # 2x4 — R < C: trailing lanes still joining while leading lanes
    # retire (the phase interleave the correction must handle).
    await run_phase(CH_INPUT, addr=16, rows=2, cols=4)
    col.check("input", model_column_streams(mem, 16, 2, 4), "input 2x4")

    # 4x3 — odd column count, three active lanes.
    await run_phase(CH_INPUT, addr=0, rows=4, cols=3)
    col.check("input", model_column_streams(mem, 0, 4, 3), "input 4x3")

    # 4x2 — legacy-correct shape; must stay bit-exact.
    await run_phase(CH_INPUT, addr=16, rows=4, cols=2)
    col.check("input", model_column_streams(mem, 16, 4, 2), "input 4x2")

    # ---- input, transposed (ascending walk) ----
    # original 4x4: lane i streams row i (4 beats), all 4 lanes.
    await run_phase(CH_INPUT, addr=0, rows=4, cols=4, transpose=1)
    col.check("input", model_row_streams(mem, 0, 4, 4), "input 4x4 T")

    # original 2x4: R' = 4 beats, C' = 2 lanes — legacy-correct window,
    # generalized beat count.
    await run_phase(CH_INPUT, addr=16, rows=2, cols=4, transpose=1)
    col.check("input", model_row_streams(mem, 16, 2, 4), "input 2x4 T")

    # original 4x2: R' = 2 (legacy-correct beat walk) with a 4-lane
    # window — must stay bit-exact.
    await run_phase(CH_INPUT, addr=0, rows=4, cols=2, transpose=1)
    col.check("input", model_row_streams(mem, 0, 4, 2), "input 4x2 T")

    # ---- Y (descending walk, no transpose) ----
    await run_phase(CH_Y, addr=0, rows=4, cols=4)
    col.check("Y", model_column_streams(mem, 0, 4, 4), "Y 4x4")

    await run_phase(CH_Y, addr=16, rows=2, cols=4)
    col.check("Y", model_column_streams(mem, 16, 2, 4), "Y 2x4")

    await run_phase(CH_Y, addr=16, rows=4, cols=2)
    col.check("Y", model_column_streams(mem, 16, 4, 2), "Y 4x2")

    # ---- H (descending walk, no transpose) ----
    await run_phase(CH_H, addr=16, rows=4, cols=4)
    col.check("H", model_column_streams(mem, 16, 4, 4), "H 4x4")

    await run_phase(CH_H, addr=0, rows=4, cols=3)
    col.check("H", model_column_streams(mem, 0, 4, 3), "H 4x3")

    # ---- grad_weight (descending walk into value_old_in) — LAST: the
    # gradient-descent write-back path may write memory while this
    # channel is active, so no read phase may follow it. ----
    await run_phase(CH_GRAD_WEIGHT, addr=0, rows=4, cols=4)
    col.check("grad_weight", model_column_streams(mem, 0, 4, 4),
              "grad_weight 4x4")

    await run_phase(CH_GRAD_WEIGHT, addr=16, rows=4, cols=2)
    col.check("grad_weight", model_column_streams(mem, 16, 4, 2),
              "grad_weight 4x2")

    collector.kill()
    print(f"N={N} N-column input/Y/H streams pass OK "
          f"(input 4x4/2x4/4x3/4x2 + T 4x4/2x4/4x2, Y 4x4/2x4/4x2, "
          f"H 4x4/4x3, grad_weight 4x4/4x2)")
