"""Gate test for src/unified_buffer_nxn.sv (roadmap item 5b), N=2
equivalence half. Written by the harness author, not the agent — per
tinytpu-loop README "the gate grows with the design".

The dump wrapper (test/dump_ub_nxn_equiv.sv) instantiates the LEGACY
unified_buffer and the new unified_buffer_nxn side by side, sharing every
input. This test drives both through all seven read channels (input,
weight, bias, Y, H, grad-bias, grad-weight) in both transpose modes plus
host writes and gradient beats, and asserts the nxn module's array ports
equal the legacy module's scalar ports CYCLE BY CYCLE. The legacy module
is the spec — this catches any port-wiring or indexing mistake outright.

Unlike the upstream tests, the asserts here are LIVE — PYTHONOPTIMIZE is
empty (NOASSERT is never defined), so they fire.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

N = 2


def to_fixed(val, frac_bits=8):
    scaled = int(round(val * (1 << frac_bits)))
    return scaled & 0xFFFF


# (legacy scalar port pair, nxn array port) for every read channel + col_size
CHANNELS = [
    ("ub_rd_input_data_out", True),
    ("ub_rd_input_valid_out", True),
    ("ub_rd_weight_data_out", True),
    ("ub_rd_weight_valid_out", True),
    ("ub_rd_bias_data_out", True),
    ("ub_rd_Y_data_out", True),
    ("ub_rd_H_data_out", True),
]

cycle_no = 0
armed = False  # compare only after reset has clocked the X's away
mismatches = []


def compare(dut):
    """Sample every read port on both instances; record any divergence."""
    global cycle_no
    if not armed:
        return
    cycle_no += 1
    for name, per_lane in CHANNELS:
        for lane in range(N):
            legacy = getattr(dut.ub_legacy, f"{name}_{lane}").value.integer
            nxn = getattr(dut.ub_nxn, name)[lane].value.integer
            if legacy != nxn:
                mismatches.append(
                    f"cycle {cycle_no}: {name}[{lane}] legacy={legacy:#06x} "
                    f"nxn={nxn:#06x}")
    for name in ("ub_rd_col_size_out", "ub_rd_col_size_valid_out"):
        legacy = getattr(dut.ub_legacy, name).value.integer
        nxn = getattr(dut.ub_nxn, name).value.integer
        if legacy != nxn:
            mismatches.append(
                f"cycle {cycle_no}: {name} legacy={legacy:#06x} "
                f"nxn={nxn:#06x}")


async def tick(dut, cycles=1):
    for _ in range(cycles):
        await RisingEdge(dut.clk)
        compare(dut)


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


async def host_write(dut, words):
    """Row-major host write: at each beat the decrementing write loop puts
    lane N-1 at the lower address, so lane i carries words[beat][N-1-i]."""
    for beat in words:
        for i in range(N):
            val = beat[N - 1 - i] if beat[N - 1 - i] is not None else 0
            dut.ub_wr_host_data_in[i].value = to_fixed(val)
            dut.ub_wr_host_valid_in[i].value = (
                1 if beat[N - 1 - i] is not None else 0)
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


async def grad_beats(dut, beats):
    """Inject VPU-side gradient beats (ub_wr_data_in/valid) row-major."""
    for beat in beats:
        for i in range(N):
            val = beat[N - 1 - i] if beat[N - 1 - i] is not None else 0
            dut.ub_wr_data_in[i].value = to_fixed(val)
            dut.ub_wr_valid_in[i].value = (
                1 if beat[N - 1 - i] is not None else 0)
        await tick(dut)
    for i in range(N):
        dut.ub_wr_data_in[i].value = 0
        dut.ub_wr_valid_in[i].value = 0
    await tick(dut)


@cocotb.test()
async def test_unified_buffer_nxn_equiv(dut):
    """Lockstep the legacy and nxn unified buffers through every channel;
    the nxn array ports must equal the legacy scalar ports every cycle."""
    global armed
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    await drive_idle(dut)
    dut.learning_rate_in.value = to_fixed(2)
    await tick(dut, 2)
    dut.rst.value = 0
    await tick(dut)
    armed = True  # registers cleared; X-free comparisons from here on

    # Host-write 16 distinctive words (8 beats x 2 lanes).
    words = [[1, 2], [3, 4], [5, 6], [7, 8],
             [9, 10], [11, 12], [13, 14], [15, 16]]
    await host_write(dut, words)

    # Input + weight reads, untransposed then transposed.
    await read_cmd(dut, 0, addr=2, rows=3, cols=2, transpose=0)
    await read_cmd(dut, 1, addr=0, rows=3, cols=2, transpose=0)
    await tick(dut, 8)
    await read_cmd(dut, 0, addr=0, rows=3, cols=2, transpose=1)
    await read_cmd(dut, 1, addr=2, rows=3, cols=2, transpose=1)
    await tick(dut, 8)

    # Bias / Y / H reads.
    await read_cmd(dut, 2, addr=5, rows=4, cols=3)
    await read_cmd(dut, 3, addr=2, rows=2, cols=2)
    await read_cmd(dut, 4, addr=4, rows=2, cols=2)
    await tick(dut, 8)

    # Gradient descent (biases) with gradient beats.
    await read_cmd(dut, 5, addr=0, rows=2, cols=2)
    await grad_beats(dut, [[5, None], [7, 6], [None, 8]])
    await tick(dut, 8)

    # Gradient descent (weights) with gradient beats.
    await read_cmd(dut, 6, addr=4, rows=2, cols=2)
    await grad_beats(dut, [[1, None], [3, 2], [None, 4]])
    await tick(dut, 10)

    assert not mismatches, (
        f"{len(mismatches)} cycle-level mismatches between legacy and nxn "
        f"unified buffers:\n" + "\n".join(mismatches[:20]))
    print(f"N=2 equivalence OK ({cycle_no} cycles compared, all channels)")
