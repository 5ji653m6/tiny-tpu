"""Gate test for the agent-authored ISA extension (src/tpu.sv 7-bit
vpu_data_pathway port). Written by the harness author, not the agent —
this is what proves software can now route a real full-chip matmul
through the NEW VPU stages from the tpu top-level port, per tinytpu-loop
README "the gate grows with the design".

Method (self-referential, two passes over the same RTL):
  Pass A: forward pass (the test_tpu XOR-network layer 1) with pathway
          0b0000000 (all stages bypassed) -> collect the raw systolic
          output beats Z at the VPU output, PER LANE (the two lanes'
          valids are skewed by 1 cycle at the chip level — see below).
  Pass B: identical pass with pathway 0b1000000 (softmax only — a bit
          that did not exist at the tpu port before this task) ->
          collect the softmax beats per lane.
  Assert: softmax beats match the float spec softmax(Z) computed from the
  pass-A beats. The systolic MAC's fixed-point rounding cancels out
  entirely (both passes see bit-identical array outputs), so only the
  softmax stage at the chip level is under test.

Pass A is anchored against the EXACT raw beats that the pristine
test_tpu's forward phase produces (verified 2026-07-24 against the
BUG-SYS-1 scoped VCD diag/tpu_scoped.vcd: lane1 = [0, -148, -148, -296],
lane2 = [0, 108, 108, 216] pre-bias — test_tpu's bias path adds
B1 = [-126, 48] raw and its leaky-relu halves the negatives, which
reconstructs those exact pre-bias values). NOTE: test_tpu.py's header
"expected Z1" comment does NOT match silicon (its row 0 claims
[-0.5614, 1.0294] for a [0,0] input row) — it is stale documentation,
never asserted. The anchor here is the silicon behavior.

LANE SKEW: the UB staggers array-row streams by 1 cycle (systolic
wavefront), so output column 1 emerges one cycle ahead of column 2:
vpu_valid_out_1 fires on cycles [base..base+3] and vpu_valid_out_2 on
[base+1..base+4]. Beats are therefore collected PER LANE and paired by
index: output row r = (lane1[r], lane2[r]). The cross-lane group stages
(layernorm, softmax) rely on this too: vpu.sv's BUG-SKEW-1 alignment
registers hold lane 1 for one cycle at each group stage's input so the
stage pairs columns OF THE SAME ROW (before that fix this test collected
3 mis-paired beats — sigmoid of (row r, col 0) vs (row r-1, col 1)).

Unlike the upstream tests, the asserts here are LIVE — PYTHONOPTIMIZE is
empty (NOASSERT is never defined), so they fire.
"""

import math

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge

FRAC_BITS = 8
LSB = 1.0 / (1 << FRAC_BITS)
# Softmax stage itself: module-level test measured max err 0.00189
# (~0.5 LSB) against the same spec; 2 LSB matches test_softmax_parent.
TOL = 2 * LSB

X = [[0., 0.],
     [0., 1.],
     [1., 0.],
     [1., 1.]]
W1 = [[0.2985, -0.5792],
      [0.0913, 0.4234]]
B1 = [-0.4939, 0.189]

# Exact raw (Q8.8 int) systolic beats of the pristine forward pass — see
# the module docstring for the VCD provenance. Deterministic RTL: exact.
ANCHOR_LANE1 = [0, -148, -148, -296]
ANCHOR_LANE2 = [0, 108, 108, 216]

PATHWAY_BYPASS = 0b0000000
PATHWAY_SOFTMAX = 0b1000000   # bit 6 — only reachable via the widened port


def to_fixed(val, frac_bits=FRAC_BITS):
    """convert python float to signed 16-bit fixed-point (Q8.8)."""
    scaled = int(round(val * (1 << frac_bits)))
    return scaled & 0xFFFF


def from_fixed(val, frac_bits=FRAC_BITS):
    """convert signed 16-bit fixed-point to python float."""
    if val >= 1 << 15:
        val -= 1 << 16
    return float(val) / (1 << frac_bits)


def softmax_spec(x1, x2):
    d = x1 - x2
    y1 = 1.0 / (1.0 + math.exp(-d))
    return y1, 1.0 - y1


async def collect_vpu_beats(dut, lane1, lane2):
    """Sample the VPU outputs every cycle, PER LANE (valids are skewed —
    see module docstring). vpu_data_out_1/2 are tpu-internal wires,
    readable via the hierarchy. NOTE (cocotb/Icarus read timing): reads
    right after `await RisingEdge` see the pre-edge state, so every cycle
    must be sampled.
    """
    while True:
        await RisingEdge(dut.clk)
        if dut.vpu_valid_out_1.value.integer:
            lane1.append(dut.vpu_data_out_1.value.integer)
        if dut.vpu_valid_out_2.value.integer:
            lane2.append(dut.vpu_data_out_2.value.integer)


def s16(v):
    """interpret a 16-bit cocotb value as signed."""
    return v - (1 << 16) if v >= 1 << 15 else v


async def forward_pass(dut, pathway):
    """Reset, load UB, run the layer-1 forward pass.

    Returns (lane1, lane2): per-lane lists of signed raw Q8.8 output
    beats (pair by index for output rows).
    """
    lane1, lane2 = [], []

    dut.rst.value = 1
    dut.ub_wr_host_data_in[0].value = 0
    dut.ub_wr_host_data_in[1].value = 0
    dut.ub_wr_host_valid_in[0].value = 0
    dut.ub_wr_host_valid_in[1].value = 0
    dut.ub_rd_start_in.value = 0
    dut.ub_rd_transpose.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_row_size.value = 0
    dut.ub_rd_col_size.value = 0
    dut.learning_rate_in.value = 0
    dut.vpu_data_pathway.value = 0
    dut.sys_switch_in.value = 0
    dut.vpu_leak_factor_in.value = 0
    dut.inv_batch_size_times_two_in.value = 0
    await RisingEdge(dut.clk)

    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # Load X, B1, W1 using test_tpu's exact host-write interleave. The UB
    # has a SINGLE flat memory and one write pointer (incremented per
    # valid lane-write), so the Y/W2/B2 slots between X and W1/B1 must
    # still be written (with dummy 0s — this test never reads them) to
    # keep W1 and B1 at the same addresses the read instructions below
    # use. Only the final B2/W2[1] cycle is dropped.
    dut.ub_wr_host_data_in[0].value = to_fixed(X[0][0])
    dut.ub_wr_host_valid_in[0].value = 1
    await RisingEdge(dut.clk)

    for i in range(len(X) - 1):
        dut.ub_wr_host_data_in[0].value = to_fixed(X[i + 1][0])
        dut.ub_wr_host_valid_in[0].value = 1
        dut.ub_wr_host_data_in[1].value = to_fixed(X[i][1])
        dut.ub_wr_host_valid_in[1].value = 1
        await RisingEdge(dut.clk)

    dut.ub_wr_host_data_in[0].value = 0   # Y[0] slot (unused by this test)
    dut.ub_wr_host_valid_in[0].value = 1
    dut.ub_wr_host_data_in[1].value = to_fixed(X[3][1])
    dut.ub_wr_host_valid_in[1].value = 1
    await RisingEdge(dut.clk)

    for i in range(3):
        dut.ub_wr_host_data_in[0].value = 0   # Y[1..3] slots (unused)
        dut.ub_wr_host_valid_in[0].value = 1
        dut.ub_wr_host_data_in[1].value = 0
        dut.ub_wr_host_valid_in[1].value = 0
        await RisingEdge(dut.clk)

    dut.ub_wr_host_data_in[0].value = to_fixed(W1[0][0])
    dut.ub_wr_host_valid_in[0].value = 1
    await RisingEdge(dut.clk)

    dut.ub_wr_host_data_in[0].value = to_fixed(W1[1][0])
    dut.ub_wr_host_valid_in[0].value = 1
    dut.ub_wr_host_data_in[1].value = to_fixed(W1[0][1])
    dut.ub_wr_host_valid_in[1].value = 1
    await RisingEdge(dut.clk)

    dut.ub_wr_host_data_in[0].value = to_fixed(B1[0])
    dut.ub_wr_host_valid_in[0].value = 1
    dut.ub_wr_host_data_in[1].value = to_fixed(W1[1][1])
    dut.ub_wr_host_valid_in[1].value = 1
    await RisingEdge(dut.clk)

    dut.ub_wr_host_data_in[0].value = 0   # W2[0] slot (unused)
    dut.ub_wr_host_valid_in[0].value = 1
    dut.ub_wr_host_data_in[1].value = to_fixed(B1[1])
    dut.ub_wr_host_valid_in[1].value = 1
    await RisingEdge(dut.clk)

    dut.ub_wr_host_data_in[0].value = 0
    dut.ub_wr_host_valid_in[0].value = 0
    dut.ub_wr_host_data_in[1].value = 0
    dut.ub_wr_host_valid_in[1].value = 0
    await RisingEdge(dut.clk)

    # Load W1^T into the systolic array (read W1 from UB to the top).
    dut.ub_rd_start_in.value = 1
    dut.ub_rd_transpose.value = 1
    dut.ub_ptr_select.value = 1
    dut.ub_rd_addr_in.value = 12
    dut.ub_rd_row_size.value = 2
    dut.ub_rd_col_size.value = 2
    await RisingEdge(dut.clk)

    dut.ub_rd_start_in.value = 0
    dut.ub_rd_transpose.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_row_size.value = 0
    dut.ub_rd_col_size.value = 0
    await RisingEdge(dut.clk)

    # Stream X into the array (read X from UB to the left side) with the
    # pathway under test set at the tpu top-level port.
    collector = cocotb.start_soon(collect_vpu_beats(dut, lane1, lane2))
    dut.ub_rd_start_in.value = 1
    dut.ub_rd_transpose.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_row_size.value = 4
    dut.ub_rd_col_size.value = 2
    dut.vpu_data_pathway.value = pathway
    await RisingEdge(dut.clk)

    dut.ub_rd_start_in.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_row_size.value = 0
    dut.ub_rd_col_size.value = 0
    dut.sys_switch_in.value = 1
    await RisingEdge(dut.clk)

    # Read B1 from UB for 4 clock cycles (bias stage is bypassed under
    # both pathways here; kept for timing fidelity with test_tpu).
    dut.ub_rd_start_in.value = 1
    dut.ub_ptr_select.value = 2
    dut.ub_rd_addr_in.value = 16
    dut.ub_rd_row_size.value = 4
    dut.ub_rd_col_size.value = 2
    dut.sys_switch_in.value = 0
    await RisingEdge(dut.clk)

    dut.ub_rd_start_in.value = 0
    dut.ub_ptr_select.value = 0
    dut.ub_rd_addr_in.value = 0
    dut.ub_rd_row_size.value = 0
    dut.ub_rd_col_size.value = 0
    await FallingEdge(dut.vpu_valid_out_1)

    # Let the collector see the final cycle state, then stop it.
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    collector.kill()

    dut.vpu_data_pathway.value = 0
    return [s16(v) for v in lane1], [s16(v) for v in lane2]


@cocotb.test()
async def test_tpu_softmax_pathway(dut):
    """Full-chip softmax via the widened tpu port, self-referential spec."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    raw1, raw2 = await forward_pass(dut, PATHWAY_BYPASS)
    assert len(raw1) == 4, f"pass A lane 1: expected 4 beats, got {len(raw1)}"
    assert len(raw2) == 4, f"pass A lane 2: expected 4 beats, got {len(raw2)}"

    # Anchor: pass A reproduces the pristine forward pass exactly.
    assert raw1 == ANCHOR_LANE1, (
        f"pass A lane 1 = {raw1}, pristine anchor {ANCHOR_LANE1}")
    assert raw2 == ANCHOR_LANE2, (
        f"pass A lane 2 = {raw2}, pristine anchor {ANCHOR_LANE2}")
    print(f"pass A anchor OK: lane1 {raw1}, lane2 {raw2}")

    soft1, soft2 = await forward_pass(dut, PATHWAY_SOFTMAX)
    assert len(soft1) == 4, (
        f"pass B lane 1: expected 4 beats, got {len(soft1)} ({soft1})")
    assert len(soft2) == 4, (
        f"pass B lane 2: expected 4 beats, got {len(soft2)} ({soft2})")

    max_err = 0.0
    for i in range(4):
        z1, z2 = from_fixed(raw1[i]), from_fixed(raw2[i])
        got1, got2 = from_fixed(soft1[i]), from_fixed(soft2[i])
        exp1, exp2 = softmax_spec(z1, z2)
        err = max(abs(got1 - exp1), abs(got2 - exp2))
        max_err = max(max_err, err)
        print(f"row[{i}]: softmax({z1:.4f}, {z2:.4f}) expected "
              f"({exp1:.5f}, {exp2:.5f}), got ({got1:.5f}, {got2:.5f}), "
              f"err {err:.5f}")
        assert err <= TOL, (
            f"row[{i}]: chip softmax({z1:.4f}, {z2:.4f}) = "
            f"({got1:.5f}, {got2:.5f}), spec ({exp1:.5f}, {exp2:.5f}), "
            f"err {err:.5f} > {TOL:.5f}")
        assert abs(got1 + got2 - 1.0) <= LSB, (
            f"row[{i}]: outputs do not sum to 1 ({got1:.5f} + {got2:.5f})")

    # The all-zero row (X' row 0) must normalize to exactly (0.5, 0.5).
    assert (from_fixed(soft1[0]), from_fixed(soft2[0])) == (0.5, 0.5), (
        f"zero-row beat should give exactly (0.5, 0.5), got "
        f"({from_fixed(soft1[0])}, {from_fixed(soft2[0])})")
    print(f"full-chip softmax pathway test passed! "
          f"(max observed err {max_err:.5f})")
