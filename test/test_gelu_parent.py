"""Gate test for the agent-authored GELU VPU stage (src/gelu_parent.sv /
src/gelu_child.sv). Written by the harness author, not the agent — this is
what makes the gelu loop task's fitness signal cover the NEW operator, per
tinytpu-loop README "the gate grows with the design".

Spec under test (from src/gelu_child.sv), Q8.8 fixed point:
    x >= +2.0 : gelu(x) = x
    x <= -2.0 : gelu(x) = 0
    else      : gelu(x) = x/2 + x^2/4   (fxp_mul ROUND=1 for x^2)
Outputs are registered: valid_out asserts one cycle after valid_in, and
valid_out must deassert (with data cleared) when valid_in is low.

Unlike the upstream tests, the asserts here are LIVE — PYTHONOPTIMIZE is
empty (NOASSERT is never defined), so they fire.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

FRAC_BITS = 8
LSB = 1.0 / (1 << FRAC_BITS)
TOL = 2 * LSB  # absorb fxp_mul round-to-nearest + shift floor effects


def to_fixed(val, frac_bits=FRAC_BITS):
    """convert python float to signed 16-bit fixed-point (Q8.8)."""
    scaled = int(round(val * (1 << frac_bits)))
    return scaled & 0xFFFF


def from_fixed(val, frac_bits=FRAC_BITS):
    """convert signed 16-bit fixed-point to python float."""
    if val >= 1 << 15:
        val -= 1 << 16
    return float(val) / (1 << frac_bits)


def gelu_spec(x):
    """the piecewise approximation the RTL documents, in float."""
    if x >= 2.0:
        return x
    if x <= -2.0:
        return 0.0
    return x / 2 + (x * x) / 4


# Coverage: identity region, both boundaries, zero region, and a spread of
# the middle region including exact powers of two (no rounding ambiguity).
COL1_INPUTS = [3.0, 2.0, 1.0, 0.5, 0.0, -1.0]
COL2_INPUTS = [1.5, 0.25, -0.5, -1.5, -2.0, -2.5]


async def drive_and_collect(dut, col1_inputs, col2_inputs):
    """Drive both columns in lockstep, collect valid outputs per column.

    NOTE (cocotb/Icarus read timing): a read right after `await RisingEdge`
    sees the pre-edge state — the registered output for an input sampled at
    edge N is only visible to reads after edge N+1. Outputs must therefore
    be sampled EVERY cycle, including while inputs are still being driven
    (the upstream tests get away with collect-after-drive only because their
    asserts are commented out).
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    dut.gelu_valid_1_in.value = 0
    dut.gelu_valid_2_in.value = 0
    dut.gelu_data_1_in.value = 0
    dut.gelu_data_2_in.value = 0
    await RisingEdge(dut.clk)
    dut.rst.value = 0

    col1_results, col2_results = [], []
    n = max(len(col1_inputs), len(col2_inputs))
    for cyc in range(n + 3):
        if cyc < n:
            if cyc < len(col1_inputs):
                dut.gelu_data_1_in.value = to_fixed(col1_inputs[cyc])
                dut.gelu_valid_1_in.value = 1
            if cyc < len(col2_inputs):
                dut.gelu_data_2_in.value = to_fixed(col2_inputs[cyc])
                dut.gelu_valid_2_in.value = 1
        else:
            dut.gelu_valid_1_in.value = 0
            dut.gelu_valid_2_in.value = 0
        await RisingEdge(dut.clk)
        if dut.gelu_valid_1_out.value.integer:
            col1_results.append(from_fixed(dut.gelu_data_1_out.value.integer))
        if dut.gelu_valid_2_out.value.integer:
            col2_results.append(from_fixed(dut.gelu_data_2_out.value.integer))

    return col1_results, col2_results


@cocotb.test()
async def test_gelu_parent_piecewise_values(dut):
    """Both columns: identity / boundary / middle / zero regions, live asserts."""
    col1_results, col2_results = await drive_and_collect(dut, COL1_INPUTS, COL2_INPUTS)

    assert len(col1_results) == len(COL1_INPUTS), (
        f"col1: expected {len(COL1_INPUTS)} valid outputs, got {len(col1_results)}")
    assert len(col2_results) == len(COL2_INPUTS), (
        f"col2: expected {len(COL2_INPUTS)} valid outputs, got {len(col2_results)}")

    for col, inputs, results in (("col1", COL1_INPUTS, col1_results),
                                 ("col2", COL2_INPUTS, col2_results)):
        for idx, (x, got) in enumerate(zip(inputs, results)):
            exp = gelu_spec(x)
            abs_err = abs(got - exp)
            print(f"{col}[{idx}]: gelu({x:.4f}) expected {exp:.5f}, "
                  f"got {got:.5f}, abs_err {abs_err:.5f}")
            assert abs_err <= TOL, (
                f"{col}[{idx}]: gelu({x}) = {got:.5f}, spec {exp:.5f}, "
                f"err {abs_err:.5f} > {TOL:.5f}")

    # Spot-check the documented continuity: x=+2 maps to 2, x=-2 maps to 0.
    assert abs(col1_results[1] - 2.0) <= TOL, "continuity at +2.0 broken"
    assert abs(col2_results[4] - 0.0) <= TOL, "continuity at -2.0 broken"
    print("piecewise value test passed!")


@cocotb.test()
async def test_gelu_parent_valid_handshake(dut):
    """valid_out follows valid_in by one cycle; invalid input -> invalid output."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    dut.gelu_valid_1_in.value = 0
    dut.gelu_valid_2_in.value = 0
    dut.gelu_data_1_in.value = 0
    dut.gelu_data_2_in.value = 0
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # No valid in -> no valid out.
    assert dut.gelu_valid_1_out.value.integer == 0, "valid_1_out high with no valid input"
    assert dut.gelu_valid_2_out.value.integer == 0, "valid_2_out high with no valid input"

    # One valid beat on col1 only.
    dut.gelu_data_1_in.value = to_fixed(1.0)
    dut.gelu_valid_1_in.value = 1
    await RisingEdge(dut.clk)
    dut.gelu_valid_1_in.value = 0

    # One cycle later: col1 valid with gelu(1.0) = 0.75; col2 still invalid.
    await RisingEdge(dut.clk)
    assert dut.gelu_valid_1_out.value.integer == 1, "valid_1_out did not assert 1 cycle after valid_1_in"
    got = from_fixed(dut.gelu_data_1_out.value.integer)
    assert abs(got - 0.75) <= TOL, f"gelu(1.0) = {got:.5f}, expected 0.75"
    assert dut.gelu_valid_2_out.value.integer == 0, "valid_2_out asserted without valid_2_in"

    # Next cycle: col1 valid must deassert (single beat, not stuck).
    await RisingEdge(dut.clk)
    assert dut.gelu_valid_1_out.value.integer == 0, "valid_1_out stuck high after single input beat"
    print("valid handshake test passed!")
