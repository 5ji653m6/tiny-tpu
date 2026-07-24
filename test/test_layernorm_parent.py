"""Gate test for the agent-authored LayerNorm VPU stage
(src/layernorm_parent.sv). Written by the harness author, not the agent —
same discipline as test_gelu_parent.py: this is what makes the layernorm
loop task's fitness signal cover the NEW operator numerically, per
tinytpu-loop README "the gate grows with the design".

Spec under test (from src/layernorm_parent.sv), Q8.8 fixed point:
    hd  = (x1 - x2) / 2                       (truncating shift, no round)
    var = hd^2                                (fxp_mul ROUND=1, Q16.8)
    std = sqrt(var + eps), eps = 1/16         (fxp_sqrt ROUND=1, Q8.8)
    y1  = hd / std                            (fxp_div ROUND=1, Q8.8)
    y2  = -y1
Outputs are registered: valid_out asserts one cycle after a beat on which
BOTH ln_valid_1_in and ln_valid_2_in were high; if either lane is invalid
the outputs are cleared and valids deassert (the two lanes form one
group). Overflow flags are sticky until reset.

Unlike the upstream tests, the asserts here are LIVE — PYTHONOPTIMIZE is
empty (NOASSERT is never defined), so they fire.
"""

import math

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

FRAC_BITS = 8
LSB = 1.0 / (1 << FRAC_BITS)
EPS = 1.0 / 16  # LN_EPS in the float domain
# The chain truncating half_diff + three ROUND=1 ops (mul/sqrt/div)
# compounds to a few LSB; 8 LSB leaves margin without hiding a real bug.
# Observed errors are printed so the tolerance can be tightened later.
TOL = 8 * LSB


def to_fixed(val, frac_bits=FRAC_BITS):
    """convert python float to signed 16-bit fixed-point (Q8.8)."""
    scaled = int(round(val * (1 << frac_bits)))
    return scaled & 0xFFFF


def from_fixed(val, frac_bits=FRAC_BITS):
    """convert signed 16-bit fixed-point to python float."""
    if val >= 1 << 15:
        val -= 1 << 16
    return float(val) / (1 << frac_bits)


def layernorm_spec(x1, x2):
    """the documented 2-lane layernorm, in float."""
    hd = (x1 - x2) / 2
    y1 = hd / math.sqrt(hd * hd + EPS)
    return y1, -y1


# Coverage: asymmetric pairs both ways, equal inputs (eps floor, no
# div-by-zero), fractional spreads, cross-zero pairs, and large-magnitude
# pairs where y saturates toward +-1.
PAIRS = [
    (3.0, 1.0),
    (1.0, 3.0),
    (2.0, 2.0),
    (0.5, -0.5),
    (-1.0, -2.5),
    (0.25, 0.0),
    (64.0, -64.0),
    (-64.0, 64.0),
]


async def drive_and_collect(dut, pairs):
    """Drive paired beats, collect valid outputs.

    NOTE (cocotb/Icarus read timing): a read right after `await RisingEdge`
    sees the pre-edge state — the registered output for an input sampled at
    edge N is only visible to reads after edge N+1. Outputs must therefore
    be sampled EVERY cycle, including while inputs are still being driven.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    dut.ln_valid_1_in.value = 0
    dut.ln_valid_2_in.value = 0
    dut.ln_data_1_in.value = 0
    dut.ln_data_2_in.value = 0
    await RisingEdge(dut.clk)
    dut.rst.value = 0

    results = []
    for cyc in range(len(pairs) + 3):
        if cyc < len(pairs):
            dut.ln_data_1_in.value = to_fixed(pairs[cyc][0])
            dut.ln_data_2_in.value = to_fixed(pairs[cyc][1])
            dut.ln_valid_1_in.value = 1
            dut.ln_valid_2_in.value = 1
        else:
            dut.ln_valid_1_in.value = 0
            dut.ln_valid_2_in.value = 0
        await RisingEdge(dut.clk)
        if dut.ln_valid_1_out.value.integer and dut.ln_valid_2_out.value.integer:
            results.append((from_fixed(dut.ln_data_1_out.value.integer),
                            from_fixed(dut.ln_data_2_out.value.integer)))

    return results


@cocotb.test()
async def test_layernorm_parent_values(dut):
    """Paired beats: spec values on both lanes, live asserts."""
    results = await drive_and_collect(dut, PAIRS)

    assert len(results) == len(PAIRS), (
        f"expected {len(PAIRS)} valid outputs, got {len(results)}")

    max_err = 0.0
    for idx, ((x1, x2), (got1, got2)) in enumerate(zip(PAIRS, results)):
        exp1, exp2 = layernorm_spec(x1, x2)
        err = max(abs(got1 - exp1), abs(got2 - exp2))
        max_err = max(max_err, err)
        print(f"pair[{idx}]: ln({x1:.4f}, {x2:.4f}) expected "
              f"({exp1:.5f}, {exp2:.5f}), got ({got1:.5f}, {got2:.5f}), "
              f"err {err:.5f}")
        assert err <= TOL, (
            f"pair[{idx}]: ln({x1}, {x2}) = ({got1:.5f}, {got2:.5f}), "
            f"spec ({exp1:.5f}, {exp2:.5f}), err {err:.5f} > {TOL:.5f}")
        # antisymmetry: y2 must be the negation of y1 (one bit of slack
        # for the fixed-point negation of a rounded value)
        assert abs(got1 + got2) <= 2 * LSB, (
            f"pair[{idx}]: y2 != -y1 ({got1:.5f} vs {got2:.5f})")

    # eps floor: equal inputs must give exactly zero, not a div-by-zero.
    assert results[2] == (0.0, 0.0), (
        f"equal-input pair should normalize to (0, 0), got {results[2]}")
    print(f"value test passed! (max observed err {max_err:.5f})")


@cocotb.test()
async def test_layernorm_parent_valid_handshake(dut):
    """Both-valid requirement, 1-cycle latency, deassert, data cleared."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    dut.ln_valid_1_in.value = 0
    dut.ln_valid_2_in.value = 0
    dut.ln_data_1_in.value = 0
    dut.ln_data_2_in.value = 0
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # No valid in -> no valid out.
    assert dut.ln_valid_1_out.value.integer == 0, "valid_1_out high with no valid input"
    assert dut.ln_valid_2_out.value.integer == 0, "valid_2_out high with no valid input"

    # Lane 1 only: the pair is incomplete, so NO valid output may appear.
    dut.ln_data_1_in.value = to_fixed(3.0)
    dut.ln_valid_1_in.value = 1
    await RisingEdge(dut.clk)
    dut.ln_valid_1_in.value = 0
    await RisingEdge(dut.clk)
    assert dut.ln_valid_1_out.value.integer == 0, "valid_1_out asserted on lane-1-only beat"
    assert dut.ln_valid_2_out.value.integer == 0, "valid_2_out asserted on lane-1-only beat"

    # Both lanes: one beat of (3.0, 1.0) -> y ~ (+0.970, -0.970) one cycle later.
    dut.ln_data_1_in.value = to_fixed(3.0)
    dut.ln_data_2_in.value = to_fixed(1.0)
    dut.ln_valid_1_in.value = 1
    dut.ln_valid_2_in.value = 1
    await RisingEdge(dut.clk)
    dut.ln_valid_1_in.value = 0
    dut.ln_valid_2_in.value = 0

    await RisingEdge(dut.clk)
    assert dut.ln_valid_1_out.value.integer == 1, "valid_1_out did not assert 1 cycle after both-valid beat"
    assert dut.ln_valid_2_out.value.integer == 1, "valid_2_out did not assert 1 cycle after both-valid beat"
    exp1, exp2 = layernorm_spec(3.0, 1.0)
    got1 = from_fixed(dut.ln_data_1_out.value.integer)
    got2 = from_fixed(dut.ln_data_2_out.value.integer)
    assert abs(got1 - exp1) <= TOL, f"y1 = {got1:.5f}, expected {exp1:.5f}"
    assert abs(got2 - exp2) <= TOL, f"y2 = {got2:.5f}, expected {exp2:.5f}"

    # Next cycle: valids deassert AND data is cleared (single beat, not stuck).
    await RisingEdge(dut.clk)
    assert dut.ln_valid_1_out.value.integer == 0, "valid_1_out stuck high after single beat"
    assert dut.ln_valid_2_out.value.integer == 0, "valid_2_out stuck high after single beat"
    assert dut.ln_data_1_out.value.integer == 0, "data_1_out not cleared when invalid"
    assert dut.ln_data_2_out.value.integer == 0, "data_2_out not cleared when invalid"
    print("valid handshake test passed!")
