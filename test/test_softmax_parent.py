"""Gate test for the agent-authored Softmax VPU stage
(src/softmax_parent.sv). Written by the harness author, not the agent —
same discipline as test_gelu_parent.py / test_layernorm_parent.py: this is
what makes the softmax loop task's fitness signal cover the NEW operator
numerically, per tinytpu-loop README "the gate grows with the design".

Spec under test (from src/softmax_parent.sv), Q8.8 fixed point:
    d   = x1 - x2                            (17-bit, exact)
    y1  = sigmoid(d) = 1 / (1 + exp(-d))     (32-segment LUT of sigmoid(k/4)
    y2  = 1 - y1                              over |d| in [0,8), 6-bit linear
                                              interpolation, |d| >= 8 clamps
                                              to 1.0; documented max error
                                              ~0.0021 < 1 LSB)
Outputs are registered: valid_out asserts one cycle after a beat on which
BOTH sm_valid_1_in and sm_valid_2_in were high; if either lane is invalid
the outputs are cleared and valids deassert (the two lanes form one
group). No overflow condition exists (the outputs saturate), so the
overflow flags must stay 0.

Unlike the upstream tests, the asserts here are LIVE — PYTHONOPTIMIZE is
empty (NOASSERT is never defined), so they fire.
"""

import math

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

FRAC_BITS = 8
LSB = 1.0 / (1 << FRAC_BITS)
# The agent documents a max LUT+interpolation error of ~0.0021 (< 1 LSB).
# 2 LSB covers that plus rounding and the input-quantization slack
# (sigmoid slope <= 0.25, so a 0.5-LSB d error moves y by <= 0.125 LSB)
# without hiding a real bug. Observed errors are printed so the tolerance
# can be tightened later.
TOL = 2 * LSB


def to_fixed(val, frac_bits=FRAC_BITS):
    """convert python float to signed 16-bit fixed-point (Q8.8)."""
    scaled = int(round(val * (1 << frac_bits)))
    return scaled & 0xFFFF


def from_fixed(val, frac_bits=FRAC_BITS):
    """convert signed 16-bit fixed-point to python float."""
    if val >= 1 << 15:
        val -= 1 << 16
    return float(val) / (1 << frac_bits)


def quantize(x):
    """the float value actually driven onto the bus for float x."""
    return from_fixed(to_fixed(x))


def softmax_spec(x1, x2):
    """the documented 2-lane softmax, in float."""
    d = x1 - x2
    y1 = 1.0 / (1.0 + math.exp(-d))
    return y1, 1.0 - y1


# Coverage: asymmetric pairs both ways (sigmoid symmetry), equal inputs
# (d = 0 -> exactly (0.5, 0.5)), a LUT boundary point (d = 0.25 = k/4),
# sub-segment interpolation (d = 0.05), mid-range, |d| near/at/above the
# 8.0 clamp, and Q8.8 input extremes. Specs are computed on the QUANTIZED
# inputs (what the DUT actually sees), not the float originals.
PAIRS = [
    (3.0, 1.0),            # d = +2
    (1.0, 3.0),            # d = -2 (swapped outputs of the above)
    (2.0, 2.0),            # d = 0 -> exactly (0.5, 0.5)
    (0.25, 0.0),           # d = +0.25, exact LUT entry k=1
    (-2.5, -1.0),          # d = -1.5
    (0.1, 0.05),           # d ~ +0.05, interpolation between segments
    (9.0, 0.0),            # d = +9 >= 8 -> clamp to (1.0, 0.0)
    (0.0, 9.0),            # d = -9 -> clamp to (0.0, 1.0)
    (3.5, -3.5),           # d = +7, just inside the clamp
    (127.996, -128.0),     # input extremes -> clamp
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
    dut.sm_valid_1_in.value = 0
    dut.sm_valid_2_in.value = 0
    dut.sm_data_1_in.value = 0
    dut.sm_data_2_in.value = 0
    await RisingEdge(dut.clk)
    dut.rst.value = 0

    results = []
    for cyc in range(len(pairs) + 3):
        if cyc < len(pairs):
            dut.sm_data_1_in.value = to_fixed(pairs[cyc][0])
            dut.sm_data_2_in.value = to_fixed(pairs[cyc][1])
            dut.sm_valid_1_in.value = 1
            dut.sm_valid_2_in.value = 1
        else:
            dut.sm_valid_1_in.value = 0
            dut.sm_valid_2_in.value = 0
        await RisingEdge(dut.clk)
        if dut.sm_valid_1_out.value.integer and dut.sm_valid_2_out.value.integer:
            results.append((from_fixed(dut.sm_data_1_out.value.integer),
                            from_fixed(dut.sm_data_2_out.value.integer)))

    return results


@cocotb.test()
async def test_softmax_parent_values(dut):
    """Paired beats: spec values on both lanes, live asserts."""
    results = await drive_and_collect(dut, PAIRS)

    assert len(results) == len(PAIRS), (
        f"expected {len(PAIRS)} valid outputs, got {len(results)}")

    max_err = 0.0
    for idx, ((x1, x2), (got1, got2)) in enumerate(zip(PAIRS, results)):
        exp1, exp2 = softmax_spec(quantize(x1), quantize(x2))
        err = max(abs(got1 - exp1), abs(got2 - exp2))
        max_err = max(max_err, err)
        print(f"pair[{idx}]: softmax({x1:.4f}, {x2:.4f}) expected "
              f"({exp1:.5f}, {exp2:.5f}), got ({got1:.5f}, {got2:.5f}), "
              f"err {err:.5f}")
        assert err <= TOL, (
            f"pair[{idx}]: softmax({x1}, {x2}) = ({got1:.5f}, {got2:.5f}), "
            f"spec ({exp1:.5f}, {exp2:.5f}), err {err:.5f} > {TOL:.5f}")
        # outputs must sum to 1 (one bit of slack for fixed-point rounding)
        assert abs(got1 + got2 - 1.0) <= LSB, (
            f"pair[{idx}]: outputs do not sum to 1 "
            f"({got1:.5f} + {got2:.5f})")

    # d = 0 must give exactly (0.5, 0.5): LUT[0] = 0.5, no interpolation.
    assert results[2] == (0.5, 0.5), (
        f"equal-input pair should give exactly (0.5, 0.5), got {results[2]}")
    print(f"value test passed! (max observed err {max_err:.5f})")


@cocotb.test()
async def test_softmax_parent_valid_handshake(dut):
    """Both-valid requirement, 1-cycle latency, deassert, data cleared."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    dut.sm_valid_1_in.value = 0
    dut.sm_valid_2_in.value = 0
    dut.sm_data_1_in.value = 0
    dut.sm_data_2_in.value = 0
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # No valid in -> no valid out.
    assert dut.sm_valid_1_out.value.integer == 0, "valid_1_out high with no valid input"
    assert dut.sm_valid_2_out.value.integer == 0, "valid_2_out high with no valid input"

    # Lane 1 only: the pair is incomplete, so NO valid output may appear.
    dut.sm_data_1_in.value = to_fixed(3.0)
    dut.sm_valid_1_in.value = 1
    await RisingEdge(dut.clk)
    dut.sm_valid_1_in.value = 0
    await RisingEdge(dut.clk)
    assert dut.sm_valid_1_out.value.integer == 0, "valid_1_out asserted on lane-1-only beat"
    assert dut.sm_valid_2_out.value.integer == 0, "valid_2_out asserted on lane-1-only beat"

    # Both lanes: one beat of (3.0, 1.0) -> y ~ (0.881, 0.119) one cycle later.
    dut.sm_data_1_in.value = to_fixed(3.0)
    dut.sm_data_2_in.value = to_fixed(1.0)
    dut.sm_valid_1_in.value = 1
    dut.sm_valid_2_in.value = 1
    await RisingEdge(dut.clk)
    dut.sm_valid_1_in.value = 0
    dut.sm_valid_2_in.value = 0

    await RisingEdge(dut.clk)
    assert dut.sm_valid_1_out.value.integer == 1, "valid_1_out did not assert 1 cycle after both-valid beat"
    assert dut.sm_valid_2_out.value.integer == 1, "valid_2_out did not assert 1 cycle after both-valid beat"
    exp1, exp2 = softmax_spec(3.0, 1.0)
    got1 = from_fixed(dut.sm_data_1_out.value.integer)
    got2 = from_fixed(dut.sm_data_2_out.value.integer)
    assert abs(got1 - exp1) <= TOL, f"y1 = {got1:.5f}, expected {exp1:.5f}"
    assert abs(got2 - exp2) <= TOL, f"y2 = {got2:.5f}, expected {exp2:.5f}"
    # no overflow condition exists: flags must stay 0 even on valid beats
    assert dut.sm_overflow_out_1.value.integer == 0, "overflow_1 asserted (softmax saturates, never overflows)"
    assert dut.sm_overflow_out_2.value.integer == 0, "overflow_2 asserted (softmax saturates, never overflows)"

    # Next cycle: valids deassert AND data is cleared (single beat, not stuck).
    await RisingEdge(dut.clk)
    assert dut.sm_valid_1_out.value.integer == 0, "valid_1_out stuck high after single beat"
    assert dut.sm_valid_2_out.value.integer == 0, "valid_2_out stuck high after single beat"
    assert dut.sm_data_1_out.value.integer == 0, "data_1_out not cleared when invalid"
    assert dut.sm_data_2_out.value.integer == 0, "data_2_out not cleared when invalid"
    print("valid handshake test passed!")
