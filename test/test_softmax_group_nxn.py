"""Gate test for the agent-authored N-lane group Softmax leaf
(src/softmax_group_nxn.sv, roadmap item 7a). Written by the harness
author, not the agent — same discipline as test_softmax_parent.py, per
tinytpu-loop README "the gate grows with the design".

Spec under test (from src/softmax_group_nxn.sv), Q8.8 fixed point,
N = 4 lanes aligned to one beat:
    m    = max_i x_i                          (signed)
    e_i  = x_i - m                            (17-bit Q9.8, <= 0)
    exp_i = piecewise-linear LUT of exp(-|e|) with segment geometry
            mirroring the legacy sigmoid LUT:
            lut[k] = round(256 * exp(-0.25k)), k = 0..32 (Q8.8)
            seg = |e|[10:6], frac = |e|[5:0]
            exp_i = lut[seg] + ((slope*frac + 32) >>> 6)  (arith shift)
            |e| >= 8.0 (raw 2048) clamps to 0
    y_i  = exp_i / sum_j exp_j                (fxp_div ROUND=1, Q8.8)
Outputs are registered: valid_out asserts one cycle after a beat on which
ALL N sm_valid_in lanes were high; otherwise outputs are cleared and
valids deassert. sum >= 1.0 always (the max lane contributes 1.0), so no
div-by-zero and y_i in [0, 1] with no overflow condition.

The golden models the exp LUT EXACTLY (raw-integer arithmetic identical
to the RTL), so only the final ROUND=1 division and output quantization
remain — the observed error should be at most ~1 LSB, and the tolerance
is tight. Degenerate cases are checked for EXACT values. Asserts are
LIVE (PYTHONOPTIMIZE is empty).
"""

import math

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

N = 4
FRAC_BITS = 8
LSB = 1.0 / (1 << FRAC_BITS)
# Only fxp_div ROUND=1 + output quantization stand between the exact
# LUT model and the RTL; 4 LSB is generous but not blind.
# Observed errors are printed so the tolerance can be tightened later.
TOL = 4 * LSB

EXP_LUT = [round(256 * math.exp(-0.25 * k)) for k in range(33)]


def to_fixed(val, frac_bits=FRAC_BITS):
    """convert python float to signed 16-bit fixed-point (Q8.8)."""
    scaled = int(round(val * (1 << frac_bits)))
    return scaled & 0xFFFF


def from_fixed(val, frac_bits=FRAC_BITS):
    """convert signed 16-bit fixed-point to python float."""
    if val >= 1 << 15:
        val -= 1 << 16
    return float(val) / (1 << frac_bits)


def exp_lut_exact(e_raw):
    """The RTL's piecewise-linear exp(-|e|) on a raw Q8.8 magnitude
    (e_raw >= 0), computed with the exact integer arithmetic of
    src/softmax_group_nxn.sv. Returns a raw Q8.8 unsigned value."""
    if e_raw >= 2048:  # |e| >= 8.0 clamps to 0
        return 0
    seg = e_raw >> 6
    frac = e_raw & 0x3F
    lut_lo = EXP_LUT[seg]
    lut_hi = EXP_LUT[seg + 1]
    slope = lut_hi - lut_lo  # signed, <= 0
    interp = slope * frac + 32
    interp_shift = interp >> 6  # Python >> is arithmetic, like RTL
    return lut_lo + interp_shift


def softmax_group_spec(xs):
    """the documented N-lane softmax, exp LUT modeled exactly."""
    raws = [to_fixed(x) if to_fixed(x) < (1 << 15) else to_fixed(x) - (1 << 16)
            for x in xs]
    m = max(raws)
    exps = [exp_lut_exact(m - r) for r in raws]  # e_i <= 0, |e| = m - r
    total = sum(exps)
    return [e / total for e in exps]


# Coverage: a spread group, all-equal (exact 1/4 each), clamp case
# (one hot lane -> exact (1,0,0,0)), all-negative, sub-segment
# fractions (exercise the interpolator), two maxima (tie), a group
# straddling a LUT segment boundary, and near-zero differences.
GROUPS = [
    (2.0, 1.0, 0.0, -1.0),
    (0.5, 0.5, 0.5, 0.5),      # exact 1/4 each
    (9.0, 0.0, 0.0, 0.0),      # |e| >= 8 clamps: exact (1, 0, 0, 0)
    (-1.0, -2.0, -3.0, -4.0),
    (0.1, 0.2, 0.05, 0.07),    # sub-segment fractions
    (1.5, 1.5, 0.5, 0.5),      # tie for max
    (0.26, 0.24, 0.0, 0.0),    # straddles the 0.25 segment boundary
    (0.0039, 0.0, 0.0, 0.0),   # 1-LSB differences
]


async def drive_and_collect(dut, groups):
    """Drive aligned N-lane beats, collect valid outputs.

    NOTE (cocotb/Icarus read timing): a read right after `await RisingEdge`
    sees the pre-edge state — the registered output for an input sampled at
    edge N is only visible to reads after edge N+1. Outputs must therefore
    be sampled EVERY cycle, including while inputs are still being driven.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    for i in range(N):
        dut.sm_valid_in[i].value = 0
        dut.sm_data_in[i].value = 0
    await RisingEdge(dut.clk)
    dut.rst.value = 0

    results = []
    for cyc in range(len(groups) + 3):
        if cyc < len(groups):
            for i in range(N):
                dut.sm_data_in[i].value = to_fixed(groups[cyc][i])
                dut.sm_valid_in[i].value = 1
        else:
            for i in range(N):
                dut.sm_valid_in[i].value = 0
        await RisingEdge(dut.clk)
        if all(dut.sm_valid_out[i].value.integer for i in range(N)):
            results.append(tuple(from_fixed(dut.sm_data_out[i].value.integer)
                                 for i in range(N)))

    return results


@cocotb.test()
async def test_softmax_group_nxn_values(dut):
    """Aligned group beats: spec values on all lanes, live asserts."""
    results = await drive_and_collect(dut, GROUPS)

    assert len(results) == len(GROUPS), (
        f"expected {len(GROUPS)} valid outputs, got {len(results)}")

    max_err = 0.0
    for idx, (xs, got) in enumerate(zip(GROUPS, results)):
        exp = softmax_group_spec(xs)
        err = max(abs(g - e) for g, e in zip(got, exp))
        max_err = max(max_err, err)
        print(f"group[{idx}]: softmax{xs} expected "
              f"({', '.join(f'{e:.5f}' for e in exp)}), got "
              f"({', '.join(f'{g:.5f}' for g in got)}), err {err:.5f}")
        assert err <= TOL, (
            f"group[{idx}]: softmax{xs} = {got}, spec {exp}, "
            f"err {err:.5f} > {TOL:.5f}")
        # partition-of-unity invariant
        assert abs(sum(got) - 1.0) <= N * TOL, (
            f"group[{idx}]: sum(y) = {sum(got):.5f} not ~1")

    # Exact degenerate cases.
    assert results[1] == (0.25, 0.25, 0.25, 0.25), (
        f"equal-input group should give exactly 1/4 each, got {results[1]}")
    assert results[2] == (1.0, 0.0, 0.0, 0.0), (
        f"clamp group should give exactly (1,0,0,0), got {results[2]}")
    print(f"value test passed! (max observed err {max_err:.5f})")


@cocotb.test()
async def test_softmax_group_nxn_valid_handshake(dut):
    """All-valid requirement, 1-cycle latency, deassert, data cleared."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    for i in range(N):
        dut.sm_valid_in[i].value = 0
        dut.sm_data_in[i].value = 0
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # No valid in -> no valid out.
    assert not any(dut.sm_valid_out[i].value.integer for i in range(N)), \
        "valid_out high with no valid input"

    # N-1 lanes only: the group is incomplete, so NO valid output may
    # appear.
    for i in range(N - 1):
        dut.sm_data_in[i].value = to_fixed(1.0)
        dut.sm_valid_in[i].value = 1
    await RisingEdge(dut.clk)
    for i in range(N - 1):
        dut.sm_valid_in[i].value = 0
    await RisingEdge(dut.clk)
    assert not any(dut.sm_valid_out[i].value.integer for i in range(N)), \
        "valid_out asserted on an (N-1)-lane beat"

    # All N lanes: one beat of (0.5, 0.5, 0.5, 0.5) -> exactly 1/4 each,
    # one cycle later.
    for i in range(N):
        dut.sm_data_in[i].value = to_fixed(0.5)
        dut.sm_valid_in[i].value = 1
    await RisingEdge(dut.clk)
    for i in range(N):
        dut.sm_valid_in[i].value = 0

    await RisingEdge(dut.clk)
    assert all(dut.sm_valid_out[i].value.integer for i in range(N)), \
        "valid_out did not assert 1 cycle after all-valid beat"
    for i in range(N):
        got = from_fixed(dut.sm_data_out[i].value.integer)
        assert abs(got - 0.25) <= TOL, f"y[{i}] = {got:.5f}, expected 0.25"

    # Next cycle: valids deassert AND data is cleared (single beat, not
    # stuck).
    await RisingEdge(dut.clk)
    assert not any(dut.sm_valid_out[i].value.integer for i in range(N)), \
        "valid_out stuck high after single beat"
    assert not any(dut.sm_data_out[i].value.integer for i in range(N)), \
        "data_out not cleared when invalid"
    print("valid handshake test passed!")
