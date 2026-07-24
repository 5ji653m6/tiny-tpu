"""Gate test for the agent-authored N-lane group LayerNorm leaf
(src/layernorm_group_nxn.sv, roadmap item 7a). Written by the harness
author, not the agent — same discipline as test_layernorm_parent.py:
this is what makes the vpu_group_leaves loop task's fitness signal cover
the NEW operator numerically, per tinytpu-loop README "the gate grows
with the design".

Spec under test (from src/layernorm_group_nxn.sv), Q8.8 fixed point,
N = 4 lanes aligned to one beat:
    mean = (sum_i x_i) >>> 2                (truncating arithmetic shift)
    dev_i = x_i - mean                      (17-bit Q9.8, exact)
    var  = (sum_i dev_i^2) >>> 2            (fxp_mul ROUND=1, Q18.8)
    std  = sqrt(var + eps), eps = 1/16      (fxp_sqrt ROUND=1, Q8.8)
    y_i  = dev_i / std                      (fxp_div ROUND=1, Q8.8)
Outputs are registered: valid_out asserts one cycle after a beat on which
ALL N ln_valid_in lanes were high; if any lane is invalid the outputs are
cleared and valids deassert (the N lanes form one group). Overflow flags
are sticky until reset.

The golden models the truncating mean shift EXACTLY in raw integers
(Python >> is arithmetic, matching the RTL slice) and the rest of the
chain in float; the ROUND=1 ops compound to a few LSB, covered by TOL.
Unlike the upstream tests, the asserts here are LIVE — PYTHONOPTIMIZE is
empty (NOASSERT is never defined), so they fire.
"""

import math

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

N = 4
FRAC_BITS = 8
LSB = 1.0 / (1 << FRAC_BITS)
EPS = 1.0 / 16  # LN_EPS in the float domain
# The chain truncating mean + truncating var + three ROUND=1 ops
# (mul/sqrt/div) compounds; the legacy 2-lane test used 8 LSB for a
# shorter chain. 12 LSB leaves margin without hiding a real bug.
# Observed errors are printed so the tolerance can be tightened later.
TOL = 12 * LSB


def to_fixed(val, frac_bits=FRAC_BITS):
    """convert python float to signed 16-bit fixed-point (Q8.8)."""
    scaled = int(round(val * (1 << frac_bits)))
    return scaled & 0xFFFF


def from_fixed(val, frac_bits=FRAC_BITS):
    """convert signed 16-bit fixed-point to python float."""
    if val >= 1 << 15:
        val -= 1 << 16
    return float(val) / (1 << frac_bits)


def layernorm_group_spec(xs):
    """the documented N-lane layernorm.

    The mean replicates the RTL's truncating arithmetic shift on raw
    integers exactly; the rest is float.
    """
    raws = [to_fixed(x) if to_fixed(x) < (1 << 15) else to_fixed(x) - (1 << 16)
            for x in xs]
    mean_raw = sum(raws) >> 2  # arithmetic shift, exactly the RTL slice
    devs = [(r - mean_raw) / (1 << FRAC_BITS) for r in raws]
    var = sum(d * d for d in devs) / N
    std = math.sqrt(var + EPS)
    return [d / std for d in devs]


# Coverage: symmetric spreads, all-equal (eps floor, no div-by-zero),
# cross-zero groups, a group whose raw sum is NOT divisible by N
# (exercises the truncating mean shift), negative-heavy, one hot lane,
# and large-magnitude groups where y saturates.
GROUPS = [
    (3.0, 1.0, -1.0, -3.0),
    (2.0, 2.0, 2.0, 2.0),
    (0.5, -0.5, 0.25, -0.25),
    (0.1, 0.0, 0.0, 0.0),      # raw sum 26, mean truncates 6.5 -> 6
    (-1.0, -2.5, 0.25, 1.5),
    (8.0, 0.0, 0.0, 0.0),
    (64.0, -64.0, 32.0, -32.0),
    (-64.0, 64.0, -32.0, 32.0),
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
        dut.ln_valid_in[i].value = 0
        dut.ln_data_in[i].value = 0
    await RisingEdge(dut.clk)
    dut.rst.value = 0

    results = []
    for cyc in range(len(groups) + 3):
        if cyc < len(groups):
            for i in range(N):
                dut.ln_data_in[i].value = to_fixed(groups[cyc][i])
                dut.ln_valid_in[i].value = 1
        else:
            for i in range(N):
                dut.ln_valid_in[i].value = 0
        await RisingEdge(dut.clk)
        if all(dut.ln_valid_out[i].value.integer for i in range(N)):
            results.append(tuple(from_fixed(dut.ln_data_out[i].value.integer)
                                 for i in range(N)))

    return results


@cocotb.test()
async def test_layernorm_group_nxn_values(dut):
    """Aligned group beats: spec values on all lanes, live asserts."""
    results = await drive_and_collect(dut, GROUPS)

    assert len(results) == len(GROUPS), (
        f"expected {len(GROUPS)} valid outputs, got {len(results)}")

    max_err = 0.0
    for idx, (xs, got) in enumerate(zip(GROUPS, results)):
        exp = layernorm_group_spec(xs)
        err = max(abs(g - e) for g, e in zip(got, exp))
        max_err = max(max_err, err)
        print(f"group[{idx}]: ln{xs} expected "
              f"({', '.join(f'{e:.5f}' for e in exp)}), got "
              f"({', '.join(f'{g:.5f}' for g in got)}), err {err:.5f}")
        assert err <= TOL, (
            f"group[{idx}]: ln{xs} = {got}, spec {exp}, "
            f"err {err:.5f} > {TOL:.5f}")
        # zero-sum invariant: sum(dev) ~ 0 -> sum(y) ~ 0
        assert abs(sum(got)) <= N * TOL, (
            f"group[{idx}]: sum(y) = {sum(got):.5f} not ~0")

    # eps floor: the all-equal group must give exactly zero, not a
    # div-by-zero.
    assert results[1] == (0.0, 0.0, 0.0, 0.0), (
        f"equal-input group should normalize to all-zero, got {results[1]}")
    print(f"value test passed! (max observed err {max_err:.5f})")


@cocotb.test()
async def test_layernorm_group_nxn_valid_handshake(dut):
    """All-valid requirement, 1-cycle latency, deassert, data cleared."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    for i in range(N):
        dut.ln_valid_in[i].value = 0
        dut.ln_data_in[i].value = 0
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # No valid in -> no valid out.
    assert not any(dut.ln_valid_out[i].value.integer for i in range(N)), \
        "valid_out high with no valid input"

    # N-1 lanes only: the group is incomplete, so NO valid output may
    # appear.
    for i in range(N - 1):
        dut.ln_data_in[i].value = to_fixed(3.0)
        dut.ln_valid_in[i].value = 1
    await RisingEdge(dut.clk)
    for i in range(N - 1):
        dut.ln_valid_in[i].value = 0
    await RisingEdge(dut.clk)
    assert not any(dut.ln_valid_out[i].value.integer for i in range(N)), \
        "valid_out asserted on an (N-1)-lane beat"

    # All N lanes: one beat of (3.0, 1.0, -1.0, -3.0) one cycle later.
    for i, x in enumerate((3.0, 1.0, -1.0, -3.0)):
        dut.ln_data_in[i].value = to_fixed(x)
        dut.ln_valid_in[i].value = 1
    await RisingEdge(dut.clk)
    for i in range(N):
        dut.ln_valid_in[i].value = 0

    await RisingEdge(dut.clk)
    assert all(dut.ln_valid_out[i].value.integer for i in range(N)), \
        "valid_out did not assert 1 cycle after all-valid beat"
    exp = layernorm_group_spec((3.0, 1.0, -1.0, -3.0))
    for i in range(N):
        got = from_fixed(dut.ln_data_out[i].value.integer)
        assert abs(got - exp[i]) <= TOL, (
            f"y[{i}] = {got:.5f}, expected {exp[i]:.5f}")

    # Next cycle: valids deassert AND data is cleared (single beat, not
    # stuck).
    await RisingEdge(dut.clk)
    assert not any(dut.ln_valid_out[i].value.integer for i in range(N)), \
        "valid_out stuck high after single beat"
    assert not any(dut.ln_data_out[i].value.integer for i in range(N)), \
        "data_out not cleared when invalid"
    print("valid handshake test passed!")
