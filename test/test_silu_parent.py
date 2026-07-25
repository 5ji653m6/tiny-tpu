"""Gate test for the agent-authored SiLU VPU stage (src/silu_parent.sv /
src/silu_child.sv), roadmap item 13. Written by the harness author, not the
agent — per tinytpu-loop README "the gate grows with the design". Red-first:
FAILS until the silu RTL lands (BASELINE_EXCLUDE'd until then).

SiLU (x * sigmoid(x)) is THE diffusion activation: U-Net denoiser blocks and
the DiT timestep-embedding MLP both use it. The spec under test is pinned
here exactly, built entirely from already-gated arithmetic (item-7a exp LUT,
item-11-validated fxp_div / fxp_mul):

    a  = |x|                       (17-bit two's-complement magnitude)
    e  = exp_lut_exact(a)          (the softmax_group_nxn LUT, verbatim)
    d  = 256 + e                   (1.0 + e in Q8.8, always in [1.0, 2.0])
    num = 256 if x >= 0 else e     (sigmoid = 1/(1+e^-x)  /  e^x/(1+e^x))
    sigma = fxp_div(num, d)        (WIIA=8, WIIB=9, ROUND=1)
    silu = fxp_mul(x, sigma)       (Q8.8, clamp16)

Because the divider always sees the SMALL exponential, the divisor stays in
[1.0, 2.0] — no widened geometry beyond WIIB=9. Saturation falls out of the
LUT clamp for free: |x| >= 8.0 -> e = 0 -> sigma = 1.0 (x >= 0) or 0.0
(x < 0), so silu(x) = x exactly / 0 exactly. There is NO overflow condition
(|silu(x)| <= |x|), so silu_overflow_out is tied to 0.

Handshake (mirrors gelu_child): outputs are registered; valid_out asserts
one cycle after valid_in; when valid_in is low, valid_out deasserts and data
clears to 0. Single-cycle latency, fully pipelined.

The asserts are LIVE (PYTHONOPTIMIZE is empty) and EXACT — raw 16-bit
integer equality against the model, no tolerances.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

# The exact exp LUT model (EXP_LUT + segment interpolation + clamp), shared
# with the softmax group-stage gate tests.
from test_softmax_group_nxn import exp_lut_exact

FRAC_BITS = 8
LSB = 1.0 / (1 << FRAC_BITS)


def to_fixed(val, frac_bits=FRAC_BITS):
    """convert python float to signed 16-bit fixed-point (Q8.8)."""
    scaled = int(round(val * (1 << frac_bits)))
    return scaled & 0xFFFF


def from_fixed(val, frac_bits=FRAC_BITS):
    """convert signed 16-bit fixed-point (two's-complement raw) to float."""
    if val >= 1 << 15:
        val -= 1 << 16
    return float(val) / (1 << frac_bits)


def clamp16(v):
    """Saturating narrow to signed 16-bit (fxp_zoom's WOI<WIF clamp)."""
    return max(-(1 << 15), min((1 << 15) - 1, v))


def fxp_mul_exact(a_raw, b_raw):
    """fxp_mul ROUND=1 on signed Q8.8 raws: clamp16((a*b + 0x80) >> 8),
    arithmetic shift. Validated beat-exact against src/fixedpoint.sv in
    item 11 (the increment guard coincides with clamping)."""
    return clamp16((a_raw * b_raw + 0x80) >> 8)


def fxp_div_exact(num_raw, den_raw):
    """fxp_div ROUND=1 on UNSIGNED Q8.8 raws (num in [0,1.0], den in
    [1.0,2.0]): round-to-nearest, ties round DOWN (2*rem must STRICTLY
    exceed den — from the RTL's `acct - divd < divd - acc`). Validated
    beat-exact in item 11."""
    q = (num_raw << 8) // den_raw
    rem = (num_raw << 8) - q * den_raw
    if q != 0xFFFF and 2 * rem > den_raw:
        q += 1
    return q


def silu_exact(x_raw):
    """The pinned SiLU spec on a signed Q8.8 raw — exact integer model."""
    a = -x_raw if x_raw < 0 else x_raw          # |x| (fits: clamp case below)
    e = exp_lut_exact(a)                        # 0 when a >= 2048 (|x| >= 8)
    den = 256 + e                               # 1.0 + e, in [1.0, 2.0]
    num = 256 if x_raw >= 0 else e              # sigmoid numerator
    sigma = fxp_div_exact(num, den)
    return fxp_mul_exact(x_raw, sigma)


# Coverage, as raw Q8.8 integers (exact by construction): zero, small
# fractions, the interpolation interior (odd fracs hit the LUT lerp),
# quarter/half/integer grid points, the |x| = 8.0 clamp boundary on both
# sides, deep saturation, and the full-scale rails.
COL1_INPUTS = [0, 64, 100, 128, 256, 384, 512, 768, 1024, 1337, 1536,
               2047, 2048, 2049, 3000, 32767]
COL2_INPUTS = [-32, -100, -128, -256, -512, -777, -1024, -1536,
               -2047, -2048, -2049, -3000, -8192, -16384, -32768, 1]


async def drive_and_collect(dut, col1_inputs, col2_inputs):
    """Drive both columns in lockstep, collect valid outputs per column.

    NOTE (cocotb/Icarus read timing): a read right after `await RisingEdge`
    sees the pre-edge state — the registered output for an input sampled at
    edge N is only visible to reads after edge N+1. Outputs must therefore
    be sampled EVERY cycle, including while inputs are still being driven.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    dut.silu_valid_1_in.value = 0
    dut.silu_valid_2_in.value = 0
    dut.silu_data_1_in.value = 0
    dut.silu_data_2_in.value = 0
    await RisingEdge(dut.clk)
    dut.rst.value = 0

    col1_results, col2_results = [], []
    n = max(len(col1_inputs), len(col2_inputs))
    for cyc in range(n + 3):
        if cyc < n:
            if cyc < len(col1_inputs):
                dut.silu_data_1_in.value = col1_inputs[cyc]
                dut.silu_valid_1_in.value = 1
            if cyc < len(col2_inputs):
                dut.silu_data_2_in.value = col2_inputs[cyc] & 0xFFFF
                dut.silu_valid_2_in.value = 1
        else:
            dut.silu_valid_1_in.value = 0
            dut.silu_valid_2_in.value = 0
        await RisingEdge(dut.clk)
        if dut.silu_valid_1_out.value.integer:
            col1_results.append(dut.silu_data_1_out.value.integer)
        if dut.silu_valid_2_out.value.integer:
            col2_results.append(dut.silu_data_2_out.value.integer)

    return col1_results, col2_results


def to_signed(raw):
    """interpret an unsigned 16-bit readback as two's-complement."""
    return raw - (1 << 16) if raw >= 1 << 15 else raw


@cocotb.test()
async def test_silu_parent_exact_values(dut):
    """Both columns: EXACT raw equality against the integer model — zero,
    lerp interior, clamp boundary, saturation, and the rails."""
    col1_results, col2_results = await drive_and_collect(dut, COL1_INPUTS, COL2_INPUTS)

    assert len(col1_results) == len(COL1_INPUTS), (
        f"col1: expected {len(COL1_INPUTS)} valid outputs, got {len(col1_results)}")
    assert len(col2_results) == len(COL2_INPUTS), (
        f"col2: expected {len(COL2_INPUTS)} valid outputs, got {len(col2_results)}")

    for col, inputs, results in (("col1", COL1_INPUTS, col1_results),
                                 ("col2", COL2_INPUTS, col2_results)):
        for idx, (x_raw, got_raw) in enumerate(zip(inputs, results)):
            x_s = to_signed(x_raw & 0xFFFF)
            exp_raw = silu_exact(x_s) & 0xFFFF
            print(f"{col}[{idx}]: silu({from_fixed(x_s):+.4f}) "
                  f"expected {from_fixed(exp_raw):+.5f} (raw {exp_raw:#06x}), "
                  f"got {from_fixed(got_raw):+.5f} (raw {got_raw:#06x})")
            assert got_raw == exp_raw, (
                f"{col}[{idx}]: silu({from_fixed(x_s):+.4f}) = raw "
                f"{got_raw:#06x}, model {exp_raw:#06x} — EXACT match required")

    # Documented saturation: |x| >= 8.0 -> silu(x) = x exactly (x >= 0).
    for idx in (12, 13, 14, 15):  # 2048, 2049, 3000, 32767
        assert col1_results[idx] == COL1_INPUTS[idx], (
            f"positive saturation broken at raw {COL1_INPUTS[idx]:#06x}: "
            f"got {col1_results[idx]:#06x}")
    # ... and silu(x) = 0 exactly (x < 0, |x| >= 8.0).
    for idx in (9, 10, 11, 12, 13, 14):  # -2048 .. -32768
        assert col2_results[idx] == 0, (
            f"negative saturation broken at raw {COL2_INPUTS[idx] & 0xFFFF:#06x}: "
            f"got {col2_results[idx]:#06x}")
    # silu(0) = 0 (sigmoid midpoint 0.5 multiplied by x = 0).
    assert col1_results[0] == 0, "silu(0) != 0"
    print("exact value test passed!")


@cocotb.test()
async def test_silu_parent_valid_handshake(dut):
    """valid_out follows valid_in by one cycle; invalid input -> invalid,
    cleared output; overflow outputs are tied low."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    dut.silu_valid_1_in.value = 0
    dut.silu_valid_2_in.value = 0
    dut.silu_data_1_in.value = 0
    dut.silu_data_2_in.value = 0
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # No valid in -> no valid out.
    assert dut.silu_valid_1_out.value.integer == 0, "valid_1_out high with no valid input"
    assert dut.silu_valid_2_out.value.integer == 0, "valid_2_out high with no valid input"

    # One valid beat on col1 only: silu(1.0) = 1 * sigmoid(1).
    # Model: e = exp_lut_exact(256) = 94, den = 350, sigma = div(256, 350).
    sigma = fxp_div_exact(256, 256 + exp_lut_exact(256))
    expected = fxp_mul_exact(256, sigma)
    dut.silu_data_1_in.value = 256
    dut.silu_valid_1_in.value = 1
    await RisingEdge(dut.clk)
    dut.silu_valid_1_in.value = 0

    # One cycle later: col1 valid with the exact model value; col2 idle.
    await RisingEdge(dut.clk)
    assert dut.silu_valid_1_out.value.integer == 1, (
        "valid_1_out did not assert 1 cycle after valid_1_in")
    got = dut.silu_data_1_out.value.integer
    assert got == expected & 0xFFFF, (
        f"silu(1.0) = {from_fixed(got):+.5f} (raw {got:#06x}), "
        f"model {from_fixed(expected):+.5f} (raw {expected & 0xFFFF:#06x})")
    assert dut.silu_valid_2_out.value.integer == 0, "valid_2_out asserted without valid_2_in"

    # Next cycle: col1 valid must deassert and data clear (single beat).
    await RisingEdge(dut.clk)
    assert dut.silu_valid_1_out.value.integer == 0, "valid_1_out stuck high after single input beat"
    assert dut.silu_data_1_out.value.integer == 0, "data_1_out not cleared when invalid"

    # No overflow condition exists (|silu(x)| <= |x|): both overflow pins low.
    assert dut.silu_overflow_out_1.value.integer == 0, "overflow_out_1 high"
    assert dut.silu_overflow_out_2.value.integer == 0, "overflow_out_2 high"
    print("valid handshake test passed!")
