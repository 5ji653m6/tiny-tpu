"""Gate test for src/vpu_nxn.sv (roadmap item 5c), N=2 equivalence half.
Written by the harness author, not the agent — per tinytpu-loop README
"the gate grows with the design".

The dump wrapper (test/dump_vpu_nxn_equiv.sv) instantiates the LEGACY vpu
and the new vpu_nxn side by side, sharing every input. This test drives
both through bursts of systolically-skewed beats (lane 0 leads lane 1 by
one cycle, matching the array's native dataflow) across all seven pathway
bits — bypass, bias+leaky_relu, +gelu, +layernorm, +softmax, the full
transition pathway (bias+lr+loss+lr_d), and the backward pathway (lr_d
alone, H_in sourced) — and asserts the nxn module's array ports equal the
legacy module's scalar ports CYCLE BY CYCLE. The legacy module is the
spec: any port-wiring, indexing, or skew-alignment mistake shows up as a
divergence within a few cycles.

Unlike the upstream tests, the asserts here are LIVE — PYTHONOPTIMIZE is
empty (NOASSERT is never defined), so they fire.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

N = 2

cycle_no = 0
armed = False  # compare only after reset has clocked the X's away
mismatches = []


def compare(dut):
    """Sample every output port on both instances; record any divergence."""
    global cycle_no
    if not armed:
        return
    cycle_no += 1
    for lane in range(N):
        # mask to 16 bits: the legacy ports are `signed` (cocotb returns a
        # negative int) while the nxn array ports are unsigned — compare
        # bit patterns, not signed interpretations
        legacy_d = getattr(
            dut.vpu_legacy, f"vpu_data_out_{lane + 1}").value.integer & 0xFFFF
        nxn_d = dut.vpu_nxn.vpu_data_out[lane].value.integer & 0xFFFF
        legacy_v = getattr(dut.vpu_legacy, f"vpu_valid_out_{lane + 1}").value.integer
        nxn_v = dut.vpu_nxn.vpu_valid_out[lane].value.integer
        if legacy_d != nxn_d:
            mismatches.append(
                f"cycle {cycle_no}: vpu_data_out[{lane}] "
                f"legacy={legacy_d:#06x} nxn={nxn_d:#06x}")
        if legacy_v != nxn_v:
            mismatches.append(
                f"cycle {cycle_no}: vpu_valid_out[{lane}] "
                f"legacy={legacy_v} nxn={nxn_v}")


async def tick(dut, cycles=1):
    for _ in range(cycles):
        await RisingEdge(dut.clk)
        compare(dut)


async def drive_idle(dut):
    for i in range(N):
        dut.vpu_data_in[i].value = 0
        dut.vpu_valid_in[i].value = 0
        dut.bias_scalar_in[i].value = 0
        dut.Y_in[i].value = 0
        dut.H_in[i].value = 0
    dut.vpu_data_pathway.value = 0


async def burst(dut, pathway, nbeats):
    """Drive nbeats of skewed beats (lane 0 at t, lane 1 at t+1) under the
    given pathway, then let the pipeline drain. Data values are distinct
    per lane and per beat so a lane swap or skew error changes the bits."""
    dut.vpu_data_pathway.value = pathway
    await tick(dut)  # let the pathway select settle before the stream
    for b in range(nbeats):
        dut.vpu_data_in[0].value = (0x0100 * (b + 1)) & 0xFFFF
        dut.vpu_valid_in[0].value = 1
        dut.vpu_data_in[1].value = 0
        dut.vpu_valid_in[1].value = 0
        await tick(dut)
        dut.vpu_data_in[0].value = 0
        dut.vpu_valid_in[0].value = 0
        dut.vpu_data_in[1].value = (0x0180 * (b + 1)) & 0xFFFF
        dut.vpu_valid_in[1].value = 1
        await tick(dut)
    dut.vpu_valid_in[0].value = 0
    dut.vpu_valid_in[1].value = 0
    dut.vpu_data_in[0].value = 0
    dut.vpu_data_in[1].value = 0
    await tick(dut, 12)  # drain the full stage chain (and re-skew regs)
    dut.vpu_data_pathway.value = 0
    await tick(dut, 2)


@cocotb.test()
async def test_vpu_nxn_equiv(dut):
    """Lockstep the legacy and nxn VPUs through every pathway; the nxn
    array ports must equal the legacy scalar ports every cycle."""
    global armed
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    await drive_idle(dut)
    dut.lr_leak_factor_in.value = 26     # 0.1 in Q8.8
    dut.inv_batch_size_times_two_in.value = 128  # 0.5 in Q8.8
    for i in range(N):
        dut.bias_scalar_in[i].value = 0x0020 * (i + 1)
        dut.Y_in[i].value = 0x0100 * (i + 1)
        dut.H_in[i].value = 0x0080 * (i + 1)
    await tick(dut, 2)
    dut.rst.value = 0
    await tick(dut)
    armed = True  # registers cleared; X-free comparisons from here on

    # All-bypass, then per-lane stages, then group stages, then the two
    # training pathways (transition uses the last_H cache; backward uses
    # H_in). Pathway encoding: |sm(6)|ln(5)|gelu(4)|bias(3)|lr(2)|loss(1)|lr_d(0)|
    await burst(dut, 0b0000000, 3)  # full bypass
    await burst(dut, 0b0001100, 3)  # bias + leaky relu (forward)
    await burst(dut, 0b0011100, 3)  # + gelu
    await burst(dut, 0b0101100, 3)  # bias + lr + layernorm (group stage)
    await burst(dut, 0b1001100, 3)  # bias + lr + softmax (group stage)
    await burst(dut, 0b0111100, 3)  # gelu + layernorm together
    await burst(dut, 0b1111100, 3)  # gelu + layernorm + softmax together
    await burst(dut, 0b0001111, 3)  # transition: bias+lr+loss+lr_d (last_H)
    await burst(dut, 0b0000001, 3)  # backward: lr_d alone, H_in source

    assert not mismatches, (
        f"{len(mismatches)} cycle-level mismatches between legacy and nxn "
        f"VPUs:\n" + "\n".join(mismatches[:20]))
    print(f"N=2 equivalence OK ({cycle_no} cycles compared, all pathways)")
