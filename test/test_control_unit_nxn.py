"""Gate test for the agent-authored N-host-lane instruction decoder
(src/control_unit_nxn.sv, roadmap item 8a). Written by the harness
author, not the agent — same discipline as test_control_unit.py, per
tinytpu-loop README "the gate grows with the design".

Spec under test (from the item-8a task spec): the instruction word is
133 + 17*(N-2) bits. ALL legacy field positions (bits 0-132) are
unchanged — including the item-4 pathway extension {instr[132:130],
instr[97:94]} — and host lanes k >= 2 are appended at the top:

    data  lane k : instruction[133 + 16*(k-2) +: 16]
    valid lane k : instruction[133 + 16*(N-2) + (k-2)]

so a legacy 133-bit image zero-extends to valid=0 on every appended
lane (no spurious writes). Array lanes 0/1 mirror the legacy _1/_2
scalars. control_unit_nxn is purely combinational — drive + settle +
read. The Makefile's N=4 target overrides the width with
iverilog -Pcontrol_unit_nxn.SYSTOLIC_ARRAY_WIDTH=4 and exports
CU_NXN_N; the default target runs N=2 (drop-in for control_unit).

Unlike the upstream tests, the asserts here are LIVE — PYTHONOPTIMIZE is
empty (NOASSERT is never defined), so they fire.
"""

import os

import cocotb
from cocotb.triggers import Timer

N = int(os.environ.get("CU_NXN_N", "2"))
SETTLE_NS = 1


def data_bit(k):
    """LSB of appended host-data lane k (k >= 2)."""
    return 133 + 16 * (k - 2)


def valid_bit(k):
    """Appended host-valid bit for lane k (k >= 2)."""
    return 133 + 16 * (N - 2) + (k - 2)


async def drive(dut, instr):
    dut.instruction.value = instr
    await Timer(SETTLE_NS, units="ns")


@cocotb.test()
async def test_control_unit_nxn_legacy_fields(dut):
    """All-zeros plus every legacy field at a distinctive value; array
    lanes 0/1 must mirror the legacy scalars; appended lanes silent."""
    await drive(dut, 0)
    assert dut.sys_switch_in.value.integer == 0
    assert dut.ub_rd_start_in.value.integer == 0
    assert dut.ub_rd_transpose.value.integer == 0
    assert dut.ub_wr_host_valid_in_1.value.integer == 0
    assert dut.ub_wr_host_valid_in_2.value.integer == 0
    assert dut.ub_rd_col_size.value.integer == 0
    assert dut.ub_rd_row_size.value.integer == 0
    assert dut.ub_rd_addr_in.value.integer == 0
    assert dut.ub_ptr_select.value.integer == 0
    assert dut.ub_wr_host_data_in_1.value.integer == 0
    assert dut.ub_wr_host_data_in_2.value.integer == 0
    assert dut.vpu_data_pathway.value.integer == 0
    assert dut.inv_batch_size_times_two_in.value.integer == 0
    assert dut.vpu_leak_factor_in.value.integer == 0
    for k in range(N):
        assert dut.ub_wr_host_valid_in[k].value.integer == 0
        assert dut.ub_wr_host_data_in[k].value.integer == 0

    # Each legacy field at a distinctive non-overlapping value, all at
    # once — a legacy 130-bit image, so appended lanes must stay silent.
    instr = 0
    instr |= 0b10101                       # bits 0-4: 1-bit signals
    instr |= 0xBEEF << 5                   # ub_rd_col_size
    instr |= 0xCAFE << 21                  # ub_rd_row_size
    instr |= 0x1234 << 37                  # ub_rd_addr_in
    instr |= 0x155 << 53                   # ub_ptr_select (9 bits)
    instr |= 0xABCD << 62                  # ub_wr_host_data_in_1
    instr |= 0x9876 << 78                  # ub_wr_host_data_in_2
    instr |= 0b1100 << 94                  # legacy pathway: forward
    instr |= 0x5555 << 98                  # inv_batch_size_times_two_in
    instr |= 0xAAAA << 114                 # vpu_leak_factor_in
    assert instr < (1 << 130), "test bug: this is a legacy 130-bit image"
    await drive(dut, instr)

    assert dut.sys_switch_in.value.integer == 1
    assert dut.ub_rd_start_in.value.integer == 0
    assert dut.ub_rd_transpose.value.integer == 1
    assert dut.ub_wr_host_valid_in_1.value.integer == 0
    assert dut.ub_wr_host_valid_in_2.value.integer == 1
    assert dut.ub_rd_col_size.value.integer == 0xBEEF
    assert dut.ub_rd_row_size.value.integer == 0xCAFE
    assert dut.ub_rd_addr_in.value.integer == 0x1234
    assert dut.ub_ptr_select.value.integer == 0x155
    assert dut.ub_wr_host_data_in_1.value.integer == 0xABCD
    assert dut.ub_wr_host_data_in_2.value.integer == 0x9876
    assert dut.vpu_data_pathway.value.integer == 0b0001100
    assert dut.inv_batch_size_times_two_in.value.integer == 0x5555
    assert dut.vpu_leak_factor_in.value.integer == 0xAAAA

    # Array lanes 0/1 mirror the legacy scalars; lanes >= 2 silent
    # (zero-extension).
    assert dut.ub_wr_host_valid_in[0].value.integer == 0
    assert dut.ub_wr_host_data_in[0].value.integer == 0xABCD
    assert dut.ub_wr_host_valid_in[1].value.integer == 1
    assert dut.ub_wr_host_data_in[1].value.integer == 0x9876
    for k in range(2, N):
        assert dut.ub_wr_host_valid_in[k].value.integer == 0, (
            f"lane {k} valid set by a legacy 130-bit image — "
            f"zero-extension broken")
        assert dut.ub_wr_host_data_in[k].value.integer == 0
    print("legacy field decode passed!")


@cocotb.test()
async def test_control_unit_nxn_pathway_extension(dut):
    """The item-4 top bits [132:130] still become pathway [6:4]."""
    for top, expected in [
        (0b000, 0b0001100),   # zero-extension == legacy behavior
        (0b001, 0b0011100),   # instruction[130] -> pathway[4] = gelu
        (0b010, 0b0101100),   # instruction[131] -> pathway[5] = layernorm
        (0b100, 0b1001100),   # instruction[132] -> pathway[6] = softmax
        (0b111, 0b1111100),   # all three new stages at once
    ]:
        instr = (top << 130) | (0b1100 << 94)
        await drive(dut, instr)
        got = dut.vpu_data_pathway.value.integer
        assert got == expected, (
            f"top={top:03b}: pathway {got:07b}, expected {expected:07b}")
    print("pathway extension decode passed!")


@cocotb.test()
async def test_control_unit_nxn_appended_lanes(dut):
    """Appended host lanes k >= 2: exact per-lane decode, no bleed."""
    if N == 2:
        print("N=2: no appended lanes — skipped (legacy drop-in)")
        return

    # Each appended lane independently: distinctive data, only its valid.
    # Data value must fit in 16 bits: use 0x1000 * (k-1) so k=15 gives
    # 0xE000 (not 0x10000 which would overflow and bleed into the next
    # lane's valid bit — caught at N=16 where k=15 is reached).
    for k in range(2, N):
        data_val = 0x1000 * (k - 1)
        instr = data_val << data_bit(k)
        instr |= 1 << valid_bit(k)
        await drive(dut, instr)
        for j in range(N):
            exp_v = 1 if j == k else 0
            exp_d = data_val if j == k else 0
            assert dut.ub_wr_host_valid_in[j].value.integer == exp_v, (
                f"lane {k} driven: lane {j} valid "
                f"{dut.ub_wr_host_valid_in[j].value.integer}, "
                f"expected {exp_v}")
            assert dut.ub_wr_host_data_in[j].value.integer == exp_d, (
                f"lane {k} driven: lane {j} data "
                f"{dut.ub_wr_host_data_in[j].value.integer:#06x}, "
                f"expected {exp_d:#06x}")
        # Appended bits must not bleed into ANY legacy field.
        assert dut.sys_switch_in.value.integer == 0
        assert dut.ub_rd_col_size.value.integer == 0
        assert dut.ub_rd_row_size.value.integer == 0
        assert dut.ub_rd_addr_in.value.integer == 0
        assert dut.ub_ptr_select.value.integer == 0
        assert dut.vpu_data_pathway.value.integer == 0
        assert dut.inv_batch_size_times_two_in.value.integer == 0
        assert dut.vpu_leak_factor_in.value.integer == 0

    # All lanes at once (a full N-lane host-write beat): legacy lanes
    # from the legacy fields, appended lanes from the appended bits.
    instr = 0b11000                        # both legacy host valids
    instr |= 0x1111 << 62                  # data lane 0
    instr |= 0x2222 << 78                  # data lane 1
    for k in range(2, N):
        instr |= (0x1000 * (k - 1)) << data_bit(k)
        instr |= 1 << valid_bit(k)
    await drive(dut, instr)
    exp_data = [0x1111, 0x2222] + [0x1000 * (k - 1) for k in range(2, N)]
    for j in range(N):
        assert dut.ub_wr_host_valid_in[j].value.integer == 1, (
            f"full beat: lane {j} valid not set")
        assert dut.ub_wr_host_data_in[j].value.integer == exp_data[j], (
            f"full beat: lane {j} data "
            f"{dut.ub_wr_host_data_in[j].value.integer:#06x}, "
            f"expected {exp_data[j]:#06x}")
    print("appended host-lane decode passed!")
