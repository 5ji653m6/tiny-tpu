"""Gate test for the agent-authored ISA extension (src/control_unit.sv).
Written by the harness author, not the agent — same discipline as
test_gelu_parent.py / test_layernorm_parent.py / test_softmax_parent.py:
this is what makes the isa loop task's fitness signal cover the NEW
instruction encoding, per tinytpu-loop README "the gate grows with the
design".

Spec under test (from src/control_unit.sv header): the instruction word is
133 bits; ALL legacy field positions (bits 0-129) are unchanged, and the
three new VPU pathway stage enables are appended at the top:

    vpu_data_pathway = {instruction[132:130], instruction[97:94]}

so legacy 130-bit instruction images (ints < 2**130) zero-extend bits
132-130 to 0, bypassing the gelu/layernorm/softmax stages, and keep
working. Pathway encoding: |sm(6)|ln(5)|gelu(4)|bias(3)|lr(2)|loss(1)|lr_d(0)|.

control_unit is purely combinational — no clock, drive + settle + read.

Unlike the upstream tests, the asserts here are LIVE — PYTHONOPTIMIZE is
empty (NOASSERT is never defined), so they fire.
"""

import cocotb
from cocotb.triggers import Timer

SETTLE_NS = 1


async def drive(dut, instr):
    dut.instruction.value = instr
    await Timer(SETTLE_NS, units="ns")


@cocotb.test()
async def test_control_unit_legacy_fields(dut):
    """All-zeros plus every legacy field at a distinctive value."""
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

    # Each field at a distinctive non-overlapping value, all at once.
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
    # legacy 4-bit pathway zero-extends into the 7-bit output
    assert dut.vpu_data_pathway.value.integer == 0b0001100
    assert dut.inv_batch_size_times_two_in.value.integer == 0x5555
    assert dut.vpu_leak_factor_in.value.integer == 0xAAAA
    print("legacy field decode passed!")


@cocotb.test()
async def test_control_unit_pathway_extension(dut):
    """The appended top bits [132:130] become pathway [6:4] (gelu/ln/sm)."""
    # Each new stage bit individually, on top of a legacy forward pathway.
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

    # New bits must not disturb the low 4 legacy pathway bits.
    instr = (0b101 << 130) | (0b0001 << 94)   # gelu+sm, legacy backward
    await drive(dut, instr)
    assert dut.vpu_data_pathway.value.integer == 0b1010001

    # New bits must not bleed into neighboring legacy fields.
    assert dut.inv_batch_size_times_two_in.value.integer == 0
    assert dut.vpu_leak_factor_in.value.integer == 0
    print("pathway extension decode passed!")
