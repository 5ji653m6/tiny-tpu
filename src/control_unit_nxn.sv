`timescale 1ns/1ps
`default_nettype none

// N-lane control unit instruction decoder, generalizing the hardcoded
// 2-lane control_unit.sv to SYSTOLIC_ARRAY_WIDTH = N host lanes
// (N >= 2). Purely combinational -- no clock.
//
// Instruction word: 134 + 17*(N-2) bits total, where
// N = SYSTOLIC_ARRAY_WIDTH. All legacy field positions (bits 0-132)
// are IDENTICAL to control_unit.sv; the extra host lanes are appended
// at the top of the word, following the item-4 ISA precedent (bits
// 132-130 were appended for the VPU pathway extension the same way),
// and the item-13 SiLU pathway bit is appended as the new MSB.
// Legacy 133-bit instruction images zero-extend the appended region
// and keep working: every appended lane decodes valid = 0, so there
// are no spurious host writes, and the SiLU stage decodes bypassed.
//
// Field layout (legacy region, bits 0-132, unchanged):
//   bits 0-4     : 1-bit signals (sys_switch_in, ub_rd_start_in, ub_rd_transpose,
//                  ub_wr_host_valid_in_1, ub_wr_host_valid_in_2)
//   bits 5-20    : ub_rd_col_size [15:0]
//   bits 21-36   : ub_rd_row_size [15:0]
//   bits 37-52   : ub_rd_addr_in [15:0]
//   bits 53-61   : ub_ptr_select [8:0]
//   bits 62-77   : ub_wr_host_data_in_1 [15:0]
//   bits 78-93   : ub_wr_host_data_in_2 [15:0]
//   bits 94-97   : vpu_data_pathway [3:0] (legacy stages: bias(3), lr(2), loss(1), lr_d(0))
//   bits 98-113  : inv_batch_size_times_two_in [15:0]
//   bits 114-129 : vpu_leak_factor_in [15:0]
//   bits 130-132 : vpu_data_pathway [6:4] (new stages: gelu(4), ln(5), sm(6))
//   bit  133+17*(N-2) (the MSB): vpu_data_pathway [7] (item-13 SiLU stage)
//
// Field layout (appended region, bits 133 and up, lane k >= 2):
//   lane k data  : bits [133+16*(k-2) +: 16]   (ub_wr_host_data_in[k])
//   lane k valid : bit  133+16*(N-2)+(k-2)     (ub_wr_host_valid_in[k])
// i.e. (N-2) 16-bit data words first, then (N-2) 1-bit valids:
//   N = 3: data[2] at 133-148,        valid[2] at 149
//   N = 4: data[2] at 133-148, data[3] at 149-164,
//          valid[2] at 165,     valid[3] at 166
//
// Host-lane array ports: ub_wr_host_valid_in[k] / ub_wr_host_data_in[k]
// carry host lane k; lanes 0 and 1 alias the legacy scalar outputs
// ub_wr_host_valid_in_1/_2 and ub_wr_host_data_in_1/_2 (same decode).
//
// vpu_data_pathway is 8 bits: |silu(7)|sm(6)|ln(5)|gelu(4)|bias(3)|lr(2)|loss(1)|lr_d(0)|
// (1 = stage enabled, 0 = bypassed). Legacy 4-bit encodings zero-extend
// into bits [3:0]:
//   0b0000 = bypass     (all stages bypassed)
//   0b1100 = forward    (bias + leaky relu)
//   0b1111 = transition (bias + leaky relu + loss + leaky relu derivative)
//   0b0001 = backward   (leaky relu derivative)
module control_unit_nxn #(
    parameter int SYSTOLIC_ARRAY_WIDTH = 2
) (
    // 134 + 17*(N-2) bits total; the SiLU pathway bit is the MSB
    input logic [133+17*(SYSTOLIC_ARRAY_WIDTH-2):0] instruction,

    // 1-bit signals - 5
    output logic sys_switch_in,
    output logic ub_rd_start_in,
    output logic ub_rd_transpose,
    output logic ub_wr_host_valid_in_1,
    output logic ub_wr_host_valid_in_2,

    output logic [15:0] ub_rd_col_size,

    output logic [15:0] ub_rd_row_size,

    output logic [15:0] ub_rd_addr_in,

    output logic [8:0] ub_ptr_select,

    //16 bit signals
    output logic [15:0] ub_wr_host_data_in_1,
    output logic [15:0] ub_wr_host_data_in_2,

    // 8-bit signal (|silu|sm|ln|gelu|bias|lr|loss|lr_d|, 1 bit per VPU stage, 0 = bypass)
    output logic [7:0] vpu_data_pathway,

    //16-bit signals
    output logic [15:0] inv_batch_size_times_two_in,
    output logic [15:0] vpu_leak_factor_in,

    // N-wide host lanes: lane 0 = legacy _1 signals, lane 1 = legacy _2
    // signals, lanes k >= 2 decode from the appended top bits.
    output wire ub_wr_host_valid_in [SYSTOLIC_ARRAY_WIDTH],
    // Unsigned (like tpu_nxn's matching input and the tpu_nxn_ic
    // interconnect wire): slang requires exact element-type equality on
    // unpacked-array port connections; the bits are raw instruction
    // slices, signedness is transport-irrelevant.
    output wire [15:0] ub_wr_host_data_in [SYSTOLIC_ARRAY_WIDTH]
);

    localparam int N = SYSTOLIC_ARRAY_WIDTH;

    // bits 0-4: 1-bit signals
    assign sys_switch_in         = instruction[0];
    assign ub_rd_start_in        = instruction[1];
    assign ub_rd_transpose       = instruction[2];
    assign ub_wr_host_valid_in_1 = instruction[3];
    assign ub_wr_host_valid_in_2 = instruction[4];

    // bits 5-20: ub_rd_col_size [15:0]
    assign ub_rd_col_size = instruction[20:5];

    // bits 21-36: ub_rd_row_size [15:0]
    assign ub_rd_row_size = instruction[36:21];

    // bits 37-52: ub_rd_addr_in [15:0]
    assign ub_rd_addr_in = instruction[52:37];

    // bits 53-61: ub_ptr_select [8:0]
    assign ub_ptr_select = instruction[61:53];

    // bits 62-77: ub_wr_host_data_in_1 [15:0]
    assign ub_wr_host_data_in_1 = instruction[77:62];

    // bits 78-93: ub_wr_host_data_in_2 [15:0]
    assign ub_wr_host_data_in_2 = instruction[93:78];

    // bits 94-97 + 130-132 + MSB: vpu_data_pathway [7:0]
    // legacy stages in bits 94-97, item-4 stages (gelu/ln/sm) in bits
    // 130-132, item-13 SiLU appended as the new MSB
    assign vpu_data_pathway = {instruction[133+17*(SYSTOLIC_ARRAY_WIDTH-2)],
                               instruction[132:130], instruction[97:94]};

    // bits 98-113: inv_batch_size_times_two_in [15:0]
    assign inv_batch_size_times_two_in = instruction[113:98];

    // bits 114-129: vpu_leak_factor_in [15:0]
    assign vpu_leak_factor_in = instruction[129:114];

    genvar gk;
    generate
        if (N >= 2) begin : g_lanes
            // BUG-TOOLS-1 (iverilog 11): variable unpacked-array
            // output ports propagate X to connected parent nets; the
            // public ports are wires assigned from these mirrors.
            logic ub_wr_host_valid_in_r [N];             // BUG-TOOLS-1 mirror
            logic [15:0] ub_wr_host_data_in_r [N];       // BUG-TOOLS-1 mirror

            assign ub_wr_host_valid_in = ub_wr_host_valid_in_r;
            assign ub_wr_host_data_in  = ub_wr_host_data_in_r;

            // N-wide host lanes. Lanes 0/1 decode from the legacy fields
            // (identical to the scalar _1/_2 outputs above); lanes k >= 2
            // decode from the appended top bits.
            assign ub_wr_host_valid_in_r[0] = instruction[3];
            assign ub_wr_host_data_in_r[0]  = instruction[77:62];
            assign ub_wr_host_valid_in_r[1] = instruction[4];
            assign ub_wr_host_data_in_r[1]  = instruction[93:78];

            for (gk = 2; gk < N; gk++) begin : g_appended_lane
                assign ub_wr_host_data_in_r[gk]  = instruction[133+16*(gk-2) +: 16];
                assign ub_wr_host_valid_in_r[gk] = instruction[133+16*(N-2)+(gk-2)];
            end
        end else begin : g_bad_width
            // N >= 2 is required: lanes 0/1 are hardwired to the legacy
            // fields. Fail loudly rather than silently mis-decoding.
            initial begin
                $error("control_unit_nxn: SYSTOLIC_ARRAY_WIDTH=%0d is not >= 2", N);
            end
        end
    endgenerate

endmodule
