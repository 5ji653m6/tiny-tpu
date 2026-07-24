`timescale 1ns/1ps
`default_nettype none

// Control unit instruction decoder.
//
// Instruction word: 133 bits total (0-132). All legacy field positions
// (bits 0-129) are unchanged; bits 132-130 are appended at the top and
// carry the three new VPU pathway stage enables. Legacy 130-bit
// instruction images zero-extend bits 132-130 to 0, bypassing the new
// stages, and keep working.
//
// Field layout:
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
//
// vpu_data_pathway is 7 bits: |sm(6)|ln(5)|gelu(4)|bias(3)|lr(2)|loss(1)|lr_d(0)|
// (1 = stage enabled, 0 = bypassed). Legacy 4-bit encodings zero-extend
// into bits [3:0]:
//   0b0000 = bypass     (all stages bypassed)
//   0b1100 = forward    (bias + leaky relu)
//   0b1111 = transition (bias + leaky relu + loss + leaky relu derivative)
//   0b0001 = backward   (leaky relu derivative)
module control_unit (
    input logic [132:0] instruction,  // 133 bits total (0-132)
    
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

    // 7-bit signal (|sm|ln|gelu|bias|lr|loss|lr_d|, 1 bit per VPU stage, 0 = bypass)
    output logic [6:0] vpu_data_pathway,

    //16-bit signals
    output logic [15:0] inv_batch_size_times_two_in,
    output logic [15:0] vpu_leak_factor_in
);

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
    
    // bits 94-97 + 130-132: vpu_data_pathway [6:0]
    // legacy stages in bits 94-97, new stages (gelu/ln/sm) appended in bits 130-132
    assign vpu_data_pathway = {instruction[132:130], instruction[97:94]};
    
    // bits 98-113: inv_batch_size_times_two_in [15:0]
    assign inv_batch_size_times_two_in = instruction[113:98];
    
    // bits 114-129: vpu_leak_factor_in [15:0]
    assign vpu_leak_factor_in = instruction[129:114];

endmodule