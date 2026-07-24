// Parameterized dump wrapper for the tpu_nxn full-chip tests (roadmap
// item 5d-2). Compiled with -Pdump.N=4 for the N=4 gate target. Outputs
// are left unconnected (results live in UB memory, read hierarchically
// by the cocotb test). Written by the harness author, not the agent.
`timescale 1ns/1ps
`default_nettype none

module dump #(
    parameter int N = 2
)();

    logic clk;
    logic rst;

    logic [15:0] ub_wr_host_data_in [0:N-1];
    logic ub_wr_host_valid_in [0:N-1];

    logic ub_rd_start_in;
    logic ub_rd_transpose;
    logic [8:0] ub_ptr_select;
    logic [15:0] ub_rd_addr_in;
    logic [15:0] ub_rd_row_size;
    logic [15:0] ub_rd_col_size;

    logic [15:0] learning_rate_in;
    logic [6:0] vpu_data_pathway;
    logic sys_switch_in;
    logic [15:0] vpu_leak_factor_in;
    logic [15:0] inv_batch_size_times_two_in;

    tpu_nxn #(
        .SYSTOLIC_ARRAY_WIDTH(N)
    ) tpu_nxn_inst (
        .clk(clk),
        .rst(rst),
        .ub_wr_host_data_in(ub_wr_host_data_in),
        .ub_wr_host_valid_in(ub_wr_host_valid_in),
        .ub_rd_start_in(ub_rd_start_in),
        .ub_rd_transpose(ub_rd_transpose),
        .ub_ptr_select(ub_ptr_select),
        .ub_rd_addr_in(ub_rd_addr_in),
        .ub_rd_row_size(ub_rd_row_size),
        .ub_rd_col_size(ub_rd_col_size),
        .learning_rate_in(learning_rate_in),
        .vpu_data_pathway(vpu_data_pathway),
        .sys_switch_in(sys_switch_in),
        .vpu_leak_factor_in(vpu_leak_factor_in),
        .inv_batch_size_times_two_in(inv_batch_size_times_two_in)
    );

    initial begin
        $dumpfile("tpu_nxn.vcd");
        $dumpvars(1, dump);
    end

endmodule
