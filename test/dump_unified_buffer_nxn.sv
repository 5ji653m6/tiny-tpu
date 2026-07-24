// N-parameterized dump wrapper for unified_buffer_nxn. The Makefile's N=4
// target overrides the width with -Pdump.N=4.
module dump #(
    parameter int N = 2
)();
    logic clk;
    logic rst;

    logic [15:0] ub_wr_data_in [N];
    logic ub_wr_valid_in [N];
    logic [15:0] ub_wr_host_data_in [N];
    logic ub_wr_host_valid_in [N];

    logic ub_rd_start_in;
    logic ub_rd_transpose;
    logic [8:0] ub_ptr_select;
    logic [15:0] ub_rd_addr_in;
    logic [15:0] ub_rd_row_size;
    logic [15:0] ub_rd_col_size;
    logic [15:0] learning_rate_in;

    unified_buffer_nxn #(
        .SYSTOLIC_ARRAY_WIDTH(N)
    ) ub_nxn (
        .clk(clk),
        .rst(rst),
        .ub_wr_data_in(ub_wr_data_in),
        .ub_wr_valid_in(ub_wr_valid_in),
        .ub_wr_host_data_in(ub_wr_host_data_in),
        .ub_wr_host_valid_in(ub_wr_host_valid_in),
        .ub_rd_start_in(ub_rd_start_in),
        .ub_rd_transpose(ub_rd_transpose),
        .ub_ptr_select(ub_ptr_select),
        .ub_rd_addr_in(ub_rd_addr_in),
        .ub_rd_row_size(ub_rd_row_size),
        .ub_rd_col_size(ub_rd_col_size),
        .learning_rate_in(learning_rate_in),
        .ub_rd_input_data_out(),
        .ub_rd_input_valid_out(),
        .ub_rd_weight_data_out(),
        .ub_rd_weight_valid_out(),
        .ub_rd_bias_data_out(),
        .ub_rd_Y_data_out(),
        .ub_rd_H_data_out(),
        .ub_rd_col_size_out(),
        .ub_rd_col_size_valid_out()
    );

    initial begin
        $dumpfile("unified_buffer_nxn.vcd");
        $dumpvars(1, dump);
    end
endmodule
