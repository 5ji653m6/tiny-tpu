// Equivalence harness (N=2): the legacy unified_buffer and the new
// unified_buffer_nxn share every input, so the cocotb test can drive both
// in lockstep and compare their read ports cycle-by-cycle. The legacy
// module IS the spec here.
module dump();
    logic clk;
    logic rst;

    logic [15:0] ub_wr_data_in [2];
    logic ub_wr_valid_in [2];
    logic [15:0] ub_wr_host_data_in [2];
    logic ub_wr_host_valid_in [2];

    logic ub_rd_start_in;
    logic ub_rd_transpose;
    logic [8:0] ub_ptr_select;
    logic [15:0] ub_rd_addr_in;
    logic [15:0] ub_rd_row_size;
    logic [15:0] ub_rd_col_size;
    logic [15:0] learning_rate_in;

    unified_buffer ub_legacy (
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
        .ub_rd_input_data_out_0(),
        .ub_rd_input_data_out_1(),
        .ub_rd_input_valid_out_0(),
        .ub_rd_input_valid_out_1(),
        .ub_rd_weight_data_out_0(),
        .ub_rd_weight_data_out_1(),
        .ub_rd_weight_valid_out_0(),
        .ub_rd_weight_valid_out_1(),
        .ub_rd_bias_data_out_0(),
        .ub_rd_bias_data_out_1(),
        .ub_rd_Y_data_out_0(),
        .ub_rd_Y_data_out_1(),
        .ub_rd_H_data_out_0(),
        .ub_rd_H_data_out_1(),
        .ub_rd_col_size_out(),
        .ub_rd_col_size_valid_out()
    );

    unified_buffer_nxn ub_nxn (
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
        $dumpfile("unified_buffer_nxn_equiv.vcd");
        $dumpvars(1, dump);
    end
endmodule
