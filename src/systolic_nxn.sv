`timescale 1ns/1ps
`default_nettype none

// NxN systolic array, generalizing the hardcoded 2x2 in systolic.sv.
//
// Weight-stationary PEs (shadow + active weight buffers). Activation data
// flows left-to-right along rows, partial sums and weights flow top-to-bottom
// along columns.
//
// VALID PROPAGATION (BUG-SYS-1 lesson from systolic.sv): the valid stream is
// chained THROUGH the array - right along row 1, then down each column - so
// every column's bottom PE fires even when only input-matrix column 1 streams
// (the single-column backward-pass case). Do NOT add per-row start signals:
// when row r does stream, the UB staggers it exactly r-1 cycles behind row 1,
// so the chained valid is cycle-identical. sys_switch_in propagates the same
// way (right along row 1, then down each column).
//
// SWITCH vs FIRST BEAT: the legacy protocol pulses sys_switch_in on the same
// cycle as the first input beat. Inside a PE the shadow->active copy and the
// first MAC would then happen on the same clock edge, so the whole first
// wavefront would multiply against the OLD active weights (the legacy array
// emits a 0 first beat here; harmless in the full TPU only because the
// control unit's switch instruction lands several cycles before UB data).
// To make the coincident-switch protocol correct, the left-edge inputs
// (sys_start_1 and every row's sys_data_in, uniformly so the UB's row stagger
// is preserved) are registered once on entry: the switch wavefront then
// reaches every PE exactly one cycle ahead of the first data/valid wavefront.
module systolic_nxn #(
    parameter int SYSTOLIC_ARRAY_WIDTH = 2
)(
    input logic clk,
    input logic rst,

    // input signals from left side of systolic array (one per row; index 0 = row 1)
    input logic [15:0] sys_data_in [SYSTOLIC_ARRAY_WIDTH],
    input logic sys_start_1,   // start signal for row 1; chained through the array

    output logic [15:0] sys_data_out [SYSTOLIC_ARRAY_WIDTH],  // bottom-row psums, per column
    output logic sys_valid_out [SYSTOLIC_ARRAY_WIDTH],

    // input signals from top of systolic array (one per column)
    input logic [15:0] sys_weight_in [SYSTOLIC_ARRAY_WIDTH],
    input logic sys_accept_w [SYSTOLIC_ARRAY_WIDTH],  // per column; broadcast to every row in that column

    input logic sys_switch_in,  // copies shadow -> active weight buffers; propagates through the array

    input logic [15:0] ub_rd_col_size_in,
    input logic ub_rd_col_size_valid_in
);

    // One-cycle input delay stage (see SWITCH vs FIRST BEAT above)
    logic [15:0] sys_data_in_q [SYSTOLIC_ARRAY_WIDTH];
    logic sys_start_1_q;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            sys_start_1_q <= 1'b0;
            for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                sys_data_in_q[i] <= 16'b0;
            end
        end else begin
            sys_start_1_q <= sys_start_1;
            for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                sys_data_in_q[i] <= sys_data_in[i];
            end
        end
    end

    // Inter-PE wires, indexed [row][col] with 0-based genvars (row 0 = row 1).
    logic [15:0] pe_input_out  [SYSTOLIC_ARRAY_WIDTH][SYSTOLIC_ARRAY_WIDTH];  // left to right
    logic [15:0] pe_psum_out   [SYSTOLIC_ARRAY_WIDTH][SYSTOLIC_ARRAY_WIDTH];  // top to bottom
    logic [15:0] pe_weight_out [SYSTOLIC_ARRAY_WIDTH][SYSTOLIC_ARRAY_WIDTH];  // top to bottom
    logic        pe_switch_out [SYSTOLIC_ARRAY_WIDTH][SYSTOLIC_ARRAY_WIDTH];
    logic        pe_valid_out  [SYSTOLIC_ARRAY_WIDTH][SYSTOLIC_ARRAY_WIDTH];

    // PE columns to enable (one bit per column; bit c = column c+1)
    logic [SYSTOLIC_ARRAY_WIDTH-1:0] pe_enabled;

    genvar r, c;
    generate
        for (r = 0; r < SYSTOLIC_ARRAY_WIDTH; r++) begin : gen_row
            for (c = 0; c < SYSTOLIC_ARRAY_WIDTH; c++) begin : gen_col
                pe pe_rc (
                    .clk(clk),
                    .rst(rst),
                    .pe_enabled(pe_enabled[c]),

                    // valid: row 1 chains left-to-right from the (delayed)
                    // start; rows below chain down the column (BUG-SYS-1 fix).
                    .pe_valid_in(r == 0 ? (c == 0 ? sys_start_1_q
                                                  : pe_valid_out[0][c-1])
                                        : pe_valid_out[r-1][c]),
                    .pe_valid_out(pe_valid_out[r][c]),

                    // accept_w: per column, broadcast to every row in that column
                    // (legacy semantics: the UB holds it for the whole weight stream)
                    .pe_accept_w_in(sys_accept_w[c]),

                    // switch: propagates the same way as valid, but enters the
                    // array UNDELAYED so it leads the first data wavefront by
                    // exactly one cycle at every PE
                    .pe_switch_in(r == 0 ? (c == 0 ? sys_switch_in
                                                   : pe_switch_out[0][c-1])
                                         : pe_switch_out[r-1][c]),
                    .pe_switch_out(pe_switch_out[r][c]),

                    // activation data flows right along each row
                    .pe_input_in(c == 0 ? sys_data_in_q[r] : pe_input_out[r][c-1]),
                    .pe_input_out(pe_input_out[r][c]),

                    // partial sums flow down each column
                    .pe_psum_in(r == 0 ? 16'b0 : pe_psum_out[r-1][c]),
                    .pe_psum_out(pe_psum_out[r][c]),

                    // weights flow down each column
                    .pe_weight_in(r == 0 ? sys_weight_in[c] : pe_weight_out[r-1][c]),
                    .pe_weight_out(pe_weight_out[r][c]),

                    .pe_overflow_out()
                );
            end
        end
    endgenerate

    // bottom-row outputs, one per column
    always_comb begin
        for (int c = 0; c < SYSTOLIC_ARRAY_WIDTH; c++) begin
            sys_data_out[c]  = pe_psum_out[SYSTOLIC_ARRAY_WIDTH-1][c];
            sys_valid_out[c] = pe_valid_out[SYSTOLIC_ARRAY_WIDTH-1][c];
        end
    end

    // Column-enable control: default all columns enabled; a col_size command
    // with k in 1..N enables exactly columns 1..k. k == 0 or k > N disables all.
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            pe_enabled <= {SYSTOLIC_ARRAY_WIDTH{1'b1}};
        end else if (ub_rd_col_size_valid_in) begin
            if (ub_rd_col_size_in >= 16'd1 && ub_rd_col_size_in <= SYSTOLIC_ARRAY_WIDTH) begin
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    pe_enabled[i] <= (i < ub_rd_col_size_in);
                end
            end else begin
                pe_enabled <= {SYSTOLIC_ARRAY_WIDTH{1'b0}};
            end
        end
    end

endmodule
