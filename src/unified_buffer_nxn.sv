`timescale 1ns/1ps
`default_nettype none

module unified_buffer_nxn #(
    parameter int UNIFIED_BUFFER_WIDTH = 128,
    parameter int SYSTOLIC_ARRAY_WIDTH = 2
)(
    input logic clk,
    input logic rst,

    // Write ports from VPU to UB
    input logic [15:0] ub_wr_data_in [SYSTOLIC_ARRAY_WIDTH],
    input logic ub_wr_valid_in [SYSTOLIC_ARRAY_WIDTH],

    // Write ports from host to UB (for loading in parameters)
    input logic [15:0] ub_wr_host_data_in [SYSTOLIC_ARRAY_WIDTH],
    input logic ub_wr_host_valid_in [SYSTOLIC_ARRAY_WIDTH],

    // Read instruction input from instruction memory
    input logic ub_rd_start_in,
    input logic ub_rd_transpose,
    input logic [8:0] ub_ptr_select,
    input logic [15:0] ub_rd_addr_in,
    input logic [15:0] ub_rd_row_size,
    input logic [15:0] ub_rd_col_size,

    // Learning rate input
    input logic [15:0] learning_rate_in,

    // Read ports from UB to left side of systolic array
    output wire [15:0] ub_rd_input_data_out [SYSTOLIC_ARRAY_WIDTH],
    output wire ub_rd_input_valid_out [SYSTOLIC_ARRAY_WIDTH],

    // Read ports from UB to top of systolic array
    output wire [15:0] ub_rd_weight_data_out [SYSTOLIC_ARRAY_WIDTH],
    output wire ub_rd_weight_valid_out [SYSTOLIC_ARRAY_WIDTH],

    // Read ports from UB to bias modules in VPU
    output wire [15:0] ub_rd_bias_data_out [SYSTOLIC_ARRAY_WIDTH],

    // Read ports from UB to loss modules (Y matrices) in VPU
    output wire [15:0] ub_rd_Y_data_out [SYSTOLIC_ARRAY_WIDTH],

    // Read ports from UB to activation derivative modules (H matrices) in VPU
    output wire [15:0] ub_rd_H_data_out [SYSTOLIC_ARRAY_WIDTH],

    // Outputs to send number of columns to systolic array
    output logic [15:0] ub_rd_col_size_out,
    output logic ub_rd_col_size_valid_out
);

    // BUG-TOOLS-1 (iverilog 11): variable unpacked-array
    // output ports propagate X to connected parent nets; the
    // public ports are wires assigned from these mirrors.
    logic [15:0] ub_rd_input_data_out_r [SYSTOLIC_ARRAY_WIDTH];  // BUG-TOOLS-1 mirror
    logic ub_rd_input_valid_out_r [SYSTOLIC_ARRAY_WIDTH];  // BUG-TOOLS-1 mirror
    logic [15:0] ub_rd_weight_data_out_r [SYSTOLIC_ARRAY_WIDTH];  // BUG-TOOLS-1 mirror
    logic ub_rd_weight_valid_out_r [SYSTOLIC_ARRAY_WIDTH];  // BUG-TOOLS-1 mirror
    logic [15:0] ub_rd_bias_data_out_r [SYSTOLIC_ARRAY_WIDTH];  // BUG-TOOLS-1 mirror
    logic [15:0] ub_rd_Y_data_out_r [SYSTOLIC_ARRAY_WIDTH];  // BUG-TOOLS-1 mirror
    logic [15:0] ub_rd_H_data_out_r [SYSTOLIC_ARRAY_WIDTH];  // BUG-TOOLS-1 mirror

    assign ub_rd_input_data_out = ub_rd_input_data_out_r;
    assign ub_rd_input_valid_out = ub_rd_input_valid_out_r;
    assign ub_rd_weight_data_out = ub_rd_weight_data_out_r;
    assign ub_rd_weight_valid_out = ub_rd_weight_valid_out_r;
    assign ub_rd_bias_data_out = ub_rd_bias_data_out_r;
    assign ub_rd_Y_data_out = ub_rd_Y_data_out_r;
    assign ub_rd_H_data_out = ub_rd_H_data_out_r;

    logic [15:0] ub_memory [0:UNIFIED_BUFFER_WIDTH-1];

    logic [15:0] wr_ptr;

    // BUG-UB-3: row-major VPU-stream writeback state. Beats (row r, col i)
    // of a rows x w output stream arrive skewed (cycle base+r+i); the legacy
    // arrival-order wr_ptr scheme only coincided with row-major at N=2, so at
    // N>=3 stored matrices were wavefront-scrambled against every row-major
    // read walk. Each beat lands at stream_base + r*w + i, where w is the
    // active output width latched from the last weight (ptr-1) read.
    logic [15:0] wr_stream_base;
    logic [15:0] wr_stream_width;
    // BUG-UB-4: the NEXT pass's weight-load (ptr-1) read can fire while the
    // current pass's output stream is still draining (observed at N=2: the
    // W2^T load relatched w=1 during the final H1 beat, misplacing it).
    // Snapshot the width at stream start and use the snapshot for every
    // beat of that stream (and its gradient writeback).
    logic [15:0] wr_stream_width_snap;
    logic        wr_stream_active_d;
    logic [15:0] wr_beat_cnt [SYSTOLIC_ARRAY_WIDTH];
    // Row-major weight-gradient writeback: done beat (r,i) of lane i updates
    // ub_memory[grad_descent_ptr + r*w + i].
    logic [15:0] grad_done_cnt [SYSTOLIC_ARRAY_WIDTH];

    // Internal logic for reading inputs from UB to left side of systolic array
    logic [15:0] rd_input_ptr;
    logic [15:0] rd_input_row_size;
    logic [15:0] rd_input_col_size;
    logic [15:0] rd_input_time_counter;
    logic rd_input_transpose;

    // Internal logic for reading weights from UB to left side of systolic array
    logic signed [15:0] rd_weight_ptr;
    logic [15:0] rd_weight_row_size;
    logic [15:0] rd_weight_col_size;
    logic [15:0] rd_weight_time_counter;
    logic rd_weight_transpose;
    logic [15:0] rd_weight_skip_size;

    // Internal logic for bias inputs from UB to bias modules in VPU
    logic [15:0] rd_bias_ptr;
    logic [15:0] rd_bias_row_size;
    logic [15:0] rd_bias_col_size;
    logic [15:0] rd_bias_time_counter;
    // Item 17a: ptr-7 RESIDUAL reads share the bias channel and its
    // skew/window, but walk the matrix elementwise (linear) instead of
    // the ptr-2 per-column broadcast. Set by a ptr-7 read command,
    // cleared by a ptr-2 read command.
    logic        rd_bias_residual;

    // Internal logic for Y inputs from UB to loss modules in VPU
    logic [15:0] rd_Y_ptr;
    logic [15:0] rd_Y_row_size;
    logic [15:0] rd_Y_col_size;
    logic [15:0] rd_Y_time_counter;

    // Internal logic for bias inputs from UB to activation derivative modules in VPU
    logic [15:0] rd_H_ptr;
    logic [15:0] rd_H_row_size;
    logic [15:0] rd_H_col_size;
    logic [15:0] rd_H_time_counter; 

    // Internal logic for bias gradient descent inputs from UB to gradient descent modules
    logic [15:0] rd_grad_bias_ptr;
    logic [15:0] rd_grad_bias_row_size;
    logic [15:0] rd_grad_bias_col_size;
    logic [15:0] rd_grad_bias_time_counter; 

    // Internal logic for weight gradient descent inputs from UB to gradient descent modules
    logic [15:0] rd_grad_weight_ptr;
    logic [15:0] rd_grad_weight_row_size;
    logic [15:0] rd_grad_weight_col_size;
    logic [15:0] rd_grad_weight_time_counter; 

    // Internal logic for gradient descent inputs from UB to gradient descent modules
    logic [15:0] value_old_in [SYSTOLIC_ARRAY_WIDTH];
    logic grad_descent_valid_in [SYSTOLIC_ARRAY_WIDTH];
    logic [15:0] value_updated_out [SYSTOLIC_ARRAY_WIDTH];
    logic grad_descent_done_out [SYSTOLIC_ARRAY_WIDTH];
    
    // Where to write gradients to UB
    logic [15:0] grad_descent_ptr;

    // Whether the gradients are biases or weights (0 for biases, 1 for weights)
    logic grad_bias_or_weight;

    // BUG-UB-1 fix: within-cycle tracking variables replacing blocking assignments in always block.
    // Each _next variable is set with blocking (=) to track intermediate address within one clock
    // edge, then written to the corresponding register with a single non-blocking (<=).
    logic [15:0]        wr_ptr_next;
    logic [15:0]        rd_input_ptr_next;
    logic signed [15:0] rd_weight_ptr_next;
    // Number of lanes that actually read from the shared weight pointer
    // during the current cycle (counted in the walk loops below).
    int                 rd_weight_lanes_read;
    logic [15:0]        rd_Y_ptr_next;
    logic [15:0]        rd_H_ptr_next;
    logic [15:0]        rd_grad_weight_ptr_next;
    // BUG-UB-3: combinational stream base (wr_ptr on the first beat of a
    // stream, the latched base afterwards).
    logic [15:0]        wr_stream_base_comb;
    // BUG-UB-4: effective stream width — the start-of-stream snapshot while
    // a stream is active, the live latch otherwise (the first beat has
    // beat_cnt == 0, so the width is only needed from the second beat on,
    // when the snapshot is already in place).
    logic [15:0]        wr_stream_w_eff;
    // iverilog 11 cannot reduce an unpacked array with |, so the any-lane
    // stream-valid flag is computed with an explicit loop.
    logic               wr_any_valid;

    always_comb begin
        wr_any_valid = 1'b0;
        for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
            wr_any_valid = wr_any_valid | ub_wr_valid_in[i];
        end
        wr_stream_w_eff = wr_stream_active_d ? wr_stream_width_snap
                                             : wr_stream_width;
    end

    // Generalized shared-pointer walk helpers (valid for ANY column count C = 1..N,
    // not just C=2; bit-identical to the legacy +1 walks at C=2 / R'=2).
    // Descending walks (input-untransposed, Y, H, grad_weight): lane i streams column i
    // top-row-first, so the within-cycle step between consecutive reading lanes is C-1.
    // End-of-cycle correction: C - (C-1)*di - (C-1), di = min(C-1, t+1) - max(0, t-R+1).
    function automatic int descending_walk_correction(input int C, input int R, input int t);
        int di;
        begin
            di = ((C-1 < t+1) ? C-1 : t+1) - ((t-R+1 > 0) ? t-R+1 : 0);
            descending_walk_correction = C - (C-1)*di - (C-1);
        end
    endfunction

    // Ascending walk (input-transposed): lane i streams row i first-element-first, so the
    // within-cycle step between consecutive reading lanes is R'-1 (latched row size).
    // End-of-cycle correction: 1 + (R'-1)*(i_min_next - i_max - 1) with
    // i_min_next = max(0, t+2-R'), i_max = min(C'-1, t).
    function automatic int ascending_walk_correction(input int Rp, input int Cp, input int t);
        int i_min_next;
        int i_max;
        begin
            i_min_next = (t+2-Rp > 0) ? t+2-Rp : 0;
            i_max = (Cp-1 < t) ? Cp-1 : t;
            ascending_walk_correction = 1 + (Rp-1)*(i_min_next - i_max - 1);
        end
    endfunction

    genvar i;
    generate
        for (i=0; i<SYSTOLIC_ARRAY_WIDTH; i++) begin : gradient_descent_gen
            gradient_descent gradient_descent_inst (
                .clk(clk),
                .rst(rst),
                .lr_in(learning_rate_in),
                .grad_in(ub_wr_data_in[i]),
                .value_old_in(value_old_in[i]),
                .grad_descent_valid_in(grad_descent_valid_in[i]),
                .grad_bias_or_weight(grad_bias_or_weight),
                .value_updated_out(value_updated_out[i]),
                .grad_descent_done_out(grad_descent_done_out[i]),
                .grad_overflow_out()  // BUG-OVF-1: overflow observable via hierarchical reference
            );
        end
    endgenerate

    // BUG-UB-3 fix: register ub_rd_col_size_valid_out to eliminate combinational glitch path.
    // Previously these were pure combinational assigns; glitches on ub_rd_start_in would
    // propagate directly into the systolic array's pe_enabled update.
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            ub_rd_col_size_valid_out <= 1'b0;
            ub_rd_col_size_out       <= '0;
        end else begin
            ub_rd_col_size_valid_out <= (ub_rd_start_in && (ub_ptr_select == 9'd1));
            ub_rd_col_size_out       <= (ub_rd_start_in && (ub_ptr_select == 9'd1)) ?
                                        (ub_rd_transpose ? ub_rd_row_size : ub_rd_col_size) : 16'b0;
        end
    end

    // BUG-UB-3: hold the active output width (the same value strobed out as
    // ub_rd_col_size_out) for row-major stream/writeback addressing.
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            wr_stream_width <= '0;
        end else if (ub_rd_start_in && (ub_ptr_select == 9'd1)) begin
            wr_stream_width <= ub_rd_transpose ? ub_rd_row_size : ub_rd_col_size;
        end
    end

    always_comb begin   // Automatically turn on gradient descent modules when bias or weight gradient descent pointers have been set by a read command
        if (
            rd_grad_bias_time_counter < rd_grad_bias_row_size + rd_grad_bias_col_size ||
            rd_grad_weight_time_counter < rd_grad_weight_row_size + rd_grad_weight_col_size
        ) begin
            for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                grad_descent_valid_in[i] = ub_wr_valid_in[i];
            end
        end else begin
            for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                grad_descent_valid_in[i] = 1'b0;
            end
        end
    end 

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            // reset all memory to 0
            for (int i = 0; i < UNIFIED_BUFFER_WIDTH; i++) begin
                ub_memory[i] <= '0;
            end

            // set internal registers to 0
            for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                ub_rd_input_data_out_r[i] <= '0;
                ub_rd_input_valid_out_r[i] <= '0;
                ub_rd_weight_data_out_r[i] <= '0;
                ub_rd_weight_valid_out_r[i] <= '0;
                ub_rd_bias_data_out_r[i] <= '0;
                ub_rd_Y_data_out_r[i] <= '0;
                ub_rd_H_data_out_r[i] <= '0;
                value_old_in[i] <= '0;
            end

            wr_ptr <= '0;

            rd_input_ptr <= '0;
            rd_input_row_size <= '0;
            rd_input_col_size <= '0;
            rd_input_time_counter <= '0;
            rd_input_transpose <= '0;

            rd_weight_ptr <= '0;
            rd_weight_row_size <= '0;
            rd_weight_col_size <= '0;
            rd_weight_time_counter <= '0;
            rd_weight_transpose <= '0;

            rd_bias_ptr <= '0;
            rd_bias_row_size <= '0;
            rd_bias_col_size <= '0;
            rd_bias_time_counter <= '0;
            rd_bias_residual <= '0;

            rd_Y_ptr <= '0;
            rd_Y_row_size <= '0;
            rd_Y_col_size <= '0;
            rd_Y_time_counter <= '0;

            rd_H_ptr <= '0;
            rd_H_row_size <= '0;
            rd_H_col_size <= '0;
            rd_H_time_counter <= '0;

            rd_grad_bias_ptr <= '0;
            rd_grad_bias_row_size <= '0;
            rd_grad_bias_col_size <= '0;
            rd_grad_bias_time_counter <= '0;

            rd_grad_weight_ptr <= '0;
            rd_grad_weight_row_size <= '0;
            rd_grad_weight_col_size <= '0;
            rd_grad_weight_time_counter <= '0;
            grad_bias_or_weight <= '0;
            grad_descent_ptr <= '0;

            wr_stream_base <= '0;
            wr_stream_width_snap <= '0;
            wr_stream_active_d <= '0;
            for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                wr_beat_cnt[i] <= '0;
                grad_done_cnt[i] <= '0;
            end
        end else begin
            // WRITING LOGIC
            // matrices are stored in row major format
            // if there are two columns, the first column will be stored at even indices and the second column will be stored at odd indices
            // BUG-UB-2 note: host loop decrements so channel[1] is at lower address than channel[0] (intentional row-major order)
            // BUG-UB-3: VPU output streams are also placed row-major — beat
            // (row r, col i) of a rows x w stream lands at
            // stream_base + r*w + i. Arrival-order placement only equaled
            // row-major at N=2; at N>=3 the skewed wavefront scrambled the
            // stored matrix against every row-major read walk.
            wr_stream_base_comb = wr_stream_active_d ? wr_stream_base : wr_ptr;
            wr_ptr_next = wr_ptr;  // BUG-UB-1 fix: use _next variable; single <= at end
            for (int i = SYSTOLIC_ARRAY_WIDTH-1; i >= 0; i--) begin     // FOR LOOP SHOULD DECREMENT TO STORE IN ROW MAJOR ORDER!!!
                if (ub_wr_valid_in[i]) begin
                    ub_memory[wr_stream_base_comb + wr_beat_cnt[i]*wr_stream_w_eff + i] <= ub_wr_data_in[i];
                end else if (ub_wr_host_valid_in[i]) begin
                    ub_memory[wr_ptr_next] <= ub_wr_host_data_in[i];
                    wr_ptr_next = wr_ptr_next + 1;
                end
            end
            if (ub_wr_valid_in[0]) begin
                // lane 0 beats every cycle from stream start until its rows
                // run out; keep wr_ptr one row past its latest beat so the
                // next stream/host region starts right after this one.
                wr_ptr <= wr_stream_base_comb + (wr_beat_cnt[0] + 1)*wr_stream_w_eff;
            end else if (!wr_any_valid) begin
                wr_ptr <= wr_ptr_next;
            end
            // (tail cycles: lane 0 done, lanes 1+ still draining — wr_ptr
            // already holds the final value)

            // stream bookkeeping: capture the base on the first beat of a
            // stream, count beats per lane while active, clear when idle.
            wr_stream_active_d <= wr_any_valid;
            if (wr_any_valid && !wr_stream_active_d) begin
                wr_stream_base <= wr_ptr;
                wr_stream_width_snap <= wr_stream_width;  // BUG-UB-4
            end
            if (wr_any_valid) begin
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    if (ub_wr_valid_in[i]) begin
                        wr_beat_cnt[i] <= wr_beat_cnt[i] + 1;
                    end
                end
            end else begin
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    wr_beat_cnt[i] <= '0;
                end
            end

            //WRITING LOGIC (for gradient descent modules to UB)
            if (grad_bias_or_weight) begin
                // BUG-UB-3: row-major writeback — done beat (r, i) of lane i
                // updates ub_memory[grad_descent_ptr + r*w + i]. The legacy
                // incrementing-pointer scheme wrote in skewed arrival order,
                // which only equaled row-major at N=2.
                for (int i = SYSTOLIC_ARRAY_WIDTH-1; i >= 0; i--) begin
                    if (grad_descent_done_out[i]) begin
                        ub_memory[grad_descent_ptr + grad_done_cnt[i]*wr_stream_w_eff + i] <= value_updated_out[i];
                        grad_done_cnt[i] <= grad_done_cnt[i] + 1;
                    end
                end
            end else begin
                for (int i = SYSTOLIC_ARRAY_WIDTH-1; i >= 0; i--) begin
                    if (grad_descent_done_out[i]) begin
                        ub_memory[grad_descent_ptr + i] <= value_updated_out[i];
                    end
                end
            end

            // READING LOGIC (for input from UB to left side of systolic array)
            if (rd_input_time_counter + 1 < rd_input_row_size + rd_input_col_size) begin
                rd_input_ptr_next = rd_input_ptr;  // BUG-UB-1 fix
                if(rd_input_transpose) begin
                    // For transposed matrices (for loop should increment)
                    for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                        if(rd_input_time_counter >= i && rd_input_time_counter < rd_input_row_size + i && i < rd_input_col_size) begin 
                            ub_rd_input_valid_out_r[i] <= 1'b1;
                            ub_rd_input_data_out_r[i] <= ub_memory[rd_input_ptr_next];
                            rd_input_ptr_next = rd_input_ptr_next + (rd_input_row_size - 1);
                        end else begin 
                            ub_rd_input_valid_out_r[i] <= 1'b0;
                            ub_rd_input_data_out_r[i] <= '0;
                        end
                    end
                    rd_input_ptr_next = rd_input_ptr_next + ascending_walk_correction(rd_input_row_size, rd_input_col_size, rd_input_time_counter);
                end else begin
                    // For untransposed matrices (for loop should decrement)
                    for (int i = SYSTOLIC_ARRAY_WIDTH-1; i >= 0; i--) begin
                        if(rd_input_time_counter >= i && rd_input_time_counter < rd_input_row_size + i && i < rd_input_col_size) begin 
                            ub_rd_input_valid_out_r[i] <= 1'b1;
                            ub_rd_input_data_out_r[i] <= ub_memory[rd_input_ptr_next];
                            rd_input_ptr_next = rd_input_ptr_next + (rd_input_col_size - 1);
                        end else begin 
                            ub_rd_input_valid_out_r[i] <= 1'b0;
                            ub_rd_input_data_out_r[i] <= '0;
                        end
                    end
                    rd_input_ptr_next = rd_input_ptr_next + descending_walk_correction(rd_input_col_size, rd_input_row_size, rd_input_time_counter);
                end
                rd_input_time_counter <= rd_input_time_counter + 1;
                rd_input_ptr <= rd_input_ptr_next;
            end else begin 
                rd_input_ptr <= 0;
                rd_input_row_size <= 0;
                rd_input_col_size <= 0;
                rd_input_time_counter <= '0;
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    ub_rd_input_valid_out_r[i] <= 1'b0;
                    ub_rd_input_data_out_r[i] <= '0;
                end
            end

            // READING LOGIC (for weights from UB to top of systolic array)
            if (rd_weight_time_counter + 1 < rd_weight_row_size + rd_weight_col_size) begin
                rd_weight_ptr_next = rd_weight_ptr;  // BUG-UB-1 fix
                rd_weight_lanes_read = 0;
                if(rd_weight_transpose) begin
                    // For transposed matrices (for loop should increment)
                    for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                        if(rd_weight_time_counter >= i && rd_weight_time_counter < rd_weight_row_size + i && i < rd_weight_col_size) begin
                            ub_rd_weight_valid_out_r[i] <= 1'b1;
                            ub_rd_weight_data_out_r[i] <= ub_memory[rd_weight_ptr_next];
                            rd_weight_ptr_next = rd_weight_ptr_next + rd_weight_skip_size;
                            rd_weight_lanes_read = rd_weight_lanes_read + 1;
                        end else begin
                            ub_rd_weight_valid_out_r[i] <= 0;
                            ub_rd_weight_data_out_r[i] <= '0;
                        end
                    end
                    // Generalized end-of-cycle correction (any column count C, not just C=2):
                    // undo all L lane skips from this cycle, step back 1 element, and once the
                    // leading lanes start retiring (t >= C-1) hop forward one row (C+1 = skip).
                    // C is the original command col size, i.e. skip-1; equals the legacy
                    // -(skip+1) constant at C=2 while any reads remain.
                    rd_weight_ptr_next = rd_weight_ptr_next - rd_weight_skip_size*rd_weight_lanes_read - 1
                                       + ((rd_weight_time_counter >= rd_weight_skip_size - 2) ? rd_weight_skip_size : 0);
                end else begin
                    // For untransposed matrices (for loop should decrement)
                    for (int i = SYSTOLIC_ARRAY_WIDTH-1; i >= 0; i--) begin
                        if(rd_weight_time_counter >= i && rd_weight_time_counter < rd_weight_row_size + i && i < rd_weight_col_size) begin
                            ub_rd_weight_valid_out_r[i] <= 1'b1;
                            ub_rd_weight_data_out_r[i] <= ub_memory[rd_weight_ptr_next];
                            rd_weight_ptr_next = rd_weight_ptr_next - rd_weight_skip_size;
                            rd_weight_lanes_read = rd_weight_lanes_read + 1;
                        end else begin
                            ub_rd_weight_valid_out_r[i] <= 0;
                            ub_rd_weight_data_out_r[i] <= '0;
                        end
                    end
                    // Generalized end-of-cycle correction (any column count C, not just C=2):
                    // undo all L lane skips from this cycle, step back one row (C = skip-1),
                    // and while the trailing lanes are still joining (t < C-1) hop forward one
                    // row plus one element (C+1 = skip). C is the original command col size;
                    // equals the legacy +(skip+1) constant at C=2 while any reads remain.
                    rd_weight_ptr_next = rd_weight_ptr_next + rd_weight_skip_size*rd_weight_lanes_read - (rd_weight_skip_size - 1)
                                       + ((rd_weight_time_counter < rd_weight_skip_size - 2) ? rd_weight_skip_size : 0);
                end
                rd_weight_time_counter <= rd_weight_time_counter + 1;
                rd_weight_ptr <= rd_weight_ptr_next;
            end else begin
                rd_weight_ptr <= 0;
                rd_weight_row_size <= 0;
                rd_weight_col_size <= 0;
                rd_weight_time_counter <= '0;
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    ub_rd_weight_valid_out_r[i] <= 0;
                    ub_rd_weight_data_out_r[i] <= '0;
                end
            end

            // READING LOGIC (for bias inputs from UB to bias modules in VPU)
            if (rd_bias_time_counter + 1 < rd_bias_row_size + rd_bias_col_size) begin
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    if(rd_bias_time_counter >= i && rd_bias_time_counter < rd_bias_row_size + i && i < rd_bias_col_size) begin
                        // ptr-2 bias: lane i's value (column i) held for
                        // every row. ptr-7 residual: elementwise linear
                        // walk of the row-major matrix at rd_bias_ptr --
                        // lane i's r-th active beat (r = time_counter - i)
                        // carries ub_memory[ptr + r*col_size + i]. Same
                        // per-lane skew and active window either way.
                        ub_rd_bias_data_out_r[i] <= rd_bias_residual
                            ? ub_memory[rd_bias_ptr + (rd_bias_time_counter - i)*rd_bias_col_size + i]
                            : ub_memory[rd_bias_ptr + i];
                    end else begin
                        ub_rd_bias_data_out_r[i] <= '0;
                    end
                end
                rd_bias_time_counter <= rd_bias_time_counter + 1;
            end else begin
                rd_bias_ptr <= 0;
                rd_bias_row_size <= 0;
                rd_bias_col_size <= 0;
                rd_bias_time_counter <= '0;
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    ub_rd_bias_data_out_r[i] <= '0;
                end
            end

            // READING LOGIC (for Y inputs from UB to loss modules in VPU)
            if (rd_Y_time_counter + 1 < rd_Y_row_size + rd_Y_col_size) begin
                rd_Y_ptr_next = rd_Y_ptr;  // BUG-UB-1 fix
                for (int i = SYSTOLIC_ARRAY_WIDTH-1; i >= 0; i--) begin
                    if(rd_Y_time_counter >= i && rd_Y_time_counter < rd_Y_row_size + i && i < rd_Y_col_size) begin
                        ub_rd_Y_data_out_r[i] <= ub_memory[rd_Y_ptr_next];
                        rd_Y_ptr_next = rd_Y_ptr_next + (rd_Y_col_size - 1);
                    end else begin
                        ub_rd_Y_data_out_r[i] <= '0;
                    end
                end
                rd_Y_ptr_next = rd_Y_ptr_next + descending_walk_correction(rd_Y_col_size, rd_Y_row_size, rd_Y_time_counter);
                rd_Y_time_counter <= rd_Y_time_counter + 1;
                rd_Y_ptr <= rd_Y_ptr_next;
            end else begin
                rd_Y_ptr <= 0;
                rd_Y_row_size <= 0;
                rd_Y_col_size <= 0;
                rd_Y_time_counter <= '0;
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    ub_rd_Y_data_out_r[i] <= '0;
                end
            end

            // READING LOGIC (for H inputs from UB to activation derivative modules in VPU)
            if (rd_H_time_counter + 1 < rd_H_row_size + rd_H_col_size) begin
                rd_H_ptr_next = rd_H_ptr;  // BUG-UB-1 fix
                for (int i = SYSTOLIC_ARRAY_WIDTH-1; i >= 0; i--) begin
                    if(rd_H_time_counter >= i && rd_H_time_counter < rd_H_row_size + i && i < rd_H_col_size) begin
                        ub_rd_H_data_out_r[i] <= ub_memory[rd_H_ptr_next];
                        rd_H_ptr_next = rd_H_ptr_next + (rd_H_col_size - 1);
                    end else begin
                        ub_rd_H_data_out_r[i] <= '0;
                    end
                end
                rd_H_ptr_next = rd_H_ptr_next + descending_walk_correction(rd_H_col_size, rd_H_row_size, rd_H_time_counter);
                rd_H_time_counter <= rd_H_time_counter + 1;
                rd_H_ptr <= rd_H_ptr_next;
            end else begin
                rd_H_ptr <= 0;
                rd_H_row_size <= 0;
                rd_H_col_size <= 0;
                rd_H_time_counter <= '0;
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    ub_rd_H_data_out_r[i] <= '0;
                end
            end

            // READING LOGIC (for bias and weight gradient descent inputs from UB to gradient descent modules)
            if (rd_grad_bias_time_counter + 1 < rd_grad_bias_row_size + rd_grad_bias_col_size) begin
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    if(rd_grad_bias_time_counter >= i && rd_grad_bias_time_counter < rd_grad_bias_row_size + i && i < rd_grad_bias_col_size) begin
                        value_old_in[i] <= ub_memory[rd_grad_bias_ptr + i];
                    end else begin
                        value_old_in[i] <= '0;
                    end
                end
                rd_grad_bias_time_counter <= rd_grad_bias_time_counter + 1;
            end else if (rd_grad_weight_time_counter + 1 < rd_grad_weight_row_size + rd_grad_weight_col_size) begin
                rd_grad_weight_ptr_next = rd_grad_weight_ptr;  // BUG-UB-1 fix
                for (int i = SYSTOLIC_ARRAY_WIDTH-1; i >= 0; i--) begin
                    if(rd_grad_weight_time_counter >= i && rd_grad_weight_time_counter < rd_grad_weight_row_size + i && i < rd_grad_weight_col_size) begin 
                        value_old_in[i] <= ub_memory[rd_grad_weight_ptr_next];
                        rd_grad_weight_ptr_next = rd_grad_weight_ptr_next + (rd_grad_weight_col_size - 1);
                    end else begin 
                        value_old_in[i] <= '0;
                    end
                end
                rd_grad_weight_ptr_next = rd_grad_weight_ptr_next + descending_walk_correction(rd_grad_weight_col_size, rd_grad_weight_row_size, rd_grad_weight_time_counter);
                rd_grad_weight_time_counter <= rd_grad_weight_time_counter + 1;
                rd_grad_weight_ptr <= rd_grad_weight_ptr_next;
            end else begin
                rd_grad_bias_ptr <= 0;
                rd_grad_bias_row_size <= 0;
                rd_grad_bias_col_size <= 0;
                rd_grad_bias_time_counter <= '0;
                rd_grad_weight_ptr <= 0;
                rd_grad_weight_row_size <= 0;
                rd_grad_weight_col_size <= 0;
                rd_grad_weight_time_counter <= '0;
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    value_old_in[i] <= '0;
                end
            end

            // Initialize read channels when ub_rd_start_in is asserted.
            // Placed last so these NBAs override any same-cycle reading-logic NBAs.
            if (ub_rd_start_in) begin
                case (ub_ptr_select)
                    0: begin
                        rd_input_transpose <= ub_rd_transpose;
                        rd_input_ptr <= ub_rd_addr_in;
                        if(ub_rd_transpose) begin
                            rd_input_row_size <= ub_rd_col_size;
                            rd_input_col_size <= ub_rd_row_size;
                        end else begin
                            rd_input_row_size <= ub_rd_row_size;
                            rd_input_col_size <= ub_rd_col_size;
                        end
                        rd_input_time_counter <= '0;
                    end
                    1: begin
                        rd_weight_transpose <= ub_rd_transpose;
                        if(ub_rd_transpose) begin
                            rd_weight_row_size <= ub_rd_col_size;
                            rd_weight_col_size <= ub_rd_row_size;
                            rd_weight_ptr <= ub_rd_addr_in + ub_rd_col_size - 1;
                        end else begin
                            rd_weight_row_size <= ub_rd_row_size;
                            rd_weight_col_size <= ub_rd_col_size;
                            rd_weight_ptr <= ub_rd_addr_in + ub_rd_row_size*ub_rd_col_size - ub_rd_col_size;
                        end
                        rd_weight_skip_size <= ub_rd_col_size + 1;
                        rd_weight_time_counter <= '0;
                    end
                    2: begin
                        rd_bias_ptr <= ub_rd_addr_in;
                        rd_bias_row_size <= ub_rd_row_size;
                        rd_bias_col_size <= ub_rd_col_size;
                        rd_bias_time_counter <= '0;
                        rd_bias_residual <= 1'b0;
                    end
                    7: begin
                        // Item 17a RESIDUAL read: arms the SAME bias
                        // operand stream (same skew, same active window)
                        // but with the elementwise linear walk. Issued
                        // mid-phase like the legacy bias read, it arms
                        // only that phase's output stream: the bias stage
                        // (pathway bit 3) then computes C = A@W + R.
                        rd_bias_ptr <= ub_rd_addr_in;
                        rd_bias_row_size <= ub_rd_row_size;
                        rd_bias_col_size <= ub_rd_col_size;
                        rd_bias_time_counter <= '0;
                        rd_bias_residual <= 1'b1;
                    end
                    3: begin
                        rd_Y_ptr <= ub_rd_addr_in;
                        rd_Y_row_size <= ub_rd_row_size;
                        rd_Y_col_size <= ub_rd_col_size;
                        rd_Y_time_counter <= '0;
                    end
                    4: begin
                        rd_H_ptr <= ub_rd_addr_in;
                        rd_H_row_size <= ub_rd_row_size;
                        rd_H_col_size <= ub_rd_col_size;
                        rd_H_time_counter <= '0;
                    end
                    5: begin
                        rd_grad_bias_ptr <= ub_rd_addr_in;
                        rd_grad_bias_row_size <= ub_rd_row_size;
                        rd_grad_bias_col_size <= ub_rd_col_size;
                        rd_grad_bias_time_counter <= '0;
                        grad_bias_or_weight <= 1'b0;
                        grad_descent_ptr <= ub_rd_addr_in;
                        for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                            grad_done_cnt[i] <= '0;
                        end
                    end
                    6: begin
                        rd_grad_weight_ptr <= ub_rd_addr_in;
                        rd_grad_weight_row_size <= ub_rd_row_size;
                        rd_grad_weight_col_size <= ub_rd_col_size;
                        rd_grad_weight_time_counter <= '0;
                        grad_bias_or_weight <= 1'b1;
                        grad_descent_ptr <= ub_rd_addr_in;
                        for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                            grad_done_cnt[i] <= '0;
                        end
                    end
                endcase
            end
        end
    end
endmodule