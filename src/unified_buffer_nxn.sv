`timescale 1ns/1ps
`default_nettype none

// Unified buffer with SRAM-macro storage (see docs/SRAM_INTEGRATION.md).
//
// The DFF array was replaced by ONE behavioral multi-port SRAM macro:
//   - 2N write ports: [0,N) gradient-descent writeback (highest priority,
//     matching the DFF version's last-NBA-wins order), [N,2N) per-lane
//     merged VPU-stream/host writes (mutually exclusive per lane there).
//   - 6N read ports: N-lane groups for input (ptr 0), weight (ptr 1),
//     bias/residual/scale (ptr 2/7/8), Y (ptr 3), H (ptr 4), and the
//     mutually-exclusive gradient pair (ptr 5/6) sharing one group.
//
// Timing is IDENTICAL to the DFF version: the SRAM's synchronous read
// (1-cycle latency) takes the place of the old per-lane output data
// registers. Every per-channel address walk is computed combinationally
// from the current state (the same arithmetic the DFF version performed
// inside always_ff) and presented to the SRAM this cycle; the registered
// SRAM output is consumed next cycle alongside the valid/window registers,
// which are updated on the same conditions as before. Writes were and stay
// synchronous. Read-during-write to the same address returns the OLD data
// in both versions (pre-edge sample), so no consumer sees a difference.
//
// Reset: CLEAR_ON_RESET=1 zeroes the array on rst (DFF parity, behavioral
// model only — a foundry macro swap needs a boot-time scrub instead).

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

    // Item 18a SCALE read (ptr 8): per-lane operand-window valid for
    // the VPU's head-of-chain scale stage (active only while
    // rd_bias_scale is set), plus the scalar arm flag itself.
    output wire ub_rd_scale_valid_out [SYSTOLIC_ARRAY_WIDTH],
    output wire ub_rd_scale_arm_out,

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
    logic ub_rd_scale_valid_out_r [SYSTOLIC_ARRAY_WIDTH];  // BUG-TOOLS-1 mirror
    logic [15:0] ub_rd_Y_data_out_r [SYSTOLIC_ARRAY_WIDTH];  // BUG-TOOLS-1 mirror
    logic [15:0] ub_rd_H_data_out_r [SYSTOLIC_ARRAY_WIDTH];  // BUG-TOOLS-1 mirror

    assign ub_rd_input_data_out = ub_rd_input_data_out_r;
    assign ub_rd_input_valid_out = ub_rd_input_valid_out_r;
    assign ub_rd_weight_data_out = ub_rd_weight_data_out_r;
    assign ub_rd_weight_valid_out = ub_rd_weight_valid_out_r;
    assign ub_rd_bias_data_out = ub_rd_bias_data_out_r;
    assign ub_rd_scale_valid_out = ub_rd_scale_valid_out_r;
    assign ub_rd_Y_data_out = ub_rd_Y_data_out_r;
    assign ub_rd_H_data_out = ub_rd_H_data_out_r;

    // ------------------------------------------------------------------
    // SRAM storage (replaces the DFF array). Read-port group bases:
    //   GRP_IN  ptr 0: systolic input (left)
    //   GRP_W   ptr 1: systolic weight (top)
    //   GRP_B   ptr 2/7/8: bias / residual / scale (shared channel)
    //   GRP_Y   ptr 3: loss Y
    //   GRP_H   ptr 4: activation derivative H
    //   GRP_G   ptr 5/6: gradient bias / weight (mutually exclusive)
    // ------------------------------------------------------------------
    localparam int GRP_IN = 0;
    localparam int GRP_W  = 1*SYSTOLIC_ARRAY_WIDTH;
    localparam int GRP_B  = 2*SYSTOLIC_ARRAY_WIDTH;
    localparam int GRP_Y  = 3*SYSTOLIC_ARRAY_WIDTH;
    localparam int GRP_H  = 4*SYSTOLIC_ARRAY_WIDTH;
    localparam int GRP_G  = 5*SYSTOLIC_ARRAY_WIDTH;

    logic [2*SYSTOLIC_ARRAY_WIDTH-1:0]    sr_wr_en;
    logic [2*SYSTOLIC_ARRAY_WIDTH*16-1:0] sr_wr_addr;
    logic [2*SYSTOLIC_ARRAY_WIDTH*16-1:0] sr_wr_data;
    logic [6*SYSTOLIC_ARRAY_WIDTH*16-1:0] sr_rd_addr;
    logic [6*SYSTOLIC_ARRAY_WIDTH*16-1:0] sr_rd_data;

    // Packed per-lane read-window mask (bit G*N+i = lane i of group G):
    // drives the banked UB's rd_act port (contract assertions) and the
    // UB_TRACE diagnostic. Wires only; pruned where unused.
    logic [6*SYSTOLIC_ARRAY_WIDTH-1:0] ub_win_packed;

`ifdef SYNTH_UB_BANKED
    // Item 25: banked 1rw1r storage — routable by construction
    // (docs/UB_REARCHITECTURE.md). Same ports/timing as sram_macro.
    ub_banked #(
        .WIDTH(16),
        .DEPTH(UNIFIED_BUFFER_WIDTH),
        .N(SYSTOLIC_ARRAY_WIDTH)
    ) ub_sram (
        .clk(clk),
        .rst(rst),
        .wr_en(sr_wr_en),
        .wr_addr(sr_wr_addr),
        .wr_data(sr_wr_data),
        .rd_addr(sr_rd_addr),
        .rd_data(sr_rd_data),
        .rd_act(ub_win_packed)
    );
`else
    sram_macro #(
        .WIDTH(16),
        .DEPTH(UNIFIED_BUFFER_WIDTH),
        .NUM_WRITE(2*SYSTOLIC_ARRAY_WIDTH),
        .NUM_READ(6*SYSTOLIC_ARRAY_WIDTH),
        .CLEAR_ON_RESET(1)
    ) ub_sram (
        .clk(clk),
        .rst(rst),
        .wr_en(sr_wr_en),
        .wr_addr(sr_wr_addr),
        .wr_data(sr_wr_data),
        .rd_addr(sr_rd_addr),
        .rd_data(sr_rd_data)
    );
`endif

    // Test observability: the full-chip gate suites read the buffer contents
    // hierarchically via VPI as `ub_inst.ub_sram.mem[a]` (pull-based reads,
    // zero sim cost). Do NOT reintroduce a continuous-assign `ub_memory`
    // alias here: an `assign alias[i] = ub_sram.mem[i]` generate loop makes
    // every mem write wake all DEPTH alias functors (vvp_fun_arrayport_sa),
    // which is O(DEPTH^2) per CLEAR_ON_RESET cycle — measured 172x slowdown
    // at N=16 (0.24s -> 41s for 60 cycles), multi-hour at real test length.

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
    // Item 18a: ptr-8 SCALE reads share the bias channel, its skew and
    // the ptr-7 elementwise linear walk. Set by a ptr-8 read command,
    // cleared by a ptr-2 (bias) or ptr-7 (residual) read command --
    // each of ptr 2/7/8 re-arms its own exclusive operand mode.
    logic        rd_bias_scale;
    // Assigned here (not with the block above): slang requires
    // declaration-before-use; iverilog tolerates the forward reference.
    assign ub_rd_scale_arm_out = rd_bias_scale;

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

    // Combinational next-address / next-state signals. The DFF version
    // computed these with blocking assignments inside always_ff (BUG-UB-1
    // _next discipline); with the SRAM they are generated one cycle ahead
    // in always_comb and registered by the SRAM read port itself.
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

    // Per-lane active-window flags (combinational). The same flags gate the
    // SRAM address lanes this cycle and are registered into the valid/window
    // bits that qualify the returning data next cycle.
    logic in_win  [SYSTOLIC_ARRAY_WIDTH];
    logic wt_win  [SYSTOLIC_ARRAY_WIDTH];
    logic b_win   [SYSTOLIC_ARRAY_WIDTH];
    logic y_win   [SYSTOLIC_ARRAY_WIDTH];
    logic h_win   [SYSTOLIC_ARRAY_WIDTH];
    logic g_win   [SYSTOLIC_ARRAY_WIDTH];

    // Registered operand windows for the channels without a valid output
    // (bias/Y/H feed VPU stages directly; grad feeds value_old_in). They
    // zero-mask the returning SRAM data exactly where the DFF version
    // registered '0 into the data output.
    logic bias_win_r [SYSTOLIC_ARRAY_WIDTH];
    logic y_win_r    [SYSTOLIC_ARRAY_WIDTH];
    logic h_win_r    [SYSTOLIC_ARRAY_WIDTH];
    logic g_win_r    [SYSTOLIC_ARRAY_WIDTH];

    // Channel-active conditions (identical to the DFF version's per-section
    // guards; gw_cond is reached only when gb_active is false, matching the
    // original if/else-if chain).
    wire in_active = (rd_input_time_counter + 1 < rd_input_row_size + rd_input_col_size);
    wire wt_active = (rd_weight_time_counter + 1 < rd_weight_row_size + rd_weight_col_size);
    wire b_active  = (rd_bias_time_counter + 1 < rd_bias_row_size + rd_bias_col_size);
    wire y_active  = (rd_Y_time_counter + 1 < rd_Y_row_size + rd_Y_col_size);
    wire h_active  = (rd_H_time_counter + 1 < rd_H_row_size + rd_H_col_size);
    wire gb_active = (rd_grad_bias_time_counter + 1 < rd_grad_bias_row_size + rd_grad_bias_col_size);
    wire gw_cond   = (rd_grad_weight_time_counter + 1 < rd_grad_weight_row_size + rd_grad_weight_col_size);

    always_comb begin
        wr_any_valid = 1'b0;
        for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
            wr_any_valid = wr_any_valid | ub_wr_valid_in[i];
        end
        wr_stream_w_eff = wr_stream_active_d ? wr_stream_width_snap
                                             : wr_stream_width;
        wr_stream_base_comb = wr_stream_active_d ? wr_stream_base : wr_ptr;
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

    // ------------------------------------------------------------------
    // Combinational SRAM port assembly: per-lane read addresses (the walk
    // arithmetic below is verbatim what the DFF version ran inside
    // always_ff), per-lane write ports, and the end-of-cycle pointer
    // corrections. Everything is defaulted first (no latches).
    // ------------------------------------------------------------------
    always_comb begin
        // Defaults
        sr_wr_en   = '0;
        sr_wr_addr = '0;
        sr_wr_data = '0;
        sr_rd_addr = '0;
        wr_ptr_next = wr_ptr;
        rd_input_ptr_next = rd_input_ptr;
        rd_weight_ptr_next = rd_weight_ptr;
        rd_weight_lanes_read = 0;
        rd_Y_ptr_next = rd_Y_ptr;
        rd_H_ptr_next = rd_H_ptr;
        rd_grad_weight_ptr_next = rd_grad_weight_ptr;
        for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
            in_win[i] = 1'b0;
            wt_win[i] = 1'b0;
            b_win[i]  = 1'b0;
            y_win[i]  = 1'b0;
            h_win[i]  = 1'b0;
            g_win[i]  = 1'b0;
        end

        // ---- WRITE PORTS ----
        // Ports [0,N): gradient-descent writeback. Highest SRAM priority,
        // matching the DFF version where these NBAs came last in the
        // always_ff block (last write wins).
        for (int i = SYSTOLIC_ARRAY_WIDTH-1; i >= 0; i--) begin
            sr_wr_en[i] = grad_descent_done_out[i];
            if (grad_bias_or_weight) begin
                // BUG-UB-3: row-major writeback — done beat (r, i) of lane i
                // updates ub_memory[grad_descent_ptr + r*w + i].
                sr_wr_addr[i*16 +: 16] = grad_descent_ptr + grad_done_cnt[i]*wr_stream_w_eff + i;
            end else begin
                sr_wr_addr[i*16 +: 16] = grad_descent_ptr + i;
            end
            sr_wr_data[i*16 +: 16] = value_updated_out[i];
        end

        // Ports [N,2N): merged VPU-stream / host writes (the DFF version's
        // per-lane if/else-if makes them mutually exclusive). Host walk
        // decrements so the highest-index host-valid lane lands at the
        // lowest address (row-major host order, BUG-UB-2 note).
        for (int i = SYSTOLIC_ARRAY_WIDTH-1; i >= 0; i--) begin
            sr_wr_en[SYSTOLIC_ARRAY_WIDTH + i] = ub_wr_valid_in[i] | ub_wr_host_valid_in[i];
            if (ub_wr_valid_in[i]) begin
                // BUG-UB-3: VPU output streams are placed row-major — beat
                // (row r, col i) lands at stream_base + r*w + i.
                sr_wr_addr[(SYSTOLIC_ARRAY_WIDTH + i)*16 +: 16] =
                    wr_stream_base_comb + wr_beat_cnt[i]*wr_stream_w_eff + i;
                sr_wr_data[(SYSTOLIC_ARRAY_WIDTH + i)*16 +: 16] = ub_wr_data_in[i];
            end else begin
                sr_wr_addr[(SYSTOLIC_ARRAY_WIDTH + i)*16 +: 16] = wr_ptr_next;
                sr_wr_data[(SYSTOLIC_ARRAY_WIDTH + i)*16 +: 16] = ub_wr_host_data_in[i];
                if (ub_wr_host_valid_in[i]) begin
                    wr_ptr_next = wr_ptr_next + 1;
                end
            end
        end

        // ---- READ PORTS ----
        // GRP_IN: input to the left of the systolic array (ptr 0)
        if (in_active) begin
            if (rd_input_transpose) begin
                // For transposed matrices (for loop should increment)
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    if (rd_input_time_counter >= i && rd_input_time_counter < rd_input_row_size + i && i < rd_input_col_size) begin
                        in_win[i] = 1'b1;
                        sr_rd_addr[(GRP_IN + i)*16 +: 16] = rd_input_ptr_next;
                        rd_input_ptr_next = rd_input_ptr_next + (rd_input_row_size - 1);
                    end
                end
                rd_input_ptr_next = rd_input_ptr_next + ascending_walk_correction(rd_input_row_size, rd_input_col_size, rd_input_time_counter);
            end else begin
                // For untransposed matrices (for loop should decrement)
                for (int i = SYSTOLIC_ARRAY_WIDTH-1; i >= 0; i--) begin
                    if (rd_input_time_counter >= i && rd_input_time_counter < rd_input_row_size + i && i < rd_input_col_size) begin
                        in_win[i] = 1'b1;
                        sr_rd_addr[(GRP_IN + i)*16 +: 16] = rd_input_ptr_next;
                        rd_input_ptr_next = rd_input_ptr_next + (rd_input_col_size - 1);
                    end
                end
                rd_input_ptr_next = rd_input_ptr_next + descending_walk_correction(rd_input_col_size, rd_input_row_size, rd_input_time_counter);
            end
        end

        // GRP_W: weights to the top of the systolic array (ptr 1)
        if (wt_active) begin
            if (rd_weight_transpose) begin
                // For transposed matrices (for loop should increment)
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    if (rd_weight_time_counter >= i && rd_weight_time_counter < rd_weight_row_size + i && i < rd_weight_col_size) begin
                        wt_win[i] = 1'b1;
                        sr_rd_addr[(GRP_W + i)*16 +: 16] = rd_weight_ptr_next;
                        rd_weight_ptr_next = rd_weight_ptr_next + rd_weight_skip_size;
                        rd_weight_lanes_read = rd_weight_lanes_read + 1;
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
                    if (rd_weight_time_counter >= i && rd_weight_time_counter < rd_weight_row_size + i && i < rd_weight_col_size) begin
                        wt_win[i] = 1'b1;
                        sr_rd_addr[(GRP_W + i)*16 +: 16] = rd_weight_ptr_next;
                        rd_weight_ptr_next = rd_weight_ptr_next - rd_weight_skip_size;
                        rd_weight_lanes_read = rd_weight_lanes_read + 1;
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
        end

        // GRP_B: bias (ptr 2) / residual (ptr 7) / scale (ptr 8) shared channel
        if (b_active) begin
            for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                if (rd_bias_time_counter >= i && rd_bias_time_counter < rd_bias_row_size + i && i < rd_bias_col_size) begin
                    b_win[i] = 1'b1;
                    // ptr-2 bias: lane i's value (column i) held for every
                    // row. ptr-7 residual / ptr-8 scale: elementwise linear
                    // walk of the row-major matrix at rd_bias_ptr -- lane i's
                    // r-th active beat (r = time_counter - i) carries
                    // ub_memory[ptr + r*col_size + i]. Same per-lane skew and
                    // active window either way.
                    sr_rd_addr[(GRP_B + i)*16 +: 16] = (rd_bias_residual || rd_bias_scale)
                        ? rd_bias_ptr + (rd_bias_time_counter - i)*rd_bias_col_size + i
                        : rd_bias_ptr + i;
                end
            end
        end

        // GRP_Y: loss Y inputs (ptr 3)
        if (y_active) begin
            for (int i = SYSTOLIC_ARRAY_WIDTH-1; i >= 0; i--) begin
                if (rd_Y_time_counter >= i && rd_Y_time_counter < rd_Y_row_size + i && i < rd_Y_col_size) begin
                    y_win[i] = 1'b1;
                    sr_rd_addr[(GRP_Y + i)*16 +: 16] = rd_Y_ptr_next;
                    rd_Y_ptr_next = rd_Y_ptr_next + (rd_Y_col_size - 1);
                end
            end
            rd_Y_ptr_next = rd_Y_ptr_next + descending_walk_correction(rd_Y_col_size, rd_Y_row_size, rd_Y_time_counter);
        end

        // GRP_H: activation derivative H inputs (ptr 4)
        if (h_active) begin
            for (int i = SYSTOLIC_ARRAY_WIDTH-1; i >= 0; i--) begin
                if (rd_H_time_counter >= i && rd_H_time_counter < rd_H_row_size + i && i < rd_H_col_size) begin
                    h_win[i] = 1'b1;
                    sr_rd_addr[(GRP_H + i)*16 +: 16] = rd_H_ptr_next;
                    rd_H_ptr_next = rd_H_ptr_next + (rd_H_col_size - 1);
                end
            end
            rd_H_ptr_next = rd_H_ptr_next + descending_walk_correction(rd_H_col_size, rd_H_row_size, rd_H_time_counter);
        end

        // GRP_G: gradient-descent old-value reads (ptr 5 bias / ptr 6
        // weight). The two channels are mutually exclusive (the DFF
        // version's if/else-if), so they share one lane group.
        if (gb_active) begin
            for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                if (rd_grad_bias_time_counter >= i && rd_grad_bias_time_counter < rd_grad_bias_row_size + i && i < rd_grad_bias_col_size) begin
                    g_win[i] = 1'b1;
                    sr_rd_addr[(GRP_G + i)*16 +: 16] = rd_grad_bias_ptr + i;
                end
            end
        end else if (gw_cond) begin
            for (int i = SYSTOLIC_ARRAY_WIDTH-1; i >= 0; i--) begin
                if (rd_grad_weight_time_counter >= i && rd_grad_weight_time_counter < rd_grad_weight_row_size + i && i < rd_grad_weight_col_size) begin
                    g_win[i] = 1'b1;
                    sr_rd_addr[(GRP_G + i)*16 +: 16] = rd_grad_weight_ptr_next;
                    rd_grad_weight_ptr_next = rd_grad_weight_ptr_next + (rd_grad_weight_col_size - 1);
                end
            end
            rd_grad_weight_ptr_next = rd_grad_weight_ptr_next + descending_walk_correction(rd_grad_weight_col_size, rd_grad_weight_row_size, rd_grad_weight_time_counter);
        end
    end

    // ------------------------------------------------------------------
    // Output data path: the SRAM's registered output is the data word for
    // the address presented LAST cycle; the registered valid/window bits
    // (same conditions, registered on the same edge) qualify it. Inactive
    // lanes read as zero, exactly as the DFF version's '0 data NBAs.
    // ------------------------------------------------------------------
    always_comb begin
        for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
            ub_rd_input_data_out_r[i]  = ub_rd_input_valid_out_r[i]  ? sr_rd_data[(GRP_IN + i)*16 +: 16] : 16'b0;
            ub_rd_weight_data_out_r[i] = ub_rd_weight_valid_out_r[i] ? sr_rd_data[(GRP_W + i)*16 +: 16] : 16'b0;
            ub_rd_bias_data_out_r[i]   = bias_win_r[i] ? sr_rd_data[(GRP_B + i)*16 +: 16] : 16'b0;
            ub_rd_Y_data_out_r[i]      = y_win_r[i]    ? sr_rd_data[(GRP_Y + i)*16 +: 16] : 16'b0;
            ub_rd_H_data_out_r[i]      = h_win_r[i]    ? sr_rd_data[(GRP_H + i)*16 +: 16] : 16'b0;
            value_old_in[i]            = g_win_r[i]    ? sr_rd_data[(GRP_G + i)*16 +: 16] : 16'b0;
        end
    end

    // Packed window mask assembly (drives ub_banked.rd_act + UB_TRACE).
    always_comb begin
        for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
            ub_win_packed[0*SYSTOLIC_ARRAY_WIDTH + i] = in_win[i];
            ub_win_packed[1*SYSTOLIC_ARRAY_WIDTH + i] = wt_win[i];
            ub_win_packed[2*SYSTOLIC_ARRAY_WIDTH + i] = b_win[i];
            ub_win_packed[3*SYSTOLIC_ARRAY_WIDTH + i] = y_win[i];
            ub_win_packed[4*SYSTOLIC_ARRAY_WIDTH + i] = h_win[i];
            ub_win_packed[5*SYSTOLIC_ARRAY_WIDTH + i] = g_win[i];
        end
    end

`ifdef UB_TRACE
    // Item 25 diagnostic D1: per-cycle dump of the full port-utilization
    // state — per-lane read windows, all read addresses, write enables and
    // write addresses. Offline analysis (diag/ub_concurrency.py) derives
    // group-concurrency maxima and bank-conflict counts for candidate
    // swizzles. Compile-time only; the gate suites never define UB_TRACE.
    integer ub_trace_fd;
    initial ub_trace_fd = $fopen("ub_trace.log", "w");
    always @(posedge clk) begin
        if (!rst && (|ub_win_packed || |sr_wr_en)) begin
            $fdisplay(ub_trace_fd, "%h %h %h %h",
                      ub_win_packed, sr_rd_addr, sr_wr_en, sr_wr_addr);
        end
    end
`endif

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            // SRAM array itself is cleared by CLEAR_ON_RESET (behavioral
            // model; see the sram_macro header note).

            // set internal registers to 0
            for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                ub_rd_input_valid_out_r[i] <= '0;
                ub_rd_weight_valid_out_r[i] <= '0;
                ub_rd_scale_valid_out_r[i] <= '0;
                bias_win_r[i] <= '0;
                y_win_r[i] <= '0;
                h_win_r[i] <= '0;
                g_win_r[i] <= '0;
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
            rd_bias_scale <= '0;

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
            // WRITING LOGIC (pointer/bookkeeping only — the SRAM write ports
            // are assembled combinationally above). matrices are stored in
            // row major format.
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

            // WRITING LOGIC (gradient descent beat counters; the writeback
            // itself is a combinational SRAM write port above)
            if (grad_bias_or_weight) begin
                for (int i = SYSTOLIC_ARRAY_WIDTH-1; i >= 0; i--) begin
                    if (grad_descent_done_out[i]) begin
                        grad_done_cnt[i] <= grad_done_cnt[i] + 1;
                    end
                end
            end

            // READING LOGIC (input from UB to left side of systolic array):
            // state advance + valid registers; addresses are the comb walk.
            if (in_active) begin
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    ub_rd_input_valid_out_r[i] <= in_win[i];
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
                end
            end

            // READING LOGIC (weights from UB to top of systolic array)
            if (wt_active) begin
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    ub_rd_weight_valid_out_r[i] <= wt_win[i];
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
                end
            end

            // READING LOGIC (bias/residual/scale from UB to VPU stages)
            if (b_active) begin
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    bias_win_r[i] <= b_win[i];
                    // Item 18a: the scale stage's per-lane operand window
                    // rides the same skewed schedule; it is asserted only
                    // while a ptr-8 read armed the scale mode (a stale-armed
                    // phase with no operand read keeps every beat clear).
                    ub_rd_scale_valid_out_r[i] <= b_win[i] && rd_bias_scale;
                end
                rd_bias_time_counter <= rd_bias_time_counter + 1;
            end else begin
                rd_bias_ptr <= 0;
                rd_bias_row_size <= 0;
                rd_bias_col_size <= 0;
                rd_bias_time_counter <= '0;
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    bias_win_r[i] <= '0;
                    ub_rd_scale_valid_out_r[i] <= 1'b0;
                end
            end

            // READING LOGIC (Y inputs from UB to loss modules in VPU)
            if (y_active) begin
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    y_win_r[i] <= y_win[i];
                end
                rd_Y_time_counter <= rd_Y_time_counter + 1;
                rd_Y_ptr <= rd_Y_ptr_next;
            end else begin
                rd_Y_ptr <= 0;
                rd_Y_row_size <= 0;
                rd_Y_col_size <= 0;
                rd_Y_time_counter <= '0;
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    y_win_r[i] <= '0;
                end
            end

            // READING LOGIC (H inputs from UB to activation derivative modules in VPU)
            if (h_active) begin
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    h_win_r[i] <= h_win[i];
                end
                rd_H_time_counter <= rd_H_time_counter + 1;
                rd_H_ptr <= rd_H_ptr_next;
            end else begin
                rd_H_ptr <= 0;
                rd_H_row_size <= 0;
                rd_H_col_size <= 0;
                rd_H_time_counter <= '0;
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    h_win_r[i] <= '0;
                end
            end

            // READING LOGIC (bias and weight gradient descent inputs from UB
            // to gradient descent modules): window registers only; data comes
            // back on the shared SRAM read group.
            if (gb_active) begin
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    g_win_r[i] <= g_win[i];
                end
                rd_grad_bias_time_counter <= rd_grad_bias_time_counter + 1;
            end else if (gw_cond) begin
                for (int i = 0; i < SYSTOLIC_ARRAY_WIDTH; i++) begin
                    g_win_r[i] <= g_win[i];
                end
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
                    g_win_r[i] <= '0;
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
                        rd_bias_scale <= 1'b0;
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
                        rd_bias_scale <= 1'b0;
                    end
                    8: begin
                        // Item 18a SCALE read: arms the SAME bias
                        // operand stream (same skew, same active
                        // window) with the ptr-7 elementwise linear
                        // walk, and sets the scale-arm flag that
                        // routes the VPU's head-of-chain multiply
                        // stage into the datapath. Issued mid-phase
                        // like the ptr-7 residual read, it arms only
                        // that phase's output stream: the scale stage
                        // then computes C = (A@W) . S. The phase
                        // pathway stays 0 -- the read alone arms the
                        // multiply.
                        rd_bias_ptr <= ub_rd_addr_in;
                        rd_bias_row_size <= ub_rd_row_size;
                        rd_bias_col_size <= ub_rd_col_size;
                        rd_bias_time_counter <= '0;
                        rd_bias_residual <= 1'b0;
                        rd_bias_scale <= 1'b1;
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
