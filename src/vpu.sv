`timescale 1ns/1ps
`default_nettype none

// three data pathways:
// (forward pass hidden layer computations) input from sys --> bias --> leaky relu --> output
// (transition) input from sys --> bias --> leaky relu --> loss --> leaky relu derivative --> output
// (backward pass) input from sys --> leaky relu derivative --> output
// during the transition pathway we need to store the H matrices that come out of the leaky relu modules AND pass them to the loss modules

/* 
vpu_data_pathway is 7 bits: |sm(6)| |ln(5)| |gelu(4)| |bias(3)| |lr(2)| |loss(1)| |lr_d(0)|

0000000: activate no modules
0001100: forward pass pathway (sys --> bias --> leaky relu --> output)
0001111: transistion pathway (sys --> bias --> leaky relu --> loss --> leaky relu derivative --> output)
0000001: backward pass pathway (sys --> leaky relu derivative --> output)
001xxxx: gelu stage enabled (inserted between leaky relu and layernorm; bit 4 = 0 bypasses it)
01xxxxx: layernorm stage enabled (inserted between gelu and softmax; bit 5 = 0 bypasses it)
        normalizes each 2-lane beat: mean = (x1+x2)/2, var = ((x1-mean)^2+(x2-mean)^2)/2,
        y_i = (x_i - mean) / sqrt(var + eps)
1xxxxxx: softmax stage enabled (inserted between layernorm and loss; bit 6 = 0 bypasses it)
        normalizes each 2-lane beat: y_i = exp(x_i) / (exp(x_1) + exp(x_2)),
        computed in numerically stable logistic form y_1 = sigmoid(x_1 - x_2), y_2 = 1 - y_1
*/

module vpu (
    input logic clk,
    input logic rst,

    input logic [6:0] vpu_data_pathway, // 7-bits to signify which modules to route the inputs to (1 bit for each module; bit 4 = gelu, bit 5 = layernorm, bit 6 = softmax)

    // Inputs from systolic array
    input logic signed [15:0] vpu_data_in_1,
    input logic signed [15:0] vpu_data_in_2,
    input logic vpu_valid_in_1,
    input logic vpu_valid_in_2,

    // Inputs from UB
    input logic signed [15:0] bias_scalar_in_1,             // For bias modules
    input logic signed [15:0] bias_scalar_in_2,             // For bias modules
    input logic signed [15:0] lr_leak_factor_in,            // For leaky relu modules
    input logic signed [15:0] Y_in_1,                       // For loss modules
    input logic signed [15:0] Y_in_2,                       // For loss modules
    input logic signed [15:0] inv_batch_size_times_two_in,  // For loss modules
    input logic signed [15:0] H_in_1,                       // For leaky relu derivative modules
    input logic signed [15:0] H_in_2,                       // For leaky relu derivative modules 

    // Outputs to UB
    output logic signed [15:0] vpu_data_out_1,
    output logic signed [15:0] vpu_data_out_2,
    output logic vpu_valid_out_1,
    output logic vpu_valid_out_2
);

    // bias
    logic signed [15:0] bias_data_1_in; 
    logic bias_valid_1_in;
    logic signed [15:0] bias_data_2_in;
    logic bias_valid_2_in;
    logic signed [15:0] bias_z_data_out_1;
    logic bias_valid_1_out;
    logic signed [15:0] bias_z_data_out_2;
    logic bias_valid_2_out;

    // bias to lr intermediate values
    logic signed [15:0] b_to_lr_data_in_1;
    logic b_to_lr_valid_in_1;
    logic signed [15:0] b_to_lr_data_in_2;
    logic b_to_lr_valid_in_2;

    // lr
    logic signed [15:0] lr_data_1_in; 
    logic lr_valid_1_in;
    logic signed [15:0] lr_data_2_in;
    logic lr_valid_2_in;
    logic signed [15:0] lr_data_1_out;
    logic lr_valid_1_out;
    logic signed [15:0] lr_data_2_out;
    logic lr_valid_2_out;

    // lr to loss intermediate values
    logic signed [15:0] lr_to_loss_data_in_1;
    logic lr_to_loss_valid_in_1;
    logic signed [15:0] lr_to_loss_data_in_2;
    logic lr_to_loss_valid_in_2;

    // gelu
    logic signed [15:0] gelu_data_1_in;
    logic gelu_valid_1_in;
    logic signed [15:0] gelu_data_2_in;
    logic gelu_valid_2_in;
    logic signed [15:0] gelu_data_1_out;
    logic gelu_valid_1_out;
    logic signed [15:0] gelu_data_2_out;
    logic gelu_valid_2_out;

    // gelu to loss intermediate values
    logic signed [15:0] gelu_to_loss_data_in_1;
    logic gelu_to_loss_valid_in_1;
    logic signed [15:0] gelu_to_loss_data_in_2;
    logic gelu_to_loss_valid_in_2;

    // layernorm
    logic signed [15:0] ln_data_1_in;
    logic ln_valid_1_in;
    logic signed [15:0] ln_data_2_in;
    logic ln_valid_2_in;
    logic signed [15:0] ln_data_1_out;
    logic ln_valid_1_out;
    logic signed [15:0] ln_data_2_out;
    logic ln_valid_2_out;

    // layernorm to loss intermediate values
    logic signed [15:0] ln_to_loss_data_in_1;
    logic ln_to_loss_valid_in_1;
    logic signed [15:0] ln_to_loss_data_in_2;
    logic ln_to_loss_valid_in_2;

    // softmax
    logic signed [15:0] sm_data_1_in;
    logic sm_valid_1_in;
    logic signed [15:0] sm_data_2_in;
    logic sm_valid_2_in;
    logic signed [15:0] sm_data_1_out;
    logic sm_valid_1_out;
    logic signed [15:0] sm_data_2_out;
    logic sm_valid_2_out;

    // softmax to loss intermediate values
    logic signed [15:0] sm_to_loss_data_in_1;
    logic sm_to_loss_valid_in_1;
    logic signed [15:0] sm_to_loss_data_in_2;
    logic sm_to_loss_valid_in_2;

    // BUG-SKEW-1 fix: the systolic array delivers output column 1 one
    // cycle ahead of column 2 (UB wavefront stagger), and the per-lane
    // stages preserve that relative skew. Cross-lane group stages
    // (layernorm, softmax) must instead pair the two columns OF THE SAME
    // OUTPUT ROW — beats that arrive one cycle apart. Align lane 1 to
    // lane 2 at each group stage's input (1-cycle delay), then re-apply
    // the native skew at its output (delay lane 2 by 1 cycle) so
    // downstream per-lane stages and VPU output timing are unchanged.
    // Legacy pathways never enable these stages, so legacy timing and
    // the module-level stage tests (which drive aligned valids) are
    // unaffected.
    logic signed [15:0] ln_data_1_in_al;
    logic               ln_valid_1_in_al;
    logic signed [15:0] ln_data_2_out_rs;
    logic               ln_valid_2_out_rs;
    logic signed [15:0] sm_data_1_in_al;
    logic               sm_valid_1_in_al;
    logic signed [15:0] sm_data_2_out_rs;
    logic               sm_valid_2_out_rs;

    // loss
    logic signed [15:0] loss_data_1_in; 
    logic loss_valid_1_in;
    logic signed [15:0] loss_data_2_in;
    logic loss_valid_2_in;
    logic signed [15:0] loss_data_1_out;
    logic loss_valid_1_out;
    logic signed [15:0] loss_data_2_out;
    logic loss_valid_2_out;

    // loss to lrd intermediate values
    logic signed [15:0] loss_to_lrd_data_in_1;
    logic loss_to_lrd_valid_in_1;
    logic signed [15:0] loss_to_lrd_data_in_2;
    logic loss_to_lrd_valid_in_2;

    // lr_d
    logic signed [15:0] lr_d_data_1_in; 
    logic lr_d_valid_1_in;
    logic signed [15:0] lr_d_data_2_in;
    logic lr_d_valid_2_in;
    logic signed [15:0] lr_d_data_1_out;
    logic lr_d_valid_1_out;
    logic signed [15:0] lr_d_data_2_out;
    logic lr_d_valid_2_out;
    logic signed [15:0] lr_d_H_in_1;
    logic signed [15:0] lr_d_H_in_2;
    

    // temp 'last H matrix' cache
    logic signed [15:0] last_H_data_1_in;  // combinational input to H-cache register
    logic signed [15:0] last_H_data_2_in;  // combinational input to H-cache register
    logic signed [15:0] last_H_data_1_out;
    logic signed [15:0] last_H_data_2_out;

    // BUG-VPU-1 fix: intermediate mux signals; vpu_data_out is registered in always_ff
    logic signed [15:0] vpu_data_mux_1;
    logic signed [15:0] vpu_data_mux_2;
    logic               vpu_valid_mux_1;
    logic               vpu_valid_mux_2;

    bias_parent bias_parent_inst (  
        .clk(clk),
        .rst(rst),
        .bias_sys_data_in_1(bias_data_1_in),
        .bias_sys_data_in_2(bias_data_2_in),
        .bias_sys_valid_in_1(bias_valid_1_in),
        .bias_sys_valid_in_2(bias_valid_2_in),

        .bias_scalar_in_1(bias_scalar_in_1),
        .bias_scalar_in_2(bias_scalar_in_2),

        .bias_Z_valid_out_1(bias_valid_1_out),
        .bias_Z_valid_out_2(bias_valid_2_out),
        .bias_z_data_out_1(bias_z_data_out_1),
        .bias_z_data_out_2(bias_z_data_out_2),
        .bias_overflow_out_1(),   // BUG-OVF-1: observable via hierarchical reference
        .bias_overflow_out_2()
    );


    leaky_relu_parent leaky_relu_parent_inst (
        .clk(clk),
        .rst(rst),

        .lr_data_1_in(lr_data_1_in),
        .lr_data_2_in(lr_data_2_in),
        .lr_valid_1_in(lr_valid_1_in),
        .lr_valid_2_in(lr_valid_2_in),

        .lr_leak_factor_in(lr_leak_factor_in),
        
        .lr_data_1_out(lr_data_1_out),
        .lr_data_2_out(lr_data_2_out),
        .lr_valid_1_out(lr_valid_1_out),
        .lr_valid_2_out(lr_valid_2_out),
        .lr_overflow_out_1(),   // BUG-OVF-1: observable via hierarchical reference
        .lr_overflow_out_2()
    );

    gelu_parent gelu_parent_inst (
        .clk(clk),
        .rst(rst),

        .gelu_data_1_in(gelu_data_1_in),
        .gelu_data_2_in(gelu_data_2_in),
        .gelu_valid_1_in(gelu_valid_1_in),
        .gelu_valid_2_in(gelu_valid_2_in),

        .gelu_data_1_out(gelu_data_1_out),
        .gelu_data_2_out(gelu_data_2_out),
        .gelu_valid_1_out(gelu_valid_1_out),
        .gelu_valid_2_out(gelu_valid_2_out),
        .gelu_overflow_out_1(),   // BUG-OVF-1: observable via hierarchical reference
        .gelu_overflow_out_2()
    );

    layernorm_parent layernorm_parent_inst (
        .clk(clk),
        .rst(rst),

        .ln_data_1_in(ln_data_1_in_al),   // BUG-SKEW-1: aligned to lane 2
        .ln_data_2_in(ln_data_2_in),
        .ln_valid_1_in(ln_valid_1_in_al), // BUG-SKEW-1: aligned to lane 2
        .ln_valid_2_in(ln_valid_2_in),

        .ln_data_1_out(ln_data_1_out),
        .ln_data_2_out(ln_data_2_out),
        .ln_valid_1_out(ln_valid_1_out),
        .ln_valid_2_out(ln_valid_2_out),
        .ln_overflow_out_1(),   // BUG-OVF-1: observable via hierarchical reference
        .ln_overflow_out_2()
    );

    softmax_parent softmax_parent_inst (
        .clk(clk),
        .rst(rst),

        .sm_data_1_in(sm_data_1_in_al),   // BUG-SKEW-1: aligned to lane 2
        .sm_data_2_in(sm_data_2_in),
        .sm_valid_1_in(sm_valid_1_in_al), // BUG-SKEW-1: aligned to lane 2
        .sm_valid_2_in(sm_valid_2_in),

        .sm_data_1_out(sm_data_1_out),
        .sm_data_2_out(sm_data_2_out),
        .sm_valid_1_out(sm_valid_1_out),
        .sm_valid_2_out(sm_valid_2_out),
        .sm_overflow_out_1(),   // BUG-OVF-1: observable via hierarchical reference
        .sm_overflow_out_2()
    );

    loss_parent loss_parent_inst (
        .clk(clk),
        .rst(rst),
        .H_1_in(loss_data_1_in),
        .H_2_in(loss_data_2_in),
        .valid_1_in(loss_valid_1_in),
        .valid_2_in(loss_valid_2_in),

        .Y_1_in(Y_in_1),
        .Y_2_in(Y_in_2),
        .inv_batch_size_times_two_in(inv_batch_size_times_two_in),

        .gradient_1_out(loss_data_1_out),
        .gradient_2_out(loss_data_2_out),
        .valid_1_out(loss_valid_1_out),
        .valid_2_out(loss_valid_2_out),
        .loss_overflow_out_1(),   // BUG-OVF-1: observable via hierarchical reference
        .loss_overflow_out_2()
    );

    leaky_relu_derivative_parent leaky_relu_derivative_parent_inst (
        .clk(clk),
        .rst(rst),
        .lr_d_data_1_in(lr_d_data_1_in),
        .lr_d_data_2_in(lr_d_data_2_in),
        .lr_d_valid_1_in(lr_d_valid_1_in),
        .lr_d_valid_2_in(lr_d_valid_2_in),
         
        .lr_d_H_1_in(lr_d_H_in_1),
        .lr_d_H_2_in(lr_d_H_in_2),
        .lr_leak_factor_in(lr_leak_factor_in),
        
        .lr_d_data_1_out(lr_d_data_1_out),
        .lr_d_data_2_out(lr_d_data_2_out),
        .lr_d_valid_1_out(lr_d_valid_1_out),
        .lr_d_valid_2_out(lr_d_valid_2_out),
        .lr_d_overflow_out_1(),   // BUG-OVF-1: observable via hierarchical reference
        .lr_d_overflow_out_2()
    );

    always @(*) begin
        // Default assignments for all intermediate signals to prevent latch inference.
        // These are overridden by the routing logic below when rst is not asserted.
        b_to_lr_data_in_1     = 16'b0;
        b_to_lr_data_in_2     = 16'b0;
        b_to_lr_valid_in_1    = 1'b0;
        b_to_lr_valid_in_2    = 1'b0;
        lr_to_loss_data_in_1  = 16'b0;
        lr_to_loss_data_in_2  = 16'b0;
        lr_to_loss_valid_in_1 = 1'b0;
        lr_to_loss_valid_in_2 = 1'b0;
        gelu_to_loss_data_in_1  = 16'b0;
        gelu_to_loss_data_in_2  = 16'b0;
        gelu_to_loss_valid_in_1 = 1'b0;
        gelu_to_loss_valid_in_2 = 1'b0;
        ln_to_loss_data_in_1  = 16'b0;
        ln_to_loss_data_in_2  = 16'b0;
        ln_to_loss_valid_in_1 = 1'b0;
        ln_to_loss_valid_in_2 = 1'b0;
        sm_to_loss_data_in_1  = 16'b0;
        sm_to_loss_data_in_2  = 16'b0;
        sm_to_loss_valid_in_1 = 1'b0;
        sm_to_loss_valid_in_2 = 1'b0;
        loss_to_lrd_data_in_1  = 16'b0;
        loss_to_lrd_data_in_2  = 16'b0;
        loss_to_lrd_valid_in_1 = 1'b0;
        loss_to_lrd_valid_in_2 = 1'b0;
        last_H_data_1_in = 16'b0;
        last_H_data_2_in = 16'b0;
        lr_d_H_in_1 = 16'b0;
        lr_d_H_in_2 = 16'b0;

        if (rst) begin
            vpu_data_mux_1 = 16'b0;
            vpu_data_mux_2 = 16'b0;
            vpu_valid_mux_1 = 1'b0;
            vpu_valid_mux_2 = 1'b0;
            
            // default internal wire assignments during reset
            bias_data_1_in = 16'b0;
            bias_data_2_in = 16'b0;
            bias_valid_1_in = 1'b0;
            bias_valid_2_in = 1'b0;
            lr_data_1_in = 16'b0;
            lr_data_2_in = 16'b0;
            lr_valid_1_in = 1'b0;
            lr_valid_2_in = 1'b0;
            gelu_data_1_in = 16'b0;
            gelu_data_2_in = 16'b0;
            gelu_valid_1_in = 1'b0;
            gelu_valid_2_in = 1'b0;
            ln_data_1_in = 16'b0;
            ln_data_2_in = 16'b0;
            ln_valid_1_in = 1'b0;
            ln_valid_2_in = 1'b0;
            sm_data_1_in = 16'b0;
            sm_data_2_in = 16'b0;
            sm_valid_1_in = 1'b0;
            sm_valid_2_in = 1'b0;
            loss_data_1_in = 16'b0;
            loss_data_2_in = 16'b0;
            loss_valid_1_in = 1'b0;
            loss_valid_2_in = 1'b0;
            lr_d_data_1_in = 16'b0;
            lr_d_data_2_in = 16'b0;
            lr_d_valid_1_in = 1'b0;
            lr_d_valid_2_in = 1'b0;
        end else begin
            // bias module
            if(vpu_data_pathway[3]) begin
                // connect vpu inputs to bias module
                bias_data_1_in = vpu_data_in_1;
                bias_data_2_in = vpu_data_in_2;
                bias_valid_1_in = vpu_valid_in_1;
                bias_valid_2_in = vpu_valid_in_2;

                // connect bias output to intermediate values
                b_to_lr_data_in_1 = bias_z_data_out_1;
                b_to_lr_data_in_2 = bias_z_data_out_2;
                b_to_lr_valid_in_1 = bias_valid_1_out;
                b_to_lr_valid_in_2 = bias_valid_2_out;
            end else begin
                // disable inputs
                bias_data_1_in = 16'b0;
                bias_data_2_in = 16'b0;
                bias_valid_1_in = 1'b0;
                bias_valid_2_in = 1'b0;

                // connect vpu input to intermediate values
                b_to_lr_data_in_1 = vpu_data_in_1;
                b_to_lr_data_in_2 = vpu_data_in_2;
                b_to_lr_valid_in_1 = vpu_valid_in_1;
                b_to_lr_valid_in_2 = vpu_valid_in_2;
            end

            // leaky relu module
            if(vpu_data_pathway[2]) begin
                // connect lr inputs to intermediate values
                lr_data_1_in = b_to_lr_data_in_1;
                lr_data_2_in = b_to_lr_data_in_2;
                lr_valid_1_in = b_to_lr_valid_in_1;
                lr_valid_2_in = b_to_lr_valid_in_2;

                // connect lr outputs to intermediate values
                lr_to_loss_data_in_1 = lr_data_1_out;
                lr_to_loss_data_in_2 = lr_data_2_out;
                lr_to_loss_valid_in_1 = lr_valid_1_out;
                lr_to_loss_valid_in_2 = lr_valid_2_out;

            end else begin
                // disable inputs
                lr_data_1_in = 16'b0;
                lr_data_2_in = 16'b0;
                lr_valid_1_in = 1'b0;
                lr_valid_2_in = 1'b0;

                // connect intermediate values to each other
                lr_to_loss_data_in_1 = b_to_lr_data_in_1;
                lr_to_loss_data_in_2 = b_to_lr_data_in_2;
                lr_to_loss_valid_in_1 = b_to_lr_valid_in_1;
                lr_to_loss_valid_in_2 = b_to_lr_valid_in_2;
            end

            // gelu module
            if(vpu_data_pathway[4]) begin
                // connect gelu inputs to intermediate values
                gelu_data_1_in = lr_to_loss_data_in_1;
                gelu_data_2_in = lr_to_loss_data_in_2;
                gelu_valid_1_in = lr_to_loss_valid_in_1;
                gelu_valid_2_in = lr_to_loss_valid_in_2;

                // connect gelu outputs to intermediate values
                gelu_to_loss_data_in_1 = gelu_data_1_out;
                gelu_to_loss_data_in_2 = gelu_data_2_out;
                gelu_to_loss_valid_in_1 = gelu_valid_1_out;
                gelu_to_loss_valid_in_2 = gelu_valid_2_out;
            end else begin
                // disable inputs
                gelu_data_1_in = 16'b0;
                gelu_data_2_in = 16'b0;
                gelu_valid_1_in = 1'b0;
                gelu_valid_2_in = 1'b0;

                // bypass: connect intermediate values to each other
                gelu_to_loss_data_in_1 = lr_to_loss_data_in_1;
                gelu_to_loss_data_in_2 = lr_to_loss_data_in_2;
                gelu_to_loss_valid_in_1 = lr_to_loss_valid_in_1;
                gelu_to_loss_valid_in_2 = lr_to_loss_valid_in_2;
            end

            // layernorm module
            if(vpu_data_pathway[5]) begin
                // connect layernorm inputs to intermediate values
                ln_data_1_in = gelu_to_loss_data_in_1;
                ln_data_2_in = gelu_to_loss_data_in_2;
                ln_valid_1_in = gelu_to_loss_valid_in_1;
                ln_valid_2_in = gelu_to_loss_valid_in_2;

                // connect layernorm outputs to intermediate values
                ln_to_loss_data_in_1 = ln_data_1_out;
                ln_to_loss_data_in_2 = ln_data_2_out_rs;   // BUG-SKEW-1: re-skewed
                ln_to_loss_valid_in_1 = ln_valid_1_out;
                ln_to_loss_valid_in_2 = ln_valid_2_out_rs; // BUG-SKEW-1: re-skewed
            end else begin
                // disable inputs
                ln_data_1_in = 16'b0;
                ln_data_2_in = 16'b0;
                ln_valid_1_in = 1'b0;
                ln_valid_2_in = 1'b0;

                // bypass: connect intermediate values to each other
                ln_to_loss_data_in_1 = gelu_to_loss_data_in_1;
                ln_to_loss_data_in_2 = gelu_to_loss_data_in_2;
                ln_to_loss_valid_in_1 = gelu_to_loss_valid_in_1;
                ln_to_loss_valid_in_2 = gelu_to_loss_valid_in_2;
            end

            // softmax module
            if(vpu_data_pathway[6]) begin
                // connect softmax inputs to intermediate values
                sm_data_1_in = ln_to_loss_data_in_1;
                sm_data_2_in = ln_to_loss_data_in_2;
                sm_valid_1_in = ln_to_loss_valid_in_1;
                sm_valid_2_in = ln_to_loss_valid_in_2;

                // connect softmax outputs to intermediate values
                sm_to_loss_data_in_1 = sm_data_1_out;
                sm_to_loss_data_in_2 = sm_data_2_out_rs;   // BUG-SKEW-1: re-skewed
                sm_to_loss_valid_in_1 = sm_valid_1_out;
                sm_to_loss_valid_in_2 = sm_valid_2_out_rs; // BUG-SKEW-1: re-skewed
            end else begin
                // disable inputs
                sm_data_1_in = 16'b0;
                sm_data_2_in = 16'b0;
                sm_valid_1_in = 1'b0;
                sm_valid_2_in = 1'b0;

                // bypass: connect intermediate values to each other
                sm_to_loss_data_in_1 = ln_to_loss_data_in_1;
                sm_to_loss_data_in_2 = ln_to_loss_data_in_2;
                sm_to_loss_valid_in_1 = ln_to_loss_valid_in_1;
                sm_to_loss_valid_in_2 = ln_to_loss_valid_in_2;
            end

            // loss module
            if(vpu_data_pathway[1]) begin
                // connect loss inputs to intermediate values
                loss_data_1_in = sm_to_loss_data_in_1;
                loss_data_2_in = sm_to_loss_data_in_2;
                loss_valid_1_in = sm_to_loss_valid_in_1;
                loss_valid_2_in = sm_to_loss_valid_in_2;

                // connect loss outputs to intermediate values
                loss_to_lrd_data_in_1 = loss_data_1_out;
                loss_to_lrd_data_in_2 = loss_data_2_out;
                loss_to_lrd_valid_in_1 = loss_valid_1_out;
                loss_to_lrd_valid_in_2 = loss_valid_2_out;

                // Cache and use 'last H matrix'
                last_H_data_1_in = lr_data_1_out;
                last_H_data_2_in = lr_data_2_out;
                lr_d_H_in_1 = last_H_data_1_out;
                lr_d_H_in_2 = last_H_data_2_out;
            end else begin
                // disable inputs
                loss_data_1_in = 16'b0;
                loss_data_2_in = 16'b0;
                loss_valid_1_in = 1'b0;
                loss_valid_2_in = 1'b0;

                // connect intermediate values to each other
                loss_to_lrd_data_in_1 = sm_to_loss_data_in_1;
                loss_to_lrd_data_in_2 = sm_to_loss_data_in_2;
                loss_to_lrd_valid_in_1 = sm_to_loss_valid_in_1;
                loss_to_lrd_valid_in_2 = sm_to_loss_valid_in_2;

                // BUG-VPU-2 fix: clear last_H cache inputs when loss is not active
                last_H_data_1_in = 16'b0;
                last_H_data_2_in = 16'b0;
                // BUG-VPU-3 fix: use UB H-input only during backward pass (pathway[0]=1)
                lr_d_H_in_1 = H_in_1;
                lr_d_H_in_2 = H_in_2;
            end

            // leaky relu derivative module
            if(vpu_data_pathway[0]) begin
                lr_d_data_1_in = loss_to_lrd_data_in_1;
                lr_d_data_2_in = loss_to_lrd_data_in_2;
                lr_d_valid_1_in = loss_to_lrd_valid_in_1;
                lr_d_valid_2_in = loss_to_lrd_valid_in_2;

                // connect lr_d outputs to vpu mux output
                vpu_data_mux_1 = lr_d_data_1_out;
                vpu_data_mux_2 = lr_d_data_2_out;
                vpu_valid_mux_1 = lr_d_valid_1_out;
                vpu_valid_mux_2 = lr_d_valid_2_out;
            end else begin
                // BUG-VPU-4 fix: zero lrd module inputs when disabled to prevent spurious state
                lr_d_data_1_in = 16'b0;
                lr_d_data_2_in = 16'b0;
                lr_d_valid_1_in = 1'b0;
                lr_d_valid_2_in = 1'b0;

                // bypass: connect intermediate values directly to vpu mux output
                vpu_data_mux_1 = loss_to_lrd_data_in_1;
                vpu_data_mux_2 = loss_to_lrd_data_in_2;
                vpu_valid_mux_1 = loss_to_lrd_valid_in_1;
                vpu_valid_mux_2 = loss_to_lrd_valid_in_2;
            end
        end
    end

    // BUG-VPU-1 fix: register VPU outputs to prevent combinational glitches
    // BUG-VPU-2 fix: removed dual driver on last_H_data_*_in (was also driven by always_ff reset)
    // BUG-SKEW-1 fix: 1-cycle shift registers implementing the lane
    // alignment described at the declarations above. Free-running (no
    // enable) so valid pulse trains are delayed exactly one cycle;
    // when a group stage is bypassed these shift zeros and are unused.
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            ln_data_1_in_al   <= '0;
            ln_valid_1_in_al  <= '0;
            ln_data_2_out_rs  <= '0;
            ln_valid_2_out_rs <= '0;
            sm_data_1_in_al   <= '0;
            sm_valid_1_in_al  <= '0;
            sm_data_2_out_rs  <= '0;
            sm_valid_2_out_rs <= '0;
        end else begin
            // align: lane 1 arrives one cycle early, hold it for lane 2
            ln_data_1_in_al   <= ln_data_1_in;
            ln_valid_1_in_al  <= ln_valid_1_in;
            sm_data_1_in_al   <= sm_data_1_in;
            sm_valid_1_in_al  <= sm_valid_1_in;
            // re-skew: group-stage outputs are paired; delay lane 2 to
            // restore the native one-cycle column skew downstream
            ln_data_2_out_rs  <= ln_data_2_out;
            ln_valid_2_out_rs <= ln_valid_2_out;
            sm_data_2_out_rs  <= sm_data_2_out;
            sm_valid_2_out_rs <= sm_valid_2_out;
        end
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            last_H_data_1_out <= '0;
            last_H_data_2_out <= '0;
            vpu_data_out_1   <= '0;
            vpu_data_out_2   <= '0;
            vpu_valid_out_1  <= '0;
            vpu_valid_out_2  <= '0;
        end else begin
            vpu_data_out_1  <= vpu_data_mux_1;
            vpu_data_out_2  <= vpu_data_mux_2;
            vpu_valid_out_1 <= vpu_valid_mux_1;
            vpu_valid_out_2 <= vpu_valid_mux_2;
            if (vpu_data_pathway[1]) begin
                last_H_data_1_out <= last_H_data_1_in;
                last_H_data_2_out <= last_H_data_2_in;
            end else begin
                last_H_data_1_out <= '0;
                last_H_data_2_out <= '0;
            end 
        end
    end

endmodule