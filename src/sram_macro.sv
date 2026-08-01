`timescale 1ns/1ps
`default_nettype none

// Behavioral SRAM macro model for teaching/simulation.
// Synthesizes to foundry SRAM macros (e.g., sky130 SRAM).
// Area: ~6T per bit vs ~40T for DFF (6-7x area reduction).
//
// This is a multi-port SRAM with:
// - NUM_WRITE write ports (synchronous, priority arbitration)
// - NUM_READ read ports (synchronous, 1-cycle latency)
//
// Write priority: port 0 > port 1 > ... > port N-1
// (if multiple writes to same address in same cycle, lower-index wins)
//
// For the unified buffer: one instance with NUM_WRITE=2N (N gradient
// writeback + N merged VPU-stream/host), NUM_READ=6N (input, weight,
// bias/residual/scale, Y, H, grad bias/weight lane groups).
// For the program buffer: use NUM_WRITE=1 (host load), NUM_READ=1 (instruction fetch).
//
// CLEAR_ON_RESET (behavioral model ONLY): when 1, mem is zeroed while rst
// is asserted, matching the DFF arrays this model replaces (the unified
// buffer's gate suites rely on reset-zeroed storage). Real foundry macros
// have no array reset — a silicon swap needs a boot-time scrub pass instead.

module sram_macro #(
    parameter int WIDTH = 16,
    parameter int DEPTH = 1024,
    parameter int NUM_WRITE = 3,
    parameter int NUM_READ = 7,
    parameter int CLEAR_ON_RESET = 0
)(
    input logic clk,
    input logic rst,

    // Write ports (synchronous, priority: 0 > 1 > ... > N-1)
    // Packed arrays for iverilog compatibility
    input logic [NUM_WRITE-1:0] wr_en,
    input logic [NUM_WRITE*16-1:0] wr_addr,
    input logic [NUM_WRITE*WIDTH-1:0] wr_data,

    // Read ports (synchronous, 1-cycle latency)
    input logic [NUM_READ*16-1:0] rd_addr,
    output logic [NUM_READ*WIDTH-1:0] rd_data
);

    // SRAM array (synthesizes to foundry macro)
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    // Write logic with priority arbitration
    always_ff @(posedge clk) begin
        if (CLEAR_ON_RESET != 0 && rst) begin
            // Behavioral-only array clear (see header note).
            for (int i = 0; i < DEPTH; i++) begin
                mem[i] <= '0;
            end
        end else begin
            // Priority: port 0 wins over port 1, etc.
            // Loop from high to low so lower-index overwrites higher-index
            for (int i = NUM_WRITE-1; i >= 0; i--) begin
                if (wr_en[i]) begin
                    mem[wr_addr[i*16 +: 16]] <= wr_data[i*WIDTH +: WIDTH];
                end
            end

            // Read logic (synchronous, 1-cycle latency)
            for (int i = 0; i < NUM_READ; i++) begin
                rd_data[i*WIDTH +: WIDTH] <= mem[rd_addr[i*16 +: 16]];
            end
        end
    end

    // Optional: write-through for port 0 (for debugging/observability)
    // Uncomment if needed:
    // assign rd_data[0] = wr_en[0] ? wr_data[0] : mem[rd_addr[0]];

endmodule
