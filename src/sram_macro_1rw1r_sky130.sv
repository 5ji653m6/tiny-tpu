`timescale 1ns/1ps
`default_nettype none

// Item 24: sky130 hard-macro implementation of the 1W/1R program-memory
// SRAM. Drop-in replacement for the behavioral sram_macro in the
// HARDENING flow only (selected by `ifdef SYNTH_SRAM_MACROS at the
// instr_seq_nxn instantiation site; simulation keeps the behavioral
// model, so the gate is unaffected).
//
// Assembles WIDTH-bit words from 32-bit-wide sky130_sram_1kbyte_1rw1r_
// 32x256_8 macros (ceil(WIDTH/32) banks; DEPTH must be <= 256). The
// macro's port 0 (RW) carries host program loads; port 1 (R) carries
// instruction fetch. Both are synchronous; the 1-cycle read latency
// matches the 2-stage fetch/execute pipeline the behavioral-model
// integration (d225881) introduced, so no sequencer changes are needed.
//
// rst is accepted for port compatibility but unused: hard macros have
// no array reset (the behavioral model's CLEAR_ON_RESET=0 for the
// program buffer already relies on this — reads only touch written
// words).
//
// This file references the foundry macro module, which exists only in
// the hardening flow (black-boxed from its LibreLane MACROS entry). It
// is never instantiated in simulation, so iverilog never elaborates it.

module sram_macro_1rw1r_sky130 #(
    parameter int WIDTH = 373,
    parameter int DEPTH = 256      // must be <= 256 (single bank row)
)(
    input  logic              clk,
    input  logic              rst,        // unused (see header)
    // 1 write port (host program load)
    input  logic [0:0]        wr_en,
    input  logic [15:0]       wr_addr,
    input  logic [WIDTH-1:0]  wr_data,
    // 1 read port (instruction fetch)
    input  logic [15:0]       rd_addr,
    output logic [WIDTH-1:0]  rd_data
);

    localparam int NUM_BANKS = (WIDTH + 31) / 32;

    // Quiet lint: rst and the upper address bits are intentionally
    // unused (single 256-deep bank row, no array reset).
    logic unused;
    assign unused = rst | (|wr_addr[15:8]) | (|rd_addr[15:8]);

    genvar b;
    generate
        for (b = 0; b < NUM_BANKS; b++) begin : bank
            localparam int LO = b * 32;
            localparam int REM = WIDTH - LO;      // bits left: >32 or 1..32
            logic [31:0] din, dout;
            if (REM >= 32) begin : full
                assign din = wr_data[LO +: 32];
                assign rd_data[LO +: 32] = dout;
            end else begin : partial
                // Top bank: zero-pad the write data, drop the unused
                // high read bits.
                assign din = {{(32-REM){1'b0}}, wr_data[LO +: REM]};
                assign rd_data[LO +: REM] = dout[REM-1:0];
            end

            sky130_sram_1kbyte_1rw1r_32x256_8 macro_inst (
                // Port 0 (RW): host program load
                .clk0   (clk),
                .csb0   (1'b0),
                .web0   (~wr_en[0]),
                .wmask0 (4'b1111),
                .addr0  (wr_addr[7:0]),
                .din0   (din),
                .dout0  (),
                // Port 1 (R): instruction fetch (synchronous read)
                .clk1   (clk),
                .csb1   (1'b0),
                .addr1  (rd_addr[7:0]),
                .dout1  (dout)
            );
        end
    endgenerate

endmodule
