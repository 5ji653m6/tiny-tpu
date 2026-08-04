`timescale 1ns/1ps
`default_nettype none

// Item 25: sky130 hard-macro implementation of ONE ub_banked bank
// (1rw1r, WIDTH=16 words). HARDENING flow only — selected by `ifdef
// SYNTH_SRAM_MACROS at the ub_banked bank_gen instantiation; simulation
// keeps the behavioral ub_bank_1rw1r, so the gate is unaffected.
//
// Packs two 16-bit words per 32-bit row of sky130_sram_2kbyte_1rw1r_
// 32x512_8 macros: word w lives at row w>>1, half w&0. Writes replicate
// the data into both halves and select with wmask0 (0011 low / 1100
// high). ROWS_PER_MACRO = 512 rows = 1024 words; NUM_MACRO =
// ceil(BDEPTH/1024) (1 at N=8/UB 8K, 2 at N=16/UB 32K — the design
// doc's 32-UB-macro census, docs/UB_REARCHITECTURE.md section 6).
//
// TIMING: both ports synchronous; the 1-cycle read latency matches the
// behavioral bank exactly. The read HALF select and MACRO select are
// REGISTERED at the request edge — the macro dout answers the request
// cycle's address while rd_addr has already advanced (the item-25
// registered-select lesson; see ub_banked.sv).
//
// No array reset: hard macros have none. The behavioral model's
// CLEAR_ON_RESET parity is sim-only; silicon gets a boot scrub (design
// doc section 6). rst is not even a port here.
//
// This file references the foundry macro module, which exists only in
// the hardening flow (black-boxed from its LibreLane MACROS entry). It
// is never instantiated in simulation, so iverilog never elaborates it.

module ub_bank_1rw1r_sky130 #(
    parameter int WIDTH = 16,        // fixed 16 (two words per 32-bit row)
    parameter int BDEPTH = 2048,     // 16-bit words; must be a multiple of 1024*NUM_MACRO rounding below
    parameter int AW = 11            // $clog2(BDEPTH)
)(
    input  logic              clk,
    input  logic              wr_en,
    input  logic [AW-1:0]     wr_addr,
    input  logic [WIDTH-1:0]  wr_data,
    input  logic [AW-1:0]     rd_addr,
    output logic [WIDTH-1:0]  rd_data
);
    localparam int WORDS_PER_MACRO = 1024;              // 512 rows x 2 words
    localparam int NUM_MACRO = (BDEPTH + WORDS_PER_MACRO - 1) / WORDS_PER_MACRO;
    localparam int MSEL = (NUM_MACRO > 1) ? $clog2(NUM_MACRO) : 1;

    // Request-cycle decode. Row is always addr[9:1] (512 rows/macro),
    // half is addr[0]; for NUM_MACRO>1 the macro select is the top bit(s).
    // Supported configs use every address bit (BDEPTH=1024: AW=10, one
    // macro; BDEPTH=2048: AW=11, two macros) — no unused slices.
    logic [MSEL-1:0] rd_msel, wr_msel;
    logic [8:0]      rd_row,  wr_row;
    logic            rd_half, wr_half;
    assign rd_msel = (NUM_MACRO > 1) ? rd_addr[AW-1 -: MSEL] : '0;
    assign wr_msel = (NUM_MACRO > 1) ? wr_addr[AW-1 -: MSEL] : '0;
    assign rd_row  = rd_addr[9:1];
    assign wr_row  = wr_addr[9:1];
    assign rd_half = rd_addr[0];
    assign wr_half = wr_addr[0];

    // Registered request-cycle selects for the read-data return (macro
    // dout answers the request; rd_addr has advanced by consumption time).
    logic [MSEL-1:0] rd_msel_r;
    logic            rd_half_r;
    always @(posedge clk) begin
        rd_msel_r <= rd_msel;
        rd_half_r <= rd_half;
    end

    logic [31:0] macro_dout [NUM_MACRO];

    genvar m;
    generate
        for (m = 0; m < NUM_MACRO; m++) begin : macro
            logic wr_here;
            assign wr_here = wr_en && (wr_msel == m);

            sky130_sram_2kbyte_1rw1r_32x512_8 macro_inst (
                // Port 0 (RW): writes only (reads never use port 0).
                .clk0   (clk),
                .csb0   (1'b0),
                .web0   (~wr_here),
                .wmask0 (wr_half ? 4'b1100 : 4'b0011),
                .addr0  (wr_row),
                .din0   ({wr_data, wr_data}),
                .dout0  (),
                // Port 1 (R): synchronous read, chip-selected to this
                // macro only.
                .clk1   (clk),
                .csb1   (rd_msel != m),
                .addr1  (rd_row),
                .dout1  (macro_dout[m])
            );
        end
    endgenerate

    // Read-data return: registered macro + half select.
    logic [31:0] dout_sel;
    assign dout_sel = macro_dout[rd_msel_r];
    assign rd_data  = rd_half_r ? dout_sel[31:16] : dout_sel[15:0];

endmodule
