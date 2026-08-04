`timescale 1ns/1ps
`default_nettype none

// Item 25: banked UB storage — a drop-in replacement for the behavioral
// 2NW/6NR sram_macro that is routable by construction.
// See docs/UB_REARCHITECTURE.md for the full analysis.
//
// N banks of 1R1W storage (one sky130 1rw1r macro per bank in the
// hardening flow; the behavioral ub_bank_1rw1r in sim).
//   bank   = addr[$clog2(N)-1:0]   (a wire slice — zero logic)
//   offset = addr >> $clog2(N)
//
// CONTRACT (verified across every gated inference program by the UB_TRACE
// instrumentation + diag/ub_concurrency.py): no cycle ever presents more
// than one read request or more than one write request to the same bank,
// except same-address duplicates (which resolve identically to sram_macro:
// reads share the bank's output, writes follow low-port-index priority).
// Violations fire sim-only UB_BANKED_ASSERT $displays:
//   - ww: two writes, same bank, different addresses
//   - rr: two active reads, same bank, different addresses
//   - rw: an active read of an address being written this cycle
//     (behavioral models return OLD data here; a silicon macro returns X —
//     no gated inference program does this)
//
// Timing is identical to sram_macro: 1-cycle synchronous read, writes
// synchronous, read-during-write same address returns OLD data (pre-edge
// sample). CLEAR_ON_RESET parity is behavioral-only (`ifndef SYNTHESIS`);
// silicon gets a boot scrub (design doc section 6).

module ub_bank_1rw1r #(
    parameter int WIDTH = 16,
    parameter int BDEPTH = 2048,
    parameter int AW = 11           // $clog2(BDEPTH)
)(
    input  logic clk,
    input  logic rst,
    input  logic wr_en,
    input  logic [AW-1:0] wr_addr,
    input  logic [WIDTH-1:0] wr_data,
    input  logic [AW-1:0] rd_addr,
    output logic [WIDTH-1:0] rd_data
);
    logic [WIDTH-1:0] mem [BDEPTH];

`ifndef SYNTHESIS
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int i = 0; i < BDEPTH; i++) mem[i] <= '0;
            rd_data <= '0;
        end else begin
            if (wr_en) mem[wr_addr] <= wr_data;
            rd_data <= mem[rd_addr];
        end
    end
`else
    always @(posedge clk) begin
        if (wr_en) mem[wr_addr] <= wr_data;
        rd_data <= mem[rd_addr];
    end
`endif
endmodule


module ub_banked #(
    parameter int WIDTH = 16,
    parameter int DEPTH = 32768,    // must divide evenly by N
    parameter int N = 16
)(
    input  logic clk,
    input  logic rst,
    input  logic [2*N-1:0] wr_en,
    input  logic [2*N*16-1:0] wr_addr,
    input  logic [2*N*WIDTH-1:0] wr_data,
    input  logic [6*N*16-1:0] rd_addr,
    // (* keep *) on rd_data: hardening anchor — same role as the keep on
    // sram_macro's rd_data (the item-24 lesson). tpu_nxn_prog exposes no
    // result outputs, so without the keep the synthesis sweep deletes the
    // entire "unobservable" datapath and hardens a sequencer skeleton.
    // With SYNTH_UB_BANKED the behavioral sram_macro (and its keep) is
    // replaced by this module, so the anchor must move here. Ignored by
    // sim. (First N=8 hardening run without it: 50 KB skeleton netlist,
    // 6 of 8 prog macros + all 8 UB macros swept.)
    (* keep *) output logic [6*N*WIDTH-1:0] rd_data,
    // Sim-only per-port read-window mask (contract assertions; pruned in
    // synthesis). Bit p = read port p presents a real request this cycle.
    input  logic [6*N-1:0] rd_act
);
    localparam int NUM_WRITE = 2*N;
    localparam int NUM_READ  = 6*N;
    localparam int BSEL   = $clog2(N);
    localparam int BDEPTH = DEPTH / N;
    localparam int AW     = $clog2(BDEPTH);

    logic [AW-1:0]     bank_rd_addr [N];
    logic [WIDTH-1:0] bank_rd_data [N];
    logic [N-1:0]     bank_wr_en;
    logic [AW-1:0]    bank_wr_addr [N];
    logic [WIDTH-1:0] bank_wr_data [N];
`ifndef SYNTHESIS
    logic [15:0]      bank_wr_full [N];  // winning full write address (assertions)
`endif

    always_comb begin
        for (int b = 0; b < N; b++) begin
            bank_rd_addr[b] = '0;
            bank_wr_en[b]   = 1'b0;
            bank_wr_addr[b] = '0;
            bank_wr_data[b] = '0;
`ifndef SYNTHESIS
            bank_wr_full[b] = '0;
`endif
        end
        // Read request OR-select: at most one active port per bank by
        // contract; inactive ports present address 0 and OR away. Two
        // ACTIVE ports on one bank with different addresses corrupt the
        // address — caught by the rr assertion below.
        for (int p = 0; p < NUM_READ; p++) begin
            bank_rd_addr[rd_addr[p*16 +: BSEL]] |= rd_addr[p*16 + BSEL +: AW];
        end
        // Write arbitration: ascending port order, first come wins —
        // identical to sram_macro's low-index priority for same-address
        // writes. Different-address collisions are ww violations.
        for (int w = 0; w < NUM_WRITE; w++) begin
            if (wr_en[w] && !bank_wr_en[wr_addr[w*16 +: BSEL]]) begin
                bank_wr_en[wr_addr[w*16 +: BSEL]]   = 1'b1;
                bank_wr_addr[wr_addr[w*16 +: BSEL]] = wr_addr[w*16 + BSEL +: AW];
                bank_wr_data[wr_addr[w*16 +: BSEL]] = wr_data[w*WIDTH +: WIDTH];
`ifndef SYNTHESIS
                bank_wr_full[wr_addr[w*16 +: BSEL]] = wr_addr[w*16 +: 16];
`endif
            end
        end
    end

    genvar b;
    generate
        for (b = 0; b < N; b++) begin : bank_gen
`ifdef SYNTH_SRAM_MACROS
            // Hardening flow: sky130 1rw1r macro wrapper (lands with the
            // item-25 hardening step; see docs/UB_REARCHITECTURE.md).
            ub_bank_1rw1r_sky130 #(
                .WIDTH(WIDTH),
                .BDEPTH(BDEPTH),
                .AW(AW)
            ) bank_inst (
                .clk(clk),
                .wr_en(bank_wr_en[b]),
                .wr_addr(bank_wr_addr[b]),
                .wr_data(bank_wr_data[b]),
                .rd_addr(bank_rd_addr[b]),
                .rd_data(bank_rd_data[b])
            );
`else
            ub_bank_1rw1r #(
                .WIDTH(WIDTH),
                .BDEPTH(BDEPTH),
                .AW(AW)
            ) bank_inst (
                .clk(clk),
                .rst(rst),
                .wr_en(bank_wr_en[b]),
                .wr_addr(bank_wr_addr[b]),
                .wr_data(bank_wr_data[b]),
                .rd_addr(bank_rd_addr[b]),
                .rd_data(bank_rd_data[b])
            );
`endif
        end
    endgenerate

    // Per-port data return: N:1 mux of bank outputs. The select must be the
    // REQUEST cycle's bank bits — the banks register read data on the
    // request edge, while rd_addr has already advanced to the next beat when
    // the data is consumed (the walk is combinational from state that
    // updates on the same edge). Registering the select per port matches
    // sram_macro's per-port registered read (rd_data[p] <= mem[rd_addr[p]])
    // exactly. Select nets stay private per port — no cross-lane sharing,
    // nothing for ABC to merge (the item-24 4096-sink mux-select pathology
    // is structurally impossible here).
    logic [BSEL-1:0] rd_bank_sel_r [NUM_READ];
    always @(posedge clk) begin
        for (int p = 0; p < NUM_READ; p++) begin
            rd_bank_sel_r[p] <= rd_addr[p*16 +: BSEL];
        end
    end
    always_comb begin
        for (int p = 0; p < NUM_READ; p++) begin
            rd_data[p*WIDTH +: WIDTH] = bank_rd_data[rd_bank_sel_r[p]];
        end
    end

`ifndef SYNTHESIS
    // Flat VPI mirror named `mem`: cocotb gate tests read back results via
    // `ub_sram.mem[a]` (the behavioral sram_macro's array name). Exposing the
    // same name here keeps every test working unchanged on both storage
    // paths. Written procedurally with each bank's post-arbitration winning
    // write — no continuous-assign alias of the bank arrays (iverilog
    // functor-storm lesson), so sim cost is one extra DFF-array.
    logic [WIDTH-1:0] mem [DEPTH];
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int i = 0; i < DEPTH; i++) mem[i] <= '0;
        end else begin
            for (int b = 0; b < N; b++) begin
                if (bank_wr_en[b]) mem[bank_wr_full[b]] <= bank_wr_data[b];
            end
        end
    end
`endif

`ifndef SYNTHESIS
    // Contract assertions (sim only). Fires on any program that leaves the
    // verified inference contract — the banked UB must not be hardened for
    // such programs (see design doc: training scope, prime-P fallback).
    always @(posedge clk) begin
        if (!rst) begin
            for (int w1 = 0; w1 < NUM_WRITE; w1++) begin
                for (int w2 = w1 + 1; w2 < NUM_WRITE; w2++) begin
                    if (wr_en[w1] && wr_en[w2]
                            && wr_addr[w1*16 +: BSEL] == wr_addr[w2*16 +: BSEL]
                            && wr_addr[w1*16 +: 16] != wr_addr[w2*16 +: 16]) begin
                        $display("UB_BANKED_ASSERT ww: ports %0d,%0d addrs %0d,%0d t=%0t",
                                 w1, w2, wr_addr[w1*16 +: 16], wr_addr[w2*16 +: 16], $time);
                    end
                end
            end
            for (int p1 = 0; p1 < NUM_READ; p1++) begin
                for (int p2 = p1 + 1; p2 < NUM_READ; p2++) begin
                    if (rd_act[p1] && rd_act[p2]
                            && rd_addr[p1*16 +: BSEL] == rd_addr[p2*16 +: BSEL]
                            && rd_addr[p1*16 +: 16] != rd_addr[p2*16 +: 16]) begin
                        $display("UB_BANKED_ASSERT rr: ports %0d,%0d addrs %0d,%0d t=%0t",
                                 p1, p2, rd_addr[p1*16 +: 16], rd_addr[p2*16 +: 16], $time);
                    end
                    // Same address read by two active ports is FINE (shared
                    // bank output) — not asserted.
                end
                if (rd_act[p1] && bank_wr_en[rd_addr[p1*16 +: BSEL]]
                        && bank_wr_full[rd_addr[p1*16 +: BSEL]] == rd_addr[p1*16 +: 16]) begin
                    $display("UB_BANKED_ASSERT rw: port %0d reads addr %0d during write t=%0t",
                             p1, rd_addr[p1*16 +: 16], $time);
                end
            end
        end
    end
`endif
endmodule

`default_nettype wire
