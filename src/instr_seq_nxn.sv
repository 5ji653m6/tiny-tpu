`timescale 1ns/1ps
`default_nettype none

// Loadable instruction sequencer (roadmap items 9b + 12): stores a
// program of {ctrl, 134+17*(N-2)-bit legacy word} program words and
// replays it, one word per cycle, into an instruction consumer
// (tpu_nxn_ic). instr_out STAYS the legacy width -- the consumer
// contract (tpu_nxn_ic.instruction) is unchanged.
//
// Program word format (item 12: ONE bit wider than item 9b):
//   ctrl = 0: plain instruction word. Replayed exactly as in item 9b
//             with the ctrl bit stripped (pre-item-12 programs replay
//             bit-identically).
//   ctrl = 1: control word. bits [1:0] select the op:
//             2'b00 = LOOP: count = bits [15:8], len = bits [7:0].
//             The `len` program words immediately following the LOOP
//             word are the body; the body executes `count` times IN
//             TOTAL (count >= 1); count = 0 skips the body entirely.
//             The LOOP word itself emits exactly ONE all-zero cycle
//             (one bubble per loop, NOT per iteration -- iterations
//             are back-to-back). After the final pass, replay
//             continues with the word after the body.
//             Other ops are RESERVED: one all-zero cycle, no other
//             action.
//   Loop state resets on every run pulse: a re-run replays the loop
//   identically. Out of contract (never gated): a control word at
//   program index 0, control words inside a body, nested loops, a
//   body running past the loaded program.
//
// Usage model (unchanged from item 9b):
//   1. While NOT busy, the host loads the program one word per cycle
//      (prog_wr_en + prog_wr_data); the write pointer auto-increments
//      from 0. Writes while busy are IGNORED (a replay in flight is
//      never corrupted by host writes).
//   2. A 1-cycle `run` pulse while not busy starts the replay IF the
//      write pointer is nonzero; a run with an empty program is
//      ignored (busy stays 0). Starting the cycle AFTER the run pulse
//      is sampled, instr_out presents the expanded emission stream
//      (LOOP bodies unrolled, one bubble per control word), one word
//      per cycle, with busy = 1. The cycle after the last emission,
//      instr_out returns to 0 and busy drops.
//   3. A later run pulse replays the SAME program: running does NOT
//      clear the write pointer. Loading a different program requires
//      rst first.
//
// Reset-note: rst clears busy/instr_out (and loop state) every cycle
// it is held, but clears the write pointer only on ENTRY into reset
// (0->1) -- and a write takes priority even over that entry clear. A
// program loaded while the chip is held in reset therefore survives
// until rst releases (tpu_nxn_prog's testbench loads exactly this
// way), and a fresh rst still resets the pointer for loading a
// different program. All state is registered, synchronous to clk.

module instr_seq_nxn #(
    parameter int SYSTOLIC_ARRAY_WIDTH = 2,
    parameter int PROG_DEPTH = 256
) (
    input logic clk,
    input logic rst,

    // Program load interface (ignored while busy). The program word
    // is ONE bit wider than instr_out: {ctrl, legacy_word}.
    input logic prog_wr_en,
    input logic [134+17*(SYSTOLIC_ARRAY_WIDTH-2):0] prog_wr_data,

    // 1-cycle pulse starts the replay (ignored while busy or empty)
    input logic run,

    output logic busy,
    output logic [133+17*(SYSTOLIC_ARRAY_WIDTH-2):0] instr_out
);

    localparam int N = SYSTOLIC_ARRAY_WIDTH;
    localparam int W = 134 + 17*(N-2);   // legacy instruction width
    localparam int PW = W + 1;           // program word = {ctrl, word}
    // Wide enough to represent PROG_DEPTH itself (full memory).
    localparam int PTR_W = $clog2(PROG_DEPTH + 1);

    // Program memory and load pointer
    logic [PW-1:0] prog_mem [PROG_DEPTH];
    logic [PTR_W-1:0] wr_ptr;

    // Replay read pointer
    logic [PTR_W-1:0] rd_ptr;

    // Loop state (single level; reset on every run pulse). loop_start
    // is the index of the first body word, loop_end one past the last
    // body word, loop_iters the passes remaining AFTER the current
    // one (LOOP count-1 at setup).
    logic loop_active;
    logic [PTR_W-1:0] loop_start;
    logic [PTR_W-1:0] loop_end;
    logic [7:0] loop_iters;

    // Previous-cycle rst, for entry-into-reset detection. Initialized
    // deasserted so a rst already high at time 0 still clears wr_ptr.
    logic rst_d = 1'b0;

    // A body pass finished this cycle with passes remaining: fetch the
    // body-start word instead of rd_ptr (iterations are back-to-back,
    // no bubble between passes).
    wire loop_jump = loop_active && (rd_ptr == loop_end)
                                 && (loop_iters != '0);
    wire [PTR_W-1:0] fetch_ptr = loop_jump ? loop_start : rd_ptr;

    // The word fetched this cycle (async read of the program memory).
    wire [PW-1:0] fetch_word = prog_mem[fetch_ptr];
    wire fetch_ctrl = fetch_word[W];
    // Op decode (bits [1:0]): op 2'b00 is LOOP; ops with bit 0 set
    // are reserved. NOTE: the LOOP len field (bits [7:0]) ALIASES op
    // bit 1 -- a len=2 body reads as op 2'b10, len=52 as 2'b00, and
    // both are LOOPs -- so op bit 0 is the reliable discriminator:
    // OP_LOOP = 2'b00 and every LOOP word has bit 0 clear; reserved
    // ops (2'b01/2'b11) have it set.
    wire fetch_is_loop = ~fetch_word[0];

    always @(posedge clk) begin
        rst_d <= rst;
        if (rst) begin
            busy        <= 1'b0;
            instr_out   <= '0;
            rd_ptr      <= '0;
            loop_active <= 1'b0;
            if (prog_wr_en && (wr_ptr < PROG_DEPTH[PTR_W-1:0])) begin
                // Write takes priority over the pointer clear so a
                // program can be loaded while the chip is held in rst.
                prog_mem[wr_ptr] <= prog_wr_data;
                wr_ptr           <= wr_ptr + 1'b1;
            end else if (!rst_d) begin
                // Entry into reset: clear the write pointer exactly once.
                wr_ptr <= '0;
            end
            // else: hold wr_ptr -- a program loaded during this reset
            // survives until rst releases.
        end else if (busy) begin
            // Replay in flight: writes are ignored (no prog_wr_en branch).
            if (loop_jump) begin
                // Another body pass: consume one remaining iteration.
                loop_iters <= loop_iters - 1'b1;
            end else if (loop_active && (rd_ptr == loop_end)) begin
                // Final pass complete: replay continues with the word
                // after the body (rd_ptr already points at it).
                loop_active <= 1'b0;
            end

            if (loop_jump || (rd_ptr < wr_ptr)) begin
                if (fetch_ctrl) begin
                    // Control word: exactly ONE all-zero bubble cycle.
                    instr_out <= '0;
                    if (fetch_is_loop) begin
                        if (fetch_word[15:8] != '0) begin
                            // LOOP, count >= 1: arm the loop state and
                            // step into the body's first pass.
                            loop_active <= 1'b1;
                            loop_start  <= fetch_ptr + 1'b1;
                            loop_end    <= fetch_ptr + 1'b1
                                         + fetch_word[7:0];
                            loop_iters  <= fetch_word[15:8] - 1'b1;
                            rd_ptr      <= fetch_ptr + 1'b1;
                        end else begin
                            // LOOP, count = 0: skip the body entirely.
                            rd_ptr <= fetch_ptr + 1'b1
                                    + fetch_word[7:0];
                        end
                    end else begin
                        // Reserved op: bubble only, no other action.
                        rd_ptr <= fetch_ptr + 1'b1;
                    end
                end else begin
                    // Plain instruction word: replay with ctrl stripped.
                    instr_out <= fetch_word[W-1:0];
                    rd_ptr    <= fetch_ptr + 1'b1;
                end
            end else begin
                // Last emission was presented last cycle: return to idle.
                instr_out   <= '0;
                busy        <= 1'b0;
                rd_ptr      <= '0;
                loop_active <= 1'b0;
            end
        end else begin
            // Idle: host may load the program and/or pulse run.
            if (prog_wr_en && (wr_ptr < PROG_DEPTH[PTR_W-1:0])) begin
                prog_mem[wr_ptr] <= prog_wr_data;
                wr_ptr           <= wr_ptr + 1'b1;
            end
            if (run && (wr_ptr != '0)) begin
                // Run pulse sampled with a non-empty program: present
                // prog[0] the very next cycle (this NBA update), with
                // busy = 1. The write pointer is NOT cleared, so a
                // later run replays the same program; the loop state
                // is reset so the re-run replays loops identically.
                busy        <= 1'b1;
                instr_out   <= prog_mem[0][W-1:0];
                rd_ptr      <= {{(PTR_W-1){1'b0}}, 1'b1};
                loop_active <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
