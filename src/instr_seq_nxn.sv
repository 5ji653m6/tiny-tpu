`timescale 1ns/1ps
`default_nettype none

// Loadable instruction sequencer (roadmap items 9b + 12 + 15): stores
// a program of {ctrl, 134+17*(N-2)-bit legacy word} program words and
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
//             Other ops are RESERVED (op bit 0 set with a ZERO count
//             field): one all-zero cycle, no other action. An odd len
//             aliases op bit 0 to 1 -- such words are still LOOPs,
//             discriminated by their nonzero count field (see the op
//             decode note below).
//
//   Item 15 (LOOPI) -- backward-compatible INDEXED LOOP extension.
//   Three fields ride in the previously-unused upper bits of the LOOP
//   control word:
//     bit  [16]    indexed flag
//     bits [39:24] stride_a (16-bit unsigned)
//     bits [55:40] stride_w (16-bit unsigned)
//   With the flag CLEAR the strides are IGNORED and behavior is
//   EXACTLY the item-12 LOOP (every pre-item-15 program replays
//   bit-identically). With the flag SET, on loop iteration i
//   (0-based) each body word that is a UB read command (bit 1 set)
//   AND whose ptr field (bits [61:53]) is exactly 0 is emitted with
//   its address field (bits [52:37]) replaced by addr + i*stride_a;
//   ptr exactly 1 gets addr + i*stride_w (16-bit unsigned add on the
//   addr field; overflow out of contract). EVERY other word --
//   non-read words, reads with any other ptr value (5/6 gradient
//   reads), switch/idle/host-write words -- passes through
//   bit-identical on every iteration.
//
//   Loop state -- the iteration counter included -- resets on every
//   run pulse: a re-run replays the loop identically. Out of contract
//   (never gated): a control word at program index 0, control words
//   inside a body, nested loops, a body running past the loaded
//   program.
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

    // Item-15 LOOPI state: when the LOOP control word's indexed flag
    // (bit [16]) is set, UB read addresses in the body advance per
    // iteration by stride_a (ptr 0) / stride_w (ptr 1). loop_iter is
    // the 0-based index of the pass currently being fetched.
    logic loop_indexed;
    logic [15:0] loop_stride_a;
    logic [15:0] loop_stride_w;
    logic [7:0] loop_iter;

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
    // Op decode (bits [1:0]): op 2'b00 is LOOP. NOTE: the LOOP len
    // field (bits [7:0]) ALIASES the op bits -- a len=2 body reads as
    // op 2'b10, len=52 as 2'b00, and an odd len (item 15, e.g. len=3)
    // reads as op 2'b01/2'b11 -- so the op bits alone cannot
    // discriminate. The item-12 RESERVED-op encodings carry a ZERO
    // count field (bits [15:8]); a LOOP word always names its pass
    // count there (count = 0 means skip, but such LOOPs keep op bit 0
    // clear). Decode: a control word is a LOOP when op bit 0 is clear
    // (the item-12 rule) OR the count field is nonzero (item 15:
    // odd-len bodies alias op bit 0 to 1).
    wire fetch_is_loop = ~fetch_word[0] | (fetch_word[15:8] != '0);

    // ---- Item 15: indexed-loop (LOOPI) emission transform ----
    // The address advance applies only while the fetched word is a
    // BODY word of an active indexed loop: the word fetched AT
    // loop_end is already the first word after the body (final pass
    // complete) and must pass through untouched.
    wire in_loop_body = loop_active && loop_indexed
                        && (fetch_ptr != loop_end);
    // UB read command (bit 1 set) with ptr exactly 0 or 1.
    wire indexable_read = in_loop_body && fetch_word[1]
                          && (fetch_word[61:53] <= 9'd1);
    // The word fetched on a jump cycle is the FIRST word of the NEXT
    // pass (loop_iter increments on the same edge), so the transform
    // uses the next-pass index there.
    wire [7:0] emit_iter = loop_jump ? (loop_iter + 1'b1) : loop_iter;
    wire [15:0] emit_stride = fetch_word[53] ? loop_stride_w
                                             : loop_stride_a;
    // 16-bit unsigned add on the addr field; the i*stride offset is
    // computed mod 2^16 (offset overflow is out of contract).
    wire [15:0] emit_offset = emit_iter * emit_stride;
    wire [15:0] indexed_addr = fetch_word[52:37] + emit_offset;
    wire [W-1:0] emit_word = indexable_read
        ? {fetch_word[W-1:53], indexed_addr, fetch_word[36:0]}
        : fetch_word[W-1:0];

    always @(posedge clk) begin
        rst_d <= rst;
        if (rst) begin
            busy         <= 1'b0;
            instr_out    <= '0;
            rd_ptr       <= '0;
            loop_active  <= 1'b0;
            loop_indexed <= 1'b0;
            loop_iter    <= '0;
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
                // Another body pass: consume one remaining iteration
                // and advance the 0-based pass index.
                loop_iters <= loop_iters - 1'b1;
                loop_iter  <= loop_iter + 1'b1;
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
                            // step into the body's first pass. The
                            // item-15 indexed flag / strides ride
                            // along; the pass index starts at 0.
                            loop_active   <= 1'b1;
                            loop_start    <= fetch_ptr + 1'b1;
                            loop_end      <= fetch_ptr + 1'b1
                                           + fetch_word[7:0];
                            loop_iters    <= fetch_word[15:8] - 1'b1;
                            loop_indexed  <= fetch_word[16];
                            loop_stride_a <= fetch_word[39:24];
                            loop_stride_w <= fetch_word[55:40];
                            loop_iter     <= '0;
                            rd_ptr        <= fetch_ptr + 1'b1;
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
                    // Plain instruction word: replay with ctrl
                    // stripped (the LOOPI address advance is applied
                    // when it is an indexable body read).
                    instr_out <= emit_word;
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
                // -- iteration counter included -- is reset so the
                // re-run replays loops identically.
                busy         <= 1'b1;
                instr_out    <= prog_mem[0][W-1:0];
                rd_ptr       <= {{(PTR_W-1){1'b0}}, 1'b1};
                loop_active  <= 1'b0;
                loop_indexed <= 1'b0;
                loop_iter    <= '0;
            end
        end
    end

endmodule

`default_nettype wire
