`timescale 1ns/1ps
`default_nettype none

// SRAM-based instruction sequencer (roadmap items 9b + 12 + 15 + SRAM
// integration). Stores a program of {ctrl, 134+17*(N-2)-bit legacy
// word} program words in an SRAM macro and replays it, one word per
// cycle, into an instruction consumer (tpu_nxn_ic). instr_out STAYS
// the legacy width -- the consumer contract (tpu_nxn_ic.instruction)
// is unchanged.
//
// SRAM integration: prog_mem is an sram_macro (behavioral model of a
// foundry SRAM). Reads are SYNCHRONOUS (1-cycle latency) instead of
// combinational, so the sequencer is a 2-stage fetch/execute pipeline:
// the SRAM read address is presented one cycle BEFORE the word is
// processed. fetch_ptr_r holds the address of the word currently on
// sr_rd_data (the execute-stage word); next_addr -- computed
// combinationally from the execute-stage word and the loop state --
// drives the SRAM read port and is registered into fetch_ptr_r.
// Throughput is unchanged (1 word/cycle); the only externally visible
// difference vs. the pre-SRAM (combinational-read) version is ONE
// all-zero bubble cycle between the run pulse and the first program
// word. Area savings: 8192x373 bits x 40T (DFF) -> 6T (SRAM) = 85%
// reduction at N=16.
//
// Program word format (item 12: ONE bit wider than item 9b):
//   ctrl = 0: plain instruction word. Replayed exactly as in item 9b
//             with the ctrl bit stripped (pre-item-12 programs replay
//             bit-identically).
//   ctrl = 1: control word. bits [1:0] select the op:
//             2'b00 = LOOP: count = bits [15:8], len = the 15-bit
//             field { bits [23:17], bits [7:0] } (item 17b1: bits
//             [23:17] ride the previously-unused control-word bits
//             between the indexed flag and stride_a; every pre-17b1
//             program has len <= 255, i.e. those bits 0, and replays
//             bit-identically).
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
//     bits [71:56] wbase    (16-bit unsigned, item 17a; ZERO = exact
//                  item-15 behavior)
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
//   Item 17b1 (LOOXL): the loop body length is the 15-bit field
//   { bits [23:17], bits [7:0] }. The extension is ORTHOGONAL to the
//   indexed flag: a large body with the flag CLEAR replays verbatim
//   every pass; with the flag SET all the item-15/17a transforms
//   apply across the WHOLE body, including reads at body word offsets
//   past 255. The op decode needs no change: a large-len loop word
//   has count (bits [15:8]) nonzero, so `~word[0] | (count != 0)`
//   keeps decoding it as a loop even when the low len byte is odd.
//
//   Item 17a (wbase): with the indexed flag SET, a ptr-1 body read
//   advances by i*stride_w ONLY when its addr >= wbase (the boundary
//   addr == wbase advances); below wbase it passes bit-identical --
//   stationary host weights live below wbase, chip-produced
//   per-iteration intermediates at/above it. A ptr-7 body read (the
//   item-17a residual read) advances by i*stride_a (same stride as
//   ptr 0 -- residuals are activations). wbase = 0 advances every
//   ptr-1 read (exact item-15 behavior). With the flag CLEAR the
//   wbase field is ignored along with the strides (exact item-12).
//
//   Loop state -- the iteration counter included -- resets on every
//   run pulse: a re-run replays the loop identically. Out of contract
//   (never gated): a control word at program index 0, control words
//   inside a body, nested loops, a body running past the loaded
//   program.
//
//   SRAM-pipeline note: loop bookkeeping (iteration decrement, pass
//   index increment, loop_active clear) happens when the LAST body
//   word of a pass is processed (fetch_ptr_r == loop_end-1), one
//   word-slot earlier than the pre-SRAM version -- the jump redirect
//   must be known at FETCH time, one cycle before the jumped word is
//   executed. loop_iter is the 0-based index of the pass currently
//   being executed and is used directly as the emission index.
//
// Usage model:
//   1. While NOT busy, the host loads the program one word per cycle
//      (prog_wr_en + prog_wr_data); the write pointer auto-increments
//      from 0. Writes while busy are IGNORED (a replay in flight is
//      never corrupted by host writes).
//   2. A 1-cycle `run` pulse while not busy starts the replay IF the
//      write pointer is nonzero; a run with an empty program is
//      ignored (busy stays 0). Starting the cycle AFTER the run pulse
//      is sampled, busy = 1 and instr_out presents ONE all-zero
//      bubble cycle (the SRAM read latency), then the expanded
//      emission stream (LOOP bodies unrolled, one bubble per control
//      word), one word per cycle. The cycle after the last emission,
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
    // Item 17b1: the default depth now covers a full extended loop
    // body (the item-17b capstone needs 838+ words). tpu_nxn_prog
    // passes its own PROG_DEPTH explicitly and is unaffected.
    parameter int PROG_DEPTH = 1024
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

    // SRAM-based program memory (replaces DFF array for 85% area
    // reduction). Behavioral model synthesizes to a foundry SRAM macro.
    logic sr_wr_en;
    logic [15:0] sr_wr_addr;
    logic [PW-1:0] sr_wr_data;
    logic [15:0] sr_rd_addr;
    logic [PW-1:0] sr_rd_data;

    sram_macro #(
        .WIDTH(PW),
        .DEPTH(PROG_DEPTH),
        .NUM_WRITE(1),
        .NUM_READ(1)
    ) prog_sram (
        .clk(clk),
        .wr_en(sr_wr_en),
        .wr_addr(sr_wr_addr),
        .wr_data(sr_wr_data),
        .rd_addr(sr_rd_addr),
        .rd_data(sr_rd_data)
    );

    logic [PTR_W-1:0] wr_ptr;

    // Fetch pipeline: the SRAM read is synchronous, so the address
    // presented this cycle yields its word on sr_rd_data NEXT cycle.
    // fetch_ptr_r holds the address of the word currently being
    // EXECUTED (the word on sr_rd_data). It is registered from
    // next_addr every cycle, keeping address and data aligned.
    logic [PTR_W-1:0] fetch_ptr_r;

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
    // the 0-based index of the pass currently being executed.
    logic loop_indexed;
    logic [15:0] loop_stride_a;
    logic [15:0] loop_stride_w;
    // Item 17a: ptr-1 reads below wbase are frozen (stationary host
    // weights); at/above wbase they advance by stride_w.
    logic [15:0] loop_wbase;
    logic [7:0] loop_iter;

    // Previous-cycle rst, for entry-into-reset detection. Initialized
    // deasserted so a rst already high at time 0 still clears wr_ptr.
    logic rst_d = 1'b0;

    // ---- Execute-stage decode (of sr_rd_data, the word fetched LAST
    // cycle at address fetch_ptr_r) ----
    wire fetch_ctrl = sr_rd_data[W];
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
    wire fetch_is_loop = ~sr_rd_data[0] | (sr_rd_data[15:8] != '0);
    // Item 17b1: the loop body length is the 15-bit field
    // { bits [23:17], bits [7:0] }.
    wire [14:0] fetch_len = {sr_rd_data[23:17], sr_rd_data[7:0]};

    // ---- Next-address computation (drives the SRAM read port) ----
    // The word on sr_rd_data next cycle is the word we execute next
    // cycle, so the read address must be the NEXT program address:
    //   - idle / done: prefetch address 0 (startup + re-run preload)
    //   - last body word of a pass with passes remaining: loop_start
    //     (the jump redirect, known one cycle before the jumped word
    //     executes)
    //   - LOOP control word with count = 0: skip the body entirely
    //   - everything else (plain word, LOOP count >= 1, reserved op,
    //     last body word of the final pass): sequential +1
    wire exec_valid = busy && (fetch_ptr_r < wr_ptr);
    wire last_body = loop_active
                     && (fetch_ptr_r == (loop_end - 1'b1));
    wire loop_jump_fetch = last_body && (loop_iters != '0);
    wire skip_body = exec_valid && fetch_ctrl && fetch_is_loop
                     && (sr_rd_data[15:8] == '0);
    wire [PTR_W-1:0] next_addr = !exec_valid ? {PTR_W{1'b0}}
        : loop_jump_fetch ? loop_start
        : skip_body ? (fetch_ptr_r + 1'b1 + fetch_len)
        : (fetch_ptr_r + 1'b1);

    // SRAM port connections (combinational)
    assign sr_wr_en = prog_wr_en && !busy && (wr_ptr < PROG_DEPTH[PTR_W-1:0]);
    assign sr_wr_addr = {{(16-PTR_W){1'b0}}, wr_ptr};
    assign sr_wr_data = prog_wr_data;
    assign sr_rd_addr = {{(16-PTR_W){1'b0}}, next_addr};

    // ---- Item 15: indexed-loop (LOOPI) emission transform ----
    // The address advance applies only while the executed word is a
    // BODY word of an active indexed loop: the word executed AT
    // loop_end is already the first word after the body (final pass
    // complete) and must pass through untouched.
    wire in_loop_body = loop_active && loop_indexed
                        && (fetch_ptr_r != loop_end);
    // UB read command (bit 1 set) with ptr exactly 0, or ptr exactly 1
    // at/above wbase, or ptr exactly 7 (item-17a residual read), or ptr
    // exactly 8 (item-18a scale read).
    wire fetch_ptr0 = (sr_rd_data[61:53] == 9'd0);
    wire fetch_ptr1 = (sr_rd_data[61:53] == 9'd1);
    wire fetch_ptr7 = (sr_rd_data[61:53] == 9'd7);
    wire fetch_ptr8 = (sr_rd_data[61:53] == 9'd8);
    // wbase gate: the boundary addr == wbase advances (inclusive);
    // wbase = 0 advances every ptr-1 read (item-15 behavior).
    wire fetch_w_adv = fetch_ptr1 && (sr_rd_data[52:37] >= loop_wbase);
    wire indexable_read = in_loop_body && sr_rd_data[1]
                          && (fetch_ptr0 || fetch_w_adv || fetch_ptr7
                              || fetch_ptr8);
    // loop_iter is the 0-based pass index of the CURRENT pass: the
    // increment happens when the LAST body word of the pass executes,
    // so every body word of pass i (including the last) sees i.
    wire [7:0] emit_iter = loop_iter;
    // ptr-1 reads stride by stride_w; ptr-0, ptr-7 and ptr-8 reads all
    // stride by stride_a (residuals and scale matrices are activations).
    wire [15:0] emit_stride = fetch_ptr1 ? loop_stride_w
                                         : loop_stride_a;
    // 16-bit unsigned add on the addr field; the i*stride offset is
    // computed mod 2^16 (offset overflow is out of contract).
    wire [15:0] emit_offset = emit_iter * emit_stride;
    wire [15:0] indexed_addr = sr_rd_data[52:37] + emit_offset;
    wire [W-1:0] emit_word = indexable_read
        ? {sr_rd_data[W-1:53], indexed_addr, sr_rd_data[36:0]}
        : sr_rd_data[W-1:0];

    always @(posedge clk) begin
        rst_d <= rst;

        if (rst) begin
            busy         <= 1'b0;
            instr_out    <= '0;
            fetch_ptr_r  <= '0;
            loop_active  <= 1'b0;
            loop_indexed <= 1'b0;
            loop_iter    <= '0;
            if (prog_wr_en && (wr_ptr < PROG_DEPTH[PTR_W-1:0])) begin
                // Write takes priority over the pointer clear so a
                // program can be loaded while the chip is held in rst.
                wr_ptr           <= wr_ptr + 1'b1;
            end else if (!rst_d) begin
                // Entry into reset: clear the write pointer exactly once.
                wr_ptr <= '0;
            end
            // else: hold wr_ptr -- a program loaded during this reset
            // survives until rst releases.
        end else begin
            // Fetch pipeline: register the address just presented to
            // the SRAM so next cycle's sr_rd_data stays paired with
            // its address. Runs every non-reset cycle.
            fetch_ptr_r <= next_addr;

            if (busy) begin
                // Replay in flight: host writes are ignored (sr_wr_en
                // is gated by !busy and wr_ptr is not touched here).
                if (exec_valid) begin
                    // Loop bookkeeping on the LAST body word of a
                    // pass (one word-slot earlier than the pre-SRAM
                    // version -- the jump must be known at fetch time).
                    if (last_body) begin
                        if (loop_iters != '0) begin
                            // Another body pass follows.
                            loop_iters <= loop_iters - 1'b1;
                            loop_iter  <= loop_iter + 1'b1;
                        end else begin
                            // Final pass complete: the next word (at
                            // loop_end) is the first word after the body.
                            loop_active <= 1'b0;
                        end
                    end

                    if (fetch_ctrl) begin
                        // Control word: exactly ONE all-zero bubble cycle.
                        instr_out <= '0;
                        if (fetch_is_loop && (sr_rd_data[15:8] != '0)) begin
                            // LOOP, count >= 1: arm the loop state and
                            // step into the body's first pass (the
                            // sequential next_addr lands on the first
                            // body word).
                            loop_active   <= 1'b1;
                            loop_start    <= fetch_ptr_r + 1'b1;
                            loop_end      <= fetch_ptr_r + 1'b1 + fetch_len;
                            loop_iters    <= sr_rd_data[15:8] - 1'b1;
                            loop_indexed  <= sr_rd_data[16];
                            loop_stride_a <= sr_rd_data[39:24];
                            loop_stride_w <= sr_rd_data[55:40];
                            loop_wbase    <= sr_rd_data[71:56];
                            loop_iter     <= '0;
                        end
                        // LOOP count = 0: no state; next_addr already
                        // skips the body. Reserved ops: bubble only.
                    end else begin
                        // Plain instruction word: replay with ctrl stripped.
                        instr_out <= emit_word;
                    end
                end else begin
                    // fetch_ptr_r walked past the loaded program: the
                    // last emission was presented last cycle. Return
                    // to idle (next_addr already prefetching word 0
                    // for a potential re-run).
                    instr_out   <= '0;
                    busy        <= 1'b0;
                    loop_active <= 1'b0;
                end
            end else begin
                // Idle: host may load the program and/or pulse run.
                if (prog_wr_en && (wr_ptr < PROG_DEPTH[PTR_W-1:0])) begin
                    wr_ptr           <= wr_ptr + 1'b1;
                end
                if (run && (wr_ptr != '0)) begin
                    // Run pulse sampled with a non-empty program. The
                    // idle prefetch has already been reading address 0,
                    // so the first program word is on sr_rd_data next
                    // cycle; instr_out presents ONE all-zero bubble
                    // cycle first (the SRAM read latency).
                    busy         <= 1'b1;
                    instr_out    <= '0;
                    loop_active  <= 1'b0;
                    loop_indexed <= 1'b0;
                    loop_iter    <= '0;
                end
            end
        end
    end

endmodule

`default_nettype wire
