`timescale 1ns/1ps
`default_nettype none

// Item 25: lockstep equivalence of ub_banked vs the behavioral 2NW/6NR
// sram_macro golden (docs/UB_REARCHITECTURE.md section 6).
//
// Identical stimulus to both storage blocks; rd_data compared every
// cycle. Stimulus respects the banked-UB contract (at most one read /
// one write per bank per cycle, no same-address R+W) — the same contract
// the UB_TRACE diagnostics verified for every gated inference program.
// Directed phases cover affine read walks (strides 0/1/3 at N=4),
// same-address multi-read, same-address write priority, and 1R+1W
// same-bank traffic; then 2000 cycles of constrained-random traffic.

module test_ub_banked_equiv;

    parameter int N = 4;
    parameter int DEPTH = 64;
    parameter int WIDTH = 16;
    localparam int NUM_WRITE = 2*N;
    localparam int NUM_READ = 6*N;

    logic clk = 0;
    always #5 clk = ~clk;
    logic rst = 1;

    logic [NUM_WRITE-1:0]      wr_en;
    logic [NUM_WRITE*16-1:0]   wr_addr;
    logic [NUM_WRITE*WIDTH-1:0] wr_data;
    logic [NUM_READ*16-1:0]    rd_addr;
    logic [NUM_READ-1:0]       rd_act;
    logic [NUM_READ*WIDTH-1:0] rd_data_golden;
    logic [NUM_READ*WIDTH-1:0] rd_data_banked;

    sram_macro #(
        .WIDTH(WIDTH),
        .DEPTH(DEPTH),
        .NUM_WRITE(NUM_WRITE),
        .NUM_READ(NUM_READ),
        .CLEAR_ON_RESET(1)
    ) golden (
        .clk(clk),
        .rst(rst),
        .wr_en(wr_en),
        .wr_addr(wr_addr),
        .wr_data(wr_data),
        .rd_addr(rd_addr),
        .rd_data(rd_data_golden)
    );

    ub_banked #(
        .WIDTH(WIDTH),
        .DEPTH(DEPTH),
        .N(N)
    ) banked (
        .clk(clk),
        .rst(rst),
        .wr_en(wr_en),
        .wr_addr(wr_addr),
        .wr_data(wr_data),
        .rd_addr(rd_addr),
        .rd_data(rd_data_banked),
        .rd_act(rd_act)
    );

    int errors = 0;
    int cycle = 0;
    logic [NUM_READ-1:0] rd_act_d = '0;

    // Lockstep compare. Stimulus drives new beat addresses at posedge+#2
    // and the compare runs at posedge+#3 — AFTER the next beat's addresses
    // are already on rd_addr. This reproduces the real UB's timing (the
    // comb walk advances addresses on the same edge that registers the
    // read data), so a comb-select/data skew like the unregistered
    // bank-select bug cannot hide: only a storage block whose outputs are
    // aligned to the REQUEST cycle (like sram_macro's per-port registered
    // read) passes. rd_act_d (sampled at the posedge, pre-drive) masks the
    // compare to active-window ports — inactive ports present address 0;
    // golden returns mem[0], banked returns the bank's shared output,
    // unobservable in the real design by construction.
    always @(posedge clk) begin
        cycle = cycle + 1;
        rd_act_d <= rd_act;
        #3;
        if (!rst) begin
            for (int p = 0; p < NUM_READ; p++) begin
                if (rd_act_d[p] && rd_data_golden[p*WIDTH +: WIDTH]
                                  !== rd_data_banked[p*WIDTH +: WIDTH]) begin
                    errors = errors + 1;
                    if (errors <= 5) begin
                        $display("ERROR: port %0d mismatch cyc=%0d golden=%h banked=%h",
                                 p, cycle,
                                 rd_data_golden[p*WIDTH +: WIDTH],
                                 rd_data_banked[p*WIDTH +: WIDTH]);
                    end
                end
            end
        end
    end

    task automatic clear_inputs;
        wr_en = '0;
        wr_addr = '0;
        wr_data = '0;
        rd_addr = '0;
        rd_act = '0;
    endtask

    task automatic write_one(input int port, input int addr, input int data);
        wr_en[port] = 1'b1;
        wr_addr[port*16 +: 16] = addr;
        wr_data[port*WIDTH +: WIDTH] = data;
    endtask

    task automatic read_one(input int port, input int addr);
        rd_act[port] = 1'b1;
        rd_addr[port*16 +: 16] = addr;
    endtask

    int a, g, s, t, p, w;
    int wb_mask, rb_mask, nw;
    int waddrs [NUM_WRITE];

    initial begin
        $display("=== UB banked equivalence test (N=%0d, DEPTH=%0d) ===", N, DEPTH);
        clear_inputs();
        rst = 1;
        repeat (4) @(posedge clk); #2;
        rst = 0;

        // Phase 1: sequential fill, one channel (N writes/cycle,
        // consecutive addresses = distinct banks).
        $display("Phase 1: sequential fill");
        for (t = 0; t < DEPTH/N; t++) begin
            @(posedge clk); #2;
            clear_inputs();
            for (w = 0; w < N; w++) begin
                write_one(w, t*N + w, 16'h1000 + t*N + w);
            end
        end
        @(posedge clk); #2;
        clear_inputs();

        // Phase 2: affine read walks per group, strides {0, 1, 3}.
        // s=0 exercises same-address multi-read sharing one bank.
        $display("Phase 2: affine read walks");
        for (g = 0; g < 6; g++) begin
            for (s = 0; s < 4; s = s + 1) begin
                if (s != 2) begin  // stride 2 collides at N=4 (out of contract)
                    for (t = 0; t < 8; t++) begin
                        @(posedge clk); #2;
                        clear_inputs();
                        for (p = 0; p < N; p++) begin
                            read_one(g*N + p, (t*7 + g*3 + p*s) & (DEPTH-1));
                        end
                    end
                end
            end
        end
        @(posedge clk); #2;
        clear_inputs();

        // Phase 3: same-address write priority (low port index wins,
        // across channels too). Readback addresses on distinct banks.
        $display("Phase 3: same-address write priority");
        @(posedge clk); #2;
        clear_inputs();
        write_one(1, 20, 16'hAAAA);
        write_one(6, 20, 16'hBBBB);
        @(posedge clk); #2;
        clear_inputs();
        write_one(0, 21, 16'hCCCC);
        write_one(3, 21, 16'hDDDD);
        @(posedge clk); #2;
        clear_inputs();
        read_one(0, 20);
        read_one(1, 21);
        @(posedge clk); #2;
        clear_inputs();

        // Phase 4: 1R+1W same bank, different address, every cycle;
        // second read on a third bank.
        $display("Phase 4: same-bank read+write");
        for (t = 0; t < 12; t++) begin
            @(posedge clk); #2;
            clear_inputs();
            write_one(0, t, 16'h2000 + t);
            read_one(0, (t + N) & (DEPTH-1));    // same bank as the write
            read_one(1, (t + N + 1) & (DEPTH-1)); // different bank
        end
        @(posedge clk); #2;
        clear_inputs();

        // Phase 5: constrained-random traffic, contract-respecting.
        $display("Phase 5: 2000 random cycles");
        for (t = 0; t < 2000; t++) begin
            @(posedge clk); #2;
            clear_inputs();
            wb_mask = 0;
            rb_mask = 0;
            nw = 0;
            for (w = 0; w < NUM_WRITE; w++) begin
                if (($random & 3) == 0) begin
                    a = $random & (DEPTH-1);
                    if (!((wb_mask >> (a & (N-1))) & 1)) begin
                        write_one(w, a, $random & 16'hFFFF);
                        wb_mask = wb_mask | (1 << (a & (N-1)));
                        waddrs[nw] = a;
                        nw = nw + 1;
                    end
                end
            end
            for (p = 0; p < NUM_READ; p++) begin
                if (($random & 3) != 0) begin
                    a = $random & (DEPTH-1);
                    // same-address R+W is out of contract: shift by N
                    // (same bank, different address) if we hit one.
                    for (w = 0; w < nw; w++) begin
                        if (a == waddrs[w]) a = (a + N) & (DEPTH-1);
                    end
                    if (!((rb_mask >> (a & (N-1))) & 1)) begin
                        read_one(p, a);
                        rb_mask = rb_mask | (1 << (a & (N-1)));
                    end
                end
            end
        end

        repeat (4) @(posedge clk); #2;
        clear_inputs();
        repeat (4) @(posedge clk);

        $display("\n=== Test Summary ===");
        if (errors == 0) begin
            $display("ALL TESTS PASSED");
        end else begin
            $display("%0d ERRORS", errors);
        end
        $finish;
    end

endmodule

`default_nettype wire
