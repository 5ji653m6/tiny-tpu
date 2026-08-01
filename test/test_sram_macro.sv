`timescale 1ns/1ps
`default_nettype none

// Testbench for sram_macro behavioral model.
// Verifies basic read/write functionality and priority arbitration.

module test_sram_macro;

    parameter int WIDTH = 16;
    parameter int DEPTH = 64;
    parameter int NUM_WRITE = 3;
    parameter int NUM_READ = 7;

    logic clk = 0;
    always #5 clk = ~clk;  // 100MHz clock

    // Write port signals (packed arrays)
    logic [NUM_WRITE-1:0] wr_en;
    logic [NUM_WRITE*16-1:0] wr_addr;
    logic [NUM_WRITE*WIDTH-1:0] wr_data;

    // Read port signals (packed arrays)
    logic [NUM_READ*16-1:0] rd_addr;
    logic [NUM_READ*WIDTH-1:0] rd_data;

    // DUT instantiation
    sram_macro #(
        .WIDTH(WIDTH),
        .DEPTH(DEPTH),
        .NUM_WRITE(NUM_WRITE),
        .NUM_READ(NUM_READ)
    ) dut (
        .clk(clk),
        .wr_en(wr_en),
        .wr_addr(wr_addr),
        .wr_data(wr_data),
        .rd_addr(rd_addr),
        .rd_data(rd_data)
    );

    int errors = 0;

    initial begin
        $display("=== SRAM Macro Test ===");
        $display("WIDTH=%0d, DEPTH=%0d, NUM_WRITE=%0d, NUM_READ=%0d",
                 WIDTH, DEPTH, NUM_WRITE, NUM_READ);

        // Initialize
        wr_en = '0;
        wr_addr = '0;
        wr_data = '0;
        rd_addr = '0;

        // Wait for reset
        repeat (5) @(posedge clk);

        // Test 1: Single write + read
        $display("\nTest 1: Single write + read");
        wr_en[0] = 1'b1;
        wr_addr[0*16 +: 16] = 16'd10;
        wr_data[0*WIDTH +: WIDTH] = 16'hABCD;
        rd_addr[0*16 +: 16] = 16'd10;
        @(posedge clk);
        wr_en[0] = 1'b0;
        @(posedge clk);  // Wait for read latency
        @(posedge clk);  // Extra cycle
        @(posedge clk);
        if (rd_data[0*WIDTH +: WIDTH] !== 16'hABCD) begin
            $display("ERROR: Expected 0xABCD, got 0x%04h", rd_data[0*WIDTH +: WIDTH]);
            errors++;
        end else begin
            $display("PASS: Read 0x%04h from addr 10", rd_data[0*WIDTH +: WIDTH]);
        end

        // Test 2: Multiple writes to different addresses
        $display("\nTest 2: Multiple writes to different addresses");
        wr_en[0] = 1'b1;
        wr_addr[0*16 +: 16] = 16'd20;
        wr_data[0*WIDTH +: WIDTH] = 16'h1111;
        wr_en[1] = 1'b1;
        wr_addr[1*16 +: 16] = 16'd30;
        wr_data[1*WIDTH +: WIDTH] = 16'h2222;
        wr_en[2] = 1'b1;
        wr_addr[2*16 +: 16] = 16'd40;
        wr_data[2*WIDTH +: WIDTH] = 16'h3333;
        @(posedge clk);
        wr_en = '0;

        // Read back
        rd_addr[0*16 +: 16] = 16'd20;
        rd_addr[1*16 +: 16] = 16'd30;
        rd_addr[2*16 +: 16] = 16'd40;
        @(posedge clk);
        @(posedge clk);
        if (rd_data[0*WIDTH +: WIDTH] !== 16'h1111 || rd_data[1*WIDTH +: WIDTH] !== 16'h2222 || rd_data[2*WIDTH +: WIDTH] !== 16'h3333) begin
            $display("ERROR: Read mismatch");
            errors++;
        end else begin
            $display("PASS: Read 0x%04h, 0x%04h, 0x%04h", rd_data[0*WIDTH +: WIDTH], rd_data[1*WIDTH +: WIDTH], rd_data[2*WIDTH +: WIDTH]);
        end

        // Test 3: Write priority (port 0 wins)
        $display("\nTest 3: Write priority (port 0 wins)");
        wr_en[0] = 1'b1;
        wr_addr[0*16 +: 16] = 16'd50;
        wr_data[0*WIDTH +: WIDTH] = 16'hAAAA;
        wr_en[1] = 1'b1;
        wr_addr[1*16 +: 16] = 16'd50;  // Same address
        wr_data[1*WIDTH +: WIDTH] = 16'hBBBB;
        wr_en[2] = 1'b1;
        wr_addr[2*16 +: 16] = 16'd50;  // Same address
        wr_data[2*WIDTH +: WIDTH] = 16'hCCCC;
        @(posedge clk);
        wr_en = '0;

        rd_addr[0*16 +: 16] = 16'd50;
        @(posedge clk);
        @(posedge clk);
        if (rd_data[0*WIDTH +: WIDTH] !== 16'hAAAA) begin
            $display("ERROR: Port 0 should win, expected 0xAAAA, got 0x%04h", rd_data[0*WIDTH +: WIDTH]);
            errors++;
        end else begin
            $display("PASS: Port 0 priority works (0x%04h)", rd_data[0*WIDTH +: WIDTH]);
        end

        // Test 4: Multiple reads from same address
        $display("\nTest 4: Multiple reads from same address");
        rd_addr[0*16 +: 16] = 16'd50;
        rd_addr[1*16 +: 16] = 16'd50;
        rd_addr[2*16 +: 16] = 16'd50;
        @(posedge clk);
        @(posedge clk);
        if (rd_data[0*WIDTH +: WIDTH] !== 16'hAAAA || rd_data[1*WIDTH +: WIDTH] !== 16'hAAAA || rd_data[2*WIDTH +: WIDTH] !== 16'hAAAA) begin
            $display("ERROR: All reads should return 0xAAAA");
            errors++;
        end else begin
            $display("PASS: Multiple reads from same address work");
        end

        // Summary
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
