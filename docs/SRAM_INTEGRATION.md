# SRAM Macro Integration for tinyTPU

## Overview

This document describes the SRAM macro integration path for the tinyTPU design, replacing the DFF-based unified buffer (UB) and program buffer with area-efficient SRAM macros.

## Area Analysis

### Current DFF-Based Storage

At N=16 (item 21c configuration):

**Unified Buffer (UB)**
- Location: `src/unified_buffer_nxn.sv` line 78
- Declaration: `logic [15:0] ub_memory [0:UNIFIED_BUFFER_WIDTH-1]`
- Parameters: UNIFIED_BUFFER_WIDTH = 32768
- Total: 32768 × 16 bits = **524,288 DFFs**
- Transistor count: 524,288 × 40T/DFF = **20.97M transistors**

**Program Buffer**
- Location: `src/instr_seq_nxn.sv` line 131
- Declaration: `logic [PW-1:0] prog_mem [PROG_DEPTH]`
- Parameters: PW = 373 bits (372-bit instruction + 1 ctrl bit), PROG_DEPTH = 8192
- Total: 8192 × 373 bits = **3,055,616 DFFs**
- Transistor count: 3,055,616 × 40T/DFF = **122.2M transistors**

**Total DFF area**: 143.2M transistors

### SRAM-Based Storage

**Unified Buffer (UB)**
- Total: 32768 × 16 bits = **524,288 bits**
- Transistor count: 524,288 × 6T/bit = **3.15M transistors**
- Area reduction: **6.7x**

**Program Buffer**
- Total: 8192 × 373 bits = **3,055,616 bits**
- Transistor count: 3,055,616 × 6T/bit = **18.3M transistors**
- Area reduction: **6.7x**

**Total SRAM area**: 21.5M transistors

### Area Savings

- **Total area reduction**: 143.2M → 21.5M transistors
- **Savings**: 121.7M transistors (85% reduction)
- **Area ratio**: 6.7x smaller with SRAM

## SRAM Macro Model

A behavioral SRAM model is provided in `src/sram_macro.sv`:

```systemverilog
module sram_macro #(
    parameter int WIDTH = 16,
    parameter int DEPTH = 1024,
    parameter int NUM_WRITE = 3,
    parameter int NUM_READ = 7
)(
    input logic clk,
    input logic [NUM_WRITE-1:0] wr_en,
    input logic [NUM_WRITE*16-1:0] wr_addr,
    input logic [NUM_WRITE*WIDTH-1:0] wr_data,
    input logic [NUM_READ*16-1:0] rd_addr,
    output logic [NUM_READ*WIDTH-1:0] rd_data
);
```

Features:
- Synchronous read (1-cycle latency)
- Synchronous write
- Priority arbitration (port 0 > port 1 > ... > port N-1)
- Parameterized width, depth, and port count
- Packed array interface (iverilog compatible)

Test: `test/test_sram_macro.sv` (all tests pass)

## Integration Strategy

### Unified Buffer

The UB requires:
- 3N write ports (N VPU + N host + N gradient descent)
- 7N read ports (N input + N weight + N bias + N Y + N H + N grad_bias + N grad_weight)

For N=16: **48 write ports, 112 read ports**

#### Option 1: Single Large Multi-Port SRAM
- Instantiate one SRAM with 48 write + 112 read ports
- Pros: Simple, matches current architecture
- Cons: Very large SRAM, complex routing

#### Option 2: Banked SRAM
- Split memory into multiple banks
- Each bank has fewer ports
- Pros: More realistic, better routing
- Cons: Requires address decoding, bank conflict handling

#### Option 3: Time-Multiplexing
- Use smaller SRAMs with fewer ports
- Share ports across cycles
- Pros: Smaller area, simpler SRAMs
- Cons: Adds latency, requires pipeline changes

### Program Buffer

The program buffer requires:
- 1 write port (host load)
- 1 read port (instruction fetch)

For N=16: **1 write port, 1 read port**

This is a simple 1R1W SRAM, straightforward to implement.

## Synthesis Path

For synthesis, replace the behavioral `sram_macro` with a foundry SRAM macro:

### sky130 Example
```systemverilog
// Replace behavioral model with sky130 SRAM
sky130_sram_1rw_32x1024 sram_inst (
    .clk(clk),
    .wen(wr_en),
    .waddr(wr_addr),
    .wdata(wr_data),
    .raddr(rd_addr),
    .rdata(rd_data)
);
```

### Multi-Port SRAM Generation
For multi-port SRAMs, use:
- **OpenRAM**: Open-source SRAM compiler (generates multi-port SRAMs)
- **Foundry macros**: Commercial SRAM libraries (e.g., TSMC, GlobalFoundries)
- **FPGA blocks**: For FPGA prototyping (BRAM, URAM)

## Current Status

- [x] Behavioral SRAM model created (`src/sram_macro.sv`)
- [x] Test suite created and passing (`test/test_sram_macro.sv`)
- [x] Area analysis documented
- [ ] Integrate SRAM into unified_buffer_nxn.sv
- [ ] Integrate SRAM into instr_seq_nxn.sv
- [ ] Synthesis validation with OpenLane
- [ ] Area comparison post-synthesis

## Next Steps

1. **Program Buffer Integration** (simpler, 1R1W)
   - Replace `prog_mem` with SRAM instantiation
   - Verify instruction fetch timing
   - Run existing tests (test_instr_seq_nxn_*)

2. **Unified Buffer Integration** (complex, multi-port)
   - Choose integration strategy (single/banked/mux)
   - Implement address generation logic
   - Verify read/write timing
   - Run existing tests (test_unified_buffer_nxn_*)

3. **Synthesis Validation**
   - Generate SRAM macros with OpenRAM
   - Run OpenLane flow
   - Compare area/power with DFF-based version

## References

- SRAM behavioral model: `src/sram_macro.sv`
- SRAM test: `test/test_sram_macro.sv`
- Unified buffer: `src/unified_buffer_nxn.sv`
- Program buffer: `src/instr_seq_nxn.sv`
- OpenRAM: https://github.com/VLSIDA/OpenRAM
- sky130 SRAM: https://github.com/google/skywater-pdk
