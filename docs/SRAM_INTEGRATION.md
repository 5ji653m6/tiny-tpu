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
    parameter int NUM_READ = 7,
    parameter int CLEAR_ON_RESET = 0  // behavioral-only array clear on rst
)(
    input logic clk,
    input logic rst,
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

### Unified Buffer — INTEGRATED (Option 1: single multi-port macro)

The UB nominally requires:
- 3N write ports (N VPU + N host + N gradient descent)
- 7N read ports (N input + N weight + N bias + N Y + N H + N grad_bias + N grad_weight)

For N=16: **48 write ports, 112 read ports** nominally.

**As integrated** (`src/unified_buffer_nxn.sv`), port collapsing brings this
to **2N write + 6N read** (32W/96R at N=16):

- Write: VPU-stream and host writes are mutually exclusive per lane (the
  DFF version's per-lane if/else-if), so they share N ports; gradient
  writeback keeps its own N ports at HIGHER priority (ports [0,N)),
  matching the DFF version's last-NBA-wins order. Total 2N.
- Read: ptr-5 (grad bias) and ptr-6 (grad weight) are mutually exclusive
  (the DFF version's if/else-if chain), so they share one N-lane group.
  Total 6N: input (ptr 0), weight (ptr 1), bias/residual/scale
  (ptr 2/7/8), Y (ptr 3), H (ptr 4), gradient (ptr 5/6).

**Timing is bit-identical to the DFF version — zero test changes.** The
SRAM's synchronous read (1-cycle latency) exactly replaces the DFF
version's per-lane output data registers: every channel's address walk is
computed combinationally from current state (verbatim the arithmetic the
DFF version ran inside always_ff) and presented to the SRAM this cycle;
the registered SRAM output is consumed next cycle alongside valid/window
registers updated on the same conditions as before. Writes were and stay
synchronous; read-during-write to the same address returns pre-edge (old)
data in both versions. Channels without a valid output (bias/Y/H/gradient)
gained per-lane registered window bits (`bias_win_r` etc.) that zero-mask
the returning data where the DFF version registered '0.

**Why not banking/time-multiplexing (Options 2/3)?** Analysis during
integration:

- All *write* patterns are stride-1 N-contiguous runs, so mod-N banking
  would be conflict-free for writes. Reads are the problem.
- Within a channel, per-lane read addresses are A + i·s with
  s ∈ {1, C−1, C+1, R'−1}. Over N = 2^k banks, even strides (odd matrix
  width C) collide, and no fixed power-of-2 bank count or XOR skew covers
  every stride in the set.
- **Cross-channel concurrency is the killer**: up to 4+ channels run
  simultaneously with independent bases (mid-phase operand reads overlap
  the draining output stream). A single odd-stride channel touches all N
  banks every cycle, so a second concurrent channel conflicts on every
  bank with certainty. Bank-conflict stalling would fire constantly, and
  stalling one channel breaks the systolic skew lockstep — a global UB
  stall would be needed, changing cycle-exact behavior for every consumer.
- Functional partitioning into per-channel SRAMs (the classic accelerator
  answer) is impossible without changing the software contract: the UB is
  one flat address space where any matrix can be read by any channel, and
  replication would multiply storage by the channel count, erasing the
  area win.

The UB was designed as a DFF array with no port discipline; the honest
SRAM mapping at RTL level is a full multi-port macro. See "Synthesis
Path" for the realism caveat.

**CLEAR_ON_RESET**: the DFF UB zeroed its array on rst, and gate suites
rely on reset-zeroed storage, so the UB instance sets
`sram_macro.CLEAR_ON_RESET=1` (behavioral-only; the program-buffer
instance leaves it 0). A foundry macro has no array reset — a silicon
swap needs a boot-time scrub pass.

**Test observability**: the full-chip gate suites read the buffer
hierarchically via VPI (cocotb) as `ub_inst.ub_sram.mem[a]` — pull-based
reads with zero simulation cost. An earlier revision kept the legacy
`ub_memory` name as a continuous-assign alias to `ub_sram.mem`; that made
every `mem` write wake all DEPTH alias functors in vvp
(`vvp_fun_arrayport_sa::check_word_change`), i.e. O(DEPTH²) work per
CLEAR_ON_RESET cycle at N=16 (DEPTH=32768): measured 0.24s vs 41s for a
60-cycle micro-benchmark (172x), multi-hour for the N=16 adaLN capstone.
The alias was removed and the tests now name the SRAM array directly.

#### Option 1: Single Large Multi-Port SRAM — CHOSEN
- Instantiate one SRAM with (collapsed) 2N write + 6N read ports
- Pros: bit-identical timing, zero consumer changes, zero test churn
- Cons: not directly buildable as a single foundry macro (see below)

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
- [x] Integrate SRAM into unified_buffer_nxn.sv (2NW/6NR macro, timing-identical, full gate sweep green)
- [x] Integrate SRAM into instr_seq_nxn.sv (1R1W prog_mem, 2-stage fetch/execute pipeline)
- [ ] Synthesis validation with OpenLane
- [ ] Area comparison post-synthesis

## Next Steps

1. **Synthesis Validation**
   - The behavioral multi-port UB macro (2NW/6NR) is not directly buildable:
     OpenRAM generates 1RW/1R1W macros, and foundry compilers top out at 2
     ports. A silicon mapping needs either a custom multi-port compiler, a
     banked redesign with a stall-tolerant datapath (see the banking
     analysis above for why that is an architecture change, not a module
     swap), or accepting register-file synthesis for the UB (yosys maps the
     behavioral model to FFs — functional but with no area win).
   - Replace `sram_macro` with generated/foundry macros where the port map
     allows (the 1R1W program buffer is directly swappable).
   - Run the OpenLane flow, compare area/power with the DFF-based version.
   - Boot-time UB scrub to replace CLEAR_ON_RESET with real macros.

## References

- SRAM behavioral model: `src/sram_macro.sv`
- SRAM test: `test/test_sram_macro.sv`
- Unified buffer: `src/unified_buffer_nxn.sv`
- Program buffer: `src/instr_seq_nxn.sv`
- OpenRAM: https://github.com/VLSIDA/OpenRAM
- sky130 SRAM: https://github.com/google/skywater-pdk
