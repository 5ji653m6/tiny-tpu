"""Gate test for array scaling item 21b: the chip executing a LOADED
PROGRAM at N=16 — a 16x16 matmul C = X @ W streamed by the instruction
sequencer. Written by the harness author, not the agent — per
tinytpu-loop README "the gate grows with the design".

Item 21a proved the N×N parameterization holds at N=16 for the unit-
level modules (five N-generic tests pass at N=16 on first probe, zero
RTL changes). 21b extends that to the full program-buffer datapath:
host image load, weight preload, activation streaming, VPU emission
and UB writeback — all at N=16 with the 372-bit instruction word
(134+17*(16-2)).

The choreography is the single-phase matmul from the N=4/N=8 program
tests with N-scaled timing constants: the weight wait is 2N+1 cycles
(the last weight column walks in after N beats + N-1 skew + 1) = 33
at N=16, and the emission wait is a generous 64 (it is the final
phase; extra idle words are harmless padding).

Layout (UB_WIDTH=1024):
  X @0..255 (16x16), W @256..511 (16x16), C @512..767 (stream append
  at wr_ptr after the 512-word prefix).

Program: 33-word prefix (32 write beats + trailing tick; the leading
idle primes but never commits) + weight read + tick(33) + activation
read + switch + tick(64) = 133 words; default PROG_DEPTH=256
suffices.

Golden: matmul_g (per-MAC-step rounding, hardware exact) on raw Q8.8
values. Stimulus is eighth/quarter multiples — exact in Q8.8 — with
distinct per-cell values so an addressing mixup reads as a value
mismatch.

Checks (LIVE asserts — PYTHONOPTIMIZE is empty):
  - per-lane VPU streams beat-exact: lane j collects C column j
    (16 beats);
  - the full 768-word UB image exact everywhere.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from test_tpu_nxn_prog_n4 import ProgGen, to_fixed, from_fixed
from test_tpu_nxn_prog_mh_attn_n4 import matmul_g, to_raw

N = int(os.environ.get("TPU_NXN_PROG_N", "16"))

# ---- stimulus: deterministic, distinct per cell, exact in Q8.8 ----
# (formulas chosen so W is NOT column-constant and no X row sums to
# zero — a rank-1 stimulus makes C rows collapse to zero and hides
# addressing bugs)
X = [[((r * 3 + c * 5 + 1) % 9 - 4) * 0.25 for c in range(16)]
     for r in range(16)]
W = [[((r * 3 + c * 2) % 7 - 3) * 0.125 for c in range(16)]
     for r in range(16)]

Xr = [[to_raw(v) for v in row] for row in X]
Wr = [[to_raw(v) for v in row] for row in W]
C = matmul_g(Xr, Wr)          # hardware-exact 16x16 golden

ADDR_X = 0
ADDR_W = 256
ADDR_C = 512
IMG_WORDS = 768

GOLD = {a: 0 for a in range(IMG_WORDS)}
for r in range(16):
    for c in range(16):
        GOLD[ADDR_X + 16 * r + c] = Xr[r][c]
        GOLD[ADDR_W + 16 * r + c] = Wr[r][c]
        GOLD[ADDR_C + 16 * r + c] = C[r][c]


def generate_program_n16():
    """133 words: 33-word host prefix (512 host words = 32 beats +
    trailing tick) + one matmul phase (weight read + tick(33) +
    activation read + switch + tick(64))."""
    g = ProgGen()
    g.idle()
    host_words = ([to_fixed(v) for row in X for v in row]
                  + [to_fixed(v) for row in W for v in row])
    assert len(host_words) == 512 and len(host_words) % N == 0
    for b in range(len(host_words) // N):
        g.write_beat([host_words[N * b + (N - 1 - i)] for i in range(N)])
    g.tick()  # trailing idle cycle after the last beat
    assert len(g.prog) == 33, \
        f"prefix is {len(g.prog)} words, expected 33"

    g.issue_read(ptr=1, addr=ADDR_W, rows=16, cols=16, transpose=0)
    g.tick(33)            # weight preload: 2N+1 at N=16
    g.issue_read(ptr=0, addr=ADDR_X, rows=16, cols=16, pathway=0)
    g.switch_pulse()
    g.tick(64)            # final phase: generous emission + writeback
    assert len(g.prog) == 133, \
        f"program is {len(g.prog)} words, expected 133"
    return g.prog


async def tick(dut, cycles=1):
    """Edge + 1ns settle: reads see post-edge values, drives land
    mid-cycle."""
    for _ in range(cycles):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")


@cocotb.test()
async def test_tpu_nxn_prog_n16(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    prog = generate_program_n16()
    nxn = dut.tpu_nxn_prog_inst.tpu_nxn_ic_inst.tpu_nxn_inst

    # Reset (the program load happens with the chip held in reset).
    dut.rst.value = 1
    dut.prog_wr_en.value = 0
    dut.prog_wr_data.value = 0
    dut.run.value = 0
    dut.learning_rate_in.value = to_fixed(0.5)  # inert: no lr_d stage
    await tick(dut, 2)

    # Load the program, one word per cycle.
    for w in prog:
        dut.prog_wr_data.value = w
        dut.prog_wr_en.value = 1
        await tick(dut)
    dut.prog_wr_en.value = 0
    dut.prog_wr_data.value = 0
    await tick(dut, 2)

    # Per-lane collectors on the VPU -> UB write interface.
    lanes = [[] for _ in range(N)]

    async def collect():
        while True:
            await RisingEdge(dut.clk)
            for i in range(N):
                if nxn.ub_wr_valid_in[i].value.integer:
                    lanes[i].append(nxn.ub_wr_data_in[i].value.integer
                                    & 0xFFFF)

    collector = cocotb.start_soon(collect())

    # Release reset and run the program.
    dut.rst.value = 0
    await tick(dut)
    dut.run.value = 1
    await tick(dut)
    dut.run.value = 0

    # The program self-paces: wait it out plus margin.
    await tick(dut, len(prog) + 20)
    collector.kill()

    # ---- per-lane streams: lane j collects C column j ----
    for j in range(N):
        expected = [C[r][j] & 0xFFFF for r in range(16)]
        assert lanes[j] == expected, (
            f"VPU lane {j}: got "
            f"{[f'{from_fixed(w):+.4f}' for w in lanes[j]]}, expected "
            f"{[f'{from_fixed(w):+.4f}' for w in expected]} "
            f"(C column {j})")

    # ---- final UB image: all 768 words exact ----
    for a in range(IMG_WORDS):
        got = nxn.ub_inst.ub_sram.mem[a].value.integer & 0xFFFF
        want = GOLD[a] & 0xFFFF
        assert got == want, (
            f"mem[{a}] = {from_fixed(got):+.4f}, expected "
            f"{from_fixed(want):+.4f}")

    print(f"tpu_nxn_prog N={N} loaded-program 16x16 matmul OK "
          f"(133-word program; 768-word UB image exact)")
