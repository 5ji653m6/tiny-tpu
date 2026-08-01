"""Gate test for item 22: K-tiling via radd — a 16x16 C = A @ B matmul
with K=32 (two K-tiles of 16x16 each) run as ONE loaded program that
reuses the item-17a residual-add pathway for tile accumulation. Written
by the harness author, not the agent — per tinytpu-loop README "the
gate grows with the design".

Item 21c proved the DiT dataflow scales cleanly from N=4 through N=16.
Item 22 extends the chip's matrix-size support beyond K=N via tiling:
a matmul with K > N is broken into K/N tiles of size NxN, with each
tile's partial product accumulated via the ptr-7 residual-add pathway.

Choreography (no RTL changes):
  - Tile 0 (bypass phase): C_0 = A_0 @ B_0, appended at wr_ptr (= addr_C
    by construction — the host preloads ONLY the input words)
  - Tile 1 (radd phase): read C_0 via ptr-7 (PATHWAY_BIAS), compute
    A_1 @ B_1, VPU bias stage adds the residual, write the final
    C = C_0 + A_1 @ B_1 appended right after C_0.

The key insight: the existing radd pathway (item 17a, originally for
the DiT residual) is a general "read old value, add to new computation,
write back" primitive — perfect for tile accumulation. No new RTL, no
new pathway bit, no VPU changes.

READ-WALK API CONTRACT (the item-22 lesson, discovered the hard way):
the UB input read walk derives its storage row-stride from the read
command's col_size field — lane k's beat r reads address
addr + k + col_size*r. For a matrix stored row-major with true stride
S, correct data requires col_size == S. Every pre-item-22 test used
square NxN matrices, where stride == width == N, so the distinction was
invisible. A 16x16 tile read IN-PLACE from a 16x32 matrix (col_size=16,
true stride 32) silently reads the WRONG words — and the original
stimulus's mod-11 degeneracy (column-step x 16 == row-step mod 11)
made the misread rows emerge as pairwise-equal outputs, masquerading
as a "VPU holds each value 2 cycles" hardware bug. Hence the layout
below stores A TILE-MAJOR: each K-tile is a standalone 16x16 stride-16
block, so every read is a well-formed square read. (Equally, col_size
doubles as the output stream width via wr_stream_width — another
reason in-place sub-tile reads are not expressible.)

Layout (UB_WIDTH=2048):
  A_0 @0..255    (16x16 tile, A[:, 0:16],  row-major stride 16)
  A_1 @256..511  (16x16 tile, A[:, 16:32], row-major stride 16)
  B_0 @512..767  (16x16 tile, B[0:16, :])
  B_1 @768..1023 (16x16 tile, B[16:32, :])
  C_0 @1024..1279  (tile-0 partial product, written by the chip)
  C   @1280..1535  (final result, written by the chip)
  Host preloads only the first 1024 words (64 beats), so wr_ptr = 1024
  when tile 0 emits (streams append at wr_ptr; only W'/B' write back
  in place). Total image: 1536 words.

Program (320 words): 65-word host prefix (64 write beats + trailing
tick) + tile 0 bypass (100 words) + tile 1 radd (155 words).
PROG_DEPTH=512 suffices.

Golden: a TILED matmul_g (NOT direct matmul_g of the full 16x32 × 32x16
— that would have different rounding because per-MAC-step rounding
within a tile differs from a single stream over all K). Each tile uses
matmul_g on its 16x16 × 16x16 sub-problem; the two tile results are
summed with ew_add. This matches the hardware's actual computation
(bypass tile + radd tile).

Checks (LIVE asserts — PYTHONOPTIMIZE is empty):
  - per-lane VPU streams beat-exact: lane j collects C column j
    (16 beats for tile 0, 16 beats for tile 1 — total 32/lane);
  - the full 1536-word UB image exact everywhere
    (A_0, A_1, B_0, B_1, C_0, C).
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from test_tpu_nxn_prog_n4 import ProgGen, to_fixed, from_fixed
from test_tpu_nxn_prog_mh_attn_n4 import matmul_g, to_raw
from test_tpu_nxn_prog_radd_n4 import ew_add

N = int(os.environ.get("TPU_NXN_PROG_N", "16"))
WORD_W = 134 + 17 * (N - 2)
CTRL = 1 << WORD_W

PATHWAY_BYPASS = 0b0000000
PATHWAY_BIAS = 0b0001000
PTR_RESIDUAL = 7

K_TILES = 2                  # K = 2N = 32 at N=16
M = N                        # 16
K = N * K_TILES              # 32

# ---- stimulus: deterministic, distinct per cell, exact in Q8.8 ----
# (A has 32 columns with coprime moduli per column; B has 32 rows with
# coprime moduli per row. Each 16x16 tile is full rank — a column- or
# row-constant tile would collapse partial products and hide
# accumulation bugs)
A = [[((r * 3 + c * 5 + 1) % 11 - 5) * 0.25 for c in range(K)]
     for r in range(M)]
B = [[((r * 7 + c * 3 + 2) % 13 - 6) * 0.125 for c in range(N)]
     for r in range(K)]

Ar = [[to_raw(v) for v in row] for row in A]
Br = [[to_raw(v) for v in row] for row in B]

# Tile sub-matrices
A0 = [row[:N] for row in Ar]       # 16x16 (A cols 0..15)
A1 = [row[N:] for row in Ar]       # 16x16 (A cols 16..31)
B0 = Br[:N]                         # 16x16 (B rows 0..15)
B1 = Br[N:]                         # 16x16 (B rows 16..31)

# Per-tile partial products (each with its own per-MAC rounding)
C0 = matmul_g(A0, B0)               # 16x16
C1 = matmul_g(A1, B1)               # 16x16

# The TILED golden: sum of partial products via ew_add (this matches
# the hardware, which adds via the ptr-7 residual pathway)
C = ew_add(C0, C1)                  # 16x16

ADDR_A0 = 0
ADDR_A1 = N * N                     # 256
ADDR_B0 = ADDR_A1 + N * N           # 512
ADDR_B1 = ADDR_B0 + N * N           # 768
ADDR_C = ADDR_B1 + N * N            # 1024 (tile-0 emission lands here)
HOST_WORDS = ADDR_C                 # 1024: host preloads inputs only
IMG_WORDS = ADDR_C + 2 * N * N      # 1536 (C_0 then final C appended)

GOLD = {a: 0 for a in range(IMG_WORDS)}
for r in range(M):
    for c in range(N):
        GOLD[ADDR_A0 + N * r + c] = A0[r][c] & 0xFFFF
        GOLD[ADDR_A1 + N * r + c] = A1[r][c] & 0xFFFF
for r in range(K):
    for c in range(N):
        if r < N:
            GOLD[ADDR_B0 + N * r + c] = Br[r][c] & 0xFFFF
        else:
            GOLD[ADDR_B1 + N * (r - N) + c] = Br[r][c] & 0xFFFF
for r in range(M):
    for c in range(N):
        GOLD[ADDR_C + N * r + c] = C0[r][c] & 0xFFFF
        GOLD[ADDR_C + N * N + N * r + c] = C[r][c] & 0xFFFF


def generate_program_ktil_n16():
    """320 words: 65-word host prefix (64 write beats + trailing tick)
    + tile 0 bypass (100 words) + tile 1 radd (155 words)."""
    g = ProgGen()
    g.idle()
    host_words = [GOLD[a] for a in range(HOST_WORDS)]
    # All GOLD entries are already in fixed-point (to_raw values stored
    # as unsigned 16-bit); write them directly.
    assert len(host_words) == HOST_WORDS and len(host_words) % N == 0
    for b in range(len(host_words) // N):
        g.write_beat([host_words[N * b + (N - 1 - i)] for i in range(N)])
    g.tick()                              # trailing idle
    _prefix = HOST_WORDS // N + 1         # 65 (idle + trailing tick merge)
    assert len(g.prog) == _prefix, \
        f"prefix is {len(g.prog)} words, expected {_prefix}"

    # ---- tile 0: bypass phase, C_0 = A_0 @ B_0 -> appends @1024 ----
    g.issue_read(ptr=1, addr=ADDR_B0, rows=N, cols=N, transpose=0)
    g.tick(2 * N + 1)                     # 33: weight preload
    g.issue_read(ptr=0, addr=ADDR_A0, rows=M, cols=N, transpose=0,
                 pathway=PATHWAY_BYPASS)
    g.switch_pulse()
    g.tick(64)                            # emission wait (matches n16 test)
    assert len(g.prog) == _prefix + (1 + (2*N+1) + 1 + 1 + 64), \
        f"after tile 0: {len(g.prog)} words"

    # ---- tile 1: radd phase, C = A_1 @ B_1 + read(C_0) -> @1280 ----
    g.issue_read(ptr=1, addr=ADDR_B1, rows=N, cols=N, transpose=0)
    g.tick(2 * N + 1)                     # 33: weight preload
    g.issue_read(ptr=0, addr=ADDR_A1, rows=M, cols=N, transpose=0,
                 pathway=PATHWAY_BIAS)
    g.switch_pulse()
    g.tick(N - 2)                         # 14: mid-phase wait
    g.issue_read(ptr=PTR_RESIDUAL, addr=ADDR_C, rows=M, cols=N)
    g.tick(13 * N // 2)                   # 104: drain
    _total = (_prefix + (1 + (2*N+1) + 1 + 1 + 64)
              + (1 + (2*N+1) + 1 + 1 + (N-2) + 1 + 13*N//2))
    assert len(g.prog) == _total, \
        f"program is {len(g.prog)} words, expected {_total}"
    return g.prog


async def tick(dut, cycles=1):
    """Edge + 1ns settle: reads see post-edge values, drives land
    mid-cycle."""
    for _ in range(cycles):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")


@cocotb.test()
async def test_tpu_nxn_prog_ktil_n16(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    prog = generate_program_ktil_n16()
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
    await tick(dut, len(prog) + 200)
    collector.kill()

    # ---- per-lane streams: lane j collects C column j from BOTH tiles ----
    # Tile 0 writes 16 beats (C_0 column j), tile 1 writes 16 beats
    # (final C column j = C_0 column j + A_1 @ B_1 contribution). Total
    # 32 beats per lane.
    for j in range(N):
        expected = [C0[r][j] & 0xFFFF for r in range(M)] \
                 + [C[r][j] & 0xFFFF for r in range(M)]
        assert lanes[j] == expected, (
            f"VPU lane {j}: got "
            f"{[f'{from_fixed(w):+.4f}' for w in lanes[j]]}, expected "
            f"{[f'{from_fixed(w):+.4f}' for w in expected]} "
            f"(C_0 column {j} then C column {j})")

    # ---- final UB image: all 1536 words exact ----
    for a in range(IMG_WORDS):
        got = nxn.ub_inst.ub_sram.mem[a].value.integer & 0xFFFF
        want = GOLD[a] & 0xFFFF
        assert got == want, (
            f"mem[{a}] = {from_fixed(got):+.4f}, expected "
            f"{from_fixed(want):+.4f}")

    print(f"tpu_nxn_prog N={N} K-tiling via radd OK "
          f"({len(prog)}-word program: 1 bypass + 1 radd phase; "
          f"16x32 × 32x16 matmul as two 16x16 tile partial products "
          f"accumulated via the ptr-7 residual-add pathway; "
          f"{IMG_WORDS}-word UB image exact)")
