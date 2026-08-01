"""Gate test for roadmap item 16 (harness-only, no new RTL): TILED
MATMUL via the item-15 indexed LOOP. Written by the harness author,
not the agent — per tinytpu-loop README "the gate grows with the
design".

C = A(4x4) @ B(4x12) computed as THREE column tiles of 4 from ONE
looped 52-word body — the output-column dimension exceeds the array
width N=4, so the body sweeps Kx4 weight tiles:

  iteration 0: W read @16 (tile T0), A read @0  -> C0 = A@T0 appends @64
  iteration 1: W read @32 (tile T1), A read @0  -> C1 = A@T1 appends @80
  iteration 2: W read @48 (tile T2), A read @0  -> C2 = A@T2 appends @96

The LOOPI control word is (count=3, len=52, stride_a=0, stride_w=16):
the ptr-1 weight read advances one 16-word tile per iteration while the
ptr-0 activation read is STATIONARY (stride 0 => addr + i*0 = unchanged
— the "activation-stationary re-read" half of the tiling pattern, the
complement of item 15's both-strides-advance). Because VPU streams only
append at the UB write pointer (items 9/14), the three 4x4 output tiles
land CONTIGUOUSLY at @64/@80/@96 = a blocked-column layout of the 4x12
C matrix; downstream consumers read a tile at a time.

K-dim tiling (K > N) is NOT this item: it needs cross-tile accumulation
(a VPU add/accumulate stage — the residual-add gap listed under item
17). Column tiling needs no adds, so item 16 is pure program
composition on top of item 15's RTL.

Host image: A @0 (16 words) + T0 @16 + T1 @32 + T2 @48 (row-major
per tile) = 64 words. Program = 17-word prefix (16 host beats +
trailing tick; ProgGen's leading idle primes but never commits) +
LOOPI + 52-word body = 70 words (vs 17 + 3x52 = 173 straight-line).
Default PROG_DEPTH=256 and UB_WIDTH=128 suffice (70 words, 112-word
image).

Golden: matmul_g from the item-14 test (per-step fxp rounding, folded
in k order) against raw Q8.8 stimulus; the tiles are deliberately
distinct so a stride failure (re-reading T0) mismatches loudly, and
eighth-multiple entries exercise the +0x80 product rounding.

Checks (LIVE asserts — PYTHONOPTIMIZE is empty):
  - per-lane VPU streams beat-exact: lane j captures C0[:,j] then
    C1[:,j] then C2[:,j] (12 beats per lane, all four lanes);
  - UB image words 0..111 exact: the 64-word host image plus the three
    output tiles — C1/C2's presence and values prove the weight read
    advanced 16 words per iteration while activations re-read @0.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# ProgGen + Q8.8 helpers from the 9b test; the per-step-rounding matmul
# golden and the signed-raw converter from the item-14 test (to_fixed
# returns the 16-bit two's-complement ENCODING — golden math needs
# to_raw's signed ints).
from test_tpu_nxn_prog_n4 import ProgGen, to_fixed, from_fixed
from test_tpu_nxn_prog_mh_attn_n4 import matmul_g, to_raw

N = int(os.environ.get("TPU_NXN_PROG_N", "4"))
WORD_W = 134 + 17 * (N - 2)   # instruction width (item 13: SiLU bit)
CTRL = 1 << WORD_W            # control-word escape bit (prog-word MSB)
IDX_FLAG = 1 << 16

TILE_COUNT = 3
BODY_LEN = 52                 # 1 + 9 + 1 + 1 + 40
STRIDE_W = 16                 # one 4x4 weight tile per iteration
STRIDE_A = 0                  # activations re-read @0 every iteration

# Stimulus: quarter/eighth multiples, tiles deliberately distinct,
# |dot| <= 4*1.25*1.25 = 6.25 (no clamp); eighth multiples exercise the
# product rounding.
A = [[ 0.50, -0.375,  1.00,  0.75],
     [-0.625,  1.25, -0.50,  0.25],
     [ 0.25,  0.75, -0.875,  0.50],
     [ 1.00, -0.50,  0.25, -0.125]]
T0 = [[ 0.25,  0.50, -0.625,  1.00],
      [ 1.00, -0.25,  0.50,  0.75],
      [-0.50,  0.75,  1.25, -0.25],
      [ 0.75,  1.00, -0.50,  0.25]]
T1 = [[-0.25,  1.00,  0.75, -0.50],
      [ 0.50, -0.875, 1.00,  0.25],
      [ 1.25,  0.25, -0.25,  0.75],
      [ 0.75, -0.50,  0.50,  1.00]]
T2 = [[ 1.00,  0.25, -0.50,  0.75],
      [-0.75,  0.50,  1.00, -0.25],
      [ 0.50,  1.25,  0.25, -0.375],
      [ 0.125, -0.75, 0.75,  0.50]]
TILES = [T0, T1, T2]

A_raw = [[to_raw(v) for v in row] for row in A]
T_raw = [[[to_raw(v) for v in row] for row in T] for T in TILES]
C_gold = [matmul_g(A_raw, T) for T in T_raw]   # signed raw Q8.8, 4x4

# UB image: host (A + tiles, row-major) then C tiles appended @64.
# Signed raws; the image check masks & 0xFFFF at compare time.
GOLD = [v for row in A_raw for v in row]
for T in T_raw:
    GOLD += [v for row in T for v in row]
assert len(GOLD) == 64
for Ct in C_gold:
    GOLD += [v for row in Ct for v in row]
assert len(GOLD) == 112


def loopi_word(count, length, stride_a, stride_w):
    return (CTRL | (count << 8) | length | IDX_FLAG
            | (stride_a << 24) | (stride_w << 40))


def generate_program_tiled():
    """70 words: 17-word host prefix + LOOPI(3, 52, sa=0, sw=16) + the
    52-word tiled-matmul body (weight read @16 T=0 -> tick(9) ->
    activation read @0 bypass -> switch -> tick(40))."""
    g = ProgGen()
    g.idle()
    # Host-write words carry the 16-bit two's-complement ENCODING
    # (to_fixed) — the prog-word packing must never see a negative int.
    host_words = ([to_fixed(v) for row in A for v in row]
                  + [to_fixed(v) for T in TILES for row in T for v in row])
    assert len(host_words) == 64 and len(host_words) % N == 0
    for b in range(len(host_words) // N):
        g.write_beat([host_words[N * b + (N - 1 - i)] for i in range(N)])
    g.tick()  # trailing idle cycle after the last beat
    assert len(g.prog) == 17, f"prefix is {len(g.prog)} words, expected 17"

    g.prog.append(loopi_word(TILE_COUNT, BODY_LEN, STRIDE_A, STRIDE_W))
    g.issue_read(ptr=1, addr=16, rows=4, cols=4, transpose=0)
    g.tick(9)
    g.issue_read(ptr=0, addr=0, rows=4, cols=4, transpose=0, pathway=0)
    g.switch_pulse()
    g.tick(40)

    return g.prog


async def tick(dut, cycles=1):
    """Edge + 1ns settle: reads see post-edge values, drives land
    mid-cycle."""
    for _ in range(cycles):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")


@cocotb.test()
async def test_tpu_nxn_prog_tiled_n4(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    prog = generate_program_tiled()
    assert len(prog) == 70, f"program is {len(prog)} words, expected 70"
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

    # Release reset and run — ONE pulse; the LOOPI word does the rest.
    dut.rst.value = 0
    await tick(dut)
    dut.run.value = 1
    await tick(dut)
    dut.run.value = 0

    # Emission: 70 loaded words + 2 extra body passes (104) = 174
    # cycles; the last body's 40 idles cover the final C tile stream.
    await tick(dut, 174 + 20)
    collector.kill()

    # ---- per-lane streams, beat-exact through C2 ----
    def col(M, j, rows):
        return [M[r][j] & 0xFFFF for r in range(rows)]

    for j in range(N):
        expected = []
        for Ct in C_gold:
            expected += col(Ct, j, 4)
        assert lanes[j] == expected, (
            f"VPU lane {j}: {len(lanes[j])} beats, expected "
            f"{len(expected)}; got "
            f"{[f'{from_fixed(w):+.4f}' for w in lanes[j][:8]]}..., "
            f"expected "
            f"{[f'{from_fixed(w):+.4f}' for w in expected[:8]]}...")

    # ---- UB image words 0..111 exact (host + three C tiles) ----
    for a in range(112):
        got = nxn.ub_inst.ub_sram.mem[a].value.integer & 0xFFFF
        want = GOLD[a] & 0xFFFF
        assert got == want, (
            f"mem[{a}] = {from_fixed(got):+.4f}, expected "
            f"{from_fixed(want):+.4f}")

    print(f"tpu_nxn_prog N={N} TILED matmul via indexed LOOP OK "
          f"(70-word program: ONE 52-word body looped 3x, weight read "
          f"advancing 16 words/tile, activations stationary @0 — the "
          f"4x12 C lands blocked-column @64/@80/@96; straight-line "
          f"would take 173 words)")
