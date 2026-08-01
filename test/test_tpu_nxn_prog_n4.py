"""Gate test for src/tpu_nxn_prog.sv (roadmap item 9b): the chip
executing a LOADED PROGRAM — an instruction sequencer replays a
program of 133+17*(N-2)-bit words into tpu_nxn_ic's instruction port.
Written by the harness author, not the agent — per tinytpu-loop README
"the gate grows with the design".

The program is generated OFFLINE (no RTL) by ProgGen, which reproduces
test_tpu_nxn_ic_train_n4's Host composition cycle by cycle: the exact
same training-step choreography, emitted as one instruction word per
cycle (phase-hold fields pathway/leak/inv_batch composed into every
word, host lanes k>=2 in the appended bits, one word per write beat /
issue_read / switch pulse, hold words through the waits). If the 8b
choreography ever changes, THIS GENERATOR MUST CHANGE WITH IT.

The test then: resets, loads the program one word per cycle
(prog_wr_en/prog_wr_data), pulses run, waits for the program plus
margin, and checks the SAME golden as the 8b test — per-lane dZ/G
streams beat-exact and the full 84-word UB image exact. The stream is
the entire spec: no field is ever poked directly.

Assertions are LIVE (PYTHONOPTIMIZE is empty).
"""

import os

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

N = int(os.environ.get("TPU_NXN_PROG_N", "4"))
FRAC = 8

# ---- stimulus + golden: identical to test_tpu_nxn_ic_train_n4 ----
X = np.array([[1.0, 0.5, 0.0, -0.5],
              [0.5, -1.0, 0.5, 1.0],
              [-0.5, 0.5, 1.0, 0.0],
              [0.0, 1.0, -0.5, 0.5]])
W = np.array([[0.5, -0.5, 1.0, 0.5],
              [1.0, 0.5, -0.5, 0.0],
              [0.0, 1.0, 0.5, -1.0],
              [-0.5, 0.0, 1.0, 0.5]])
B = np.array([0.5, -0.5, 0.5, -0.5])
Y = np.array([[1.0, -0.5, 0.5, 0.0],
              [0.5, 1.0, -0.5, 0.5],
              [-0.5, 0.0, 1.0, 0.5],
              [0.5, -1.0, 0.0, 1.0]])
LR = 0.5
LEAK = 0.5
INV2B = 0.5

Z = X @ W.T
Hpre = Z + B
H = np.where(Hpre >= 0, Hpre, Hpre * LEAK)
dZ = np.where(H >= 0, (H - Y) * INV2B, (H - Y) * INV2B * LEAK)
B_new = B - LR * dZ.sum(axis=0)
G = dZ.T @ X
W_new = W - LR * G

GOLD = {}
for r in range(4):
    for c in range(4):
        GOLD[0 + 4 * r + c] = X[r][c]
        GOLD[16 + 4 * r + c] = W_new[r][c]
        GOLD[36 + 4 * r + c] = Y[r][c]
        GOLD[52 + 4 * r + c] = dZ[r][c]
        GOLD[68 + 4 * r + c] = G[r][c]
for i in range(4):
    GOLD[32 + i] = B_new[i]


def to_fixed(val):
    scaled = val * (1 << FRAC)
    assert scaled == int(scaled), f"{val} not exact in Q8.8"
    return int(scaled) & 0xFFFF


def from_fixed(word):
    w = word & 0xFFFF
    v = w - 0x10000 if w & 0x8000 else w
    return v / (1 << FRAC)


class ProgGen:
    """Offline copy of test_tpu_nxn_ic_train_n4's Host: composes the
    identical per-cycle instruction-word stream. emit() sets the
    current word; tick() commits one cycle of it to the program."""

    def __init__(self):
        self.pathway = 0
        self.leak = to_fixed(LEAK)
        self.inv2b = to_fixed(INV2B)
        self.cur = 0
        self.prog = []

    def hold_word(self):
        # pathway[3:0] -> [97:94]; pathway[6:4] -> [132:130]; pathway[7]
        # (silu) -> the MSB 133+17*(N-2) (item 13: the SiLU bit is the
        # NEW top bit, NOT bit 133 — at N>2 bit 133 is inside the
        # appended host data-lane field). At N=2 the MSB is 133, so
        # N=2 programs are bit-identical to the old mapping.
        return ((self.pathway & 0xF) << 94) | (self.inv2b << 98) | \
               (self.leak << 114) | ((self.pathway & 0x70) << 126) | \
               ((self.pathway >> 7) << (133 + 17 * (N - 2)))

    def emit(self, word):
        self.cur = self.hold_word() | word

    def idle(self):
        self.emit(0)

    def tick(self, cycles=1):
        for _ in range(cycles):
            self.prog.append(self.cur)

    def write_beat(self, lane_words):
        assert len(lane_words) == N
        word = 0
        for i, w in enumerate(lane_words):
            if i == 0:
                word |= (w << 62) | (1 << 3)
            elif i == 1:
                word |= (w << 78) | (1 << 4)
            else:
                word |= (w << (133 + 16 * (i - 2))) | \
                        (1 << (133 + 16 * (N - 2) + (i - 2)))
        self.emit(word)
        self.tick()
        self.idle()

    def issue_read(self, ptr, addr, rows, cols, transpose=0,
                   pathway=None):
        word = (1 << 1) | (transpose << 2) | (cols << 5) | (rows << 21) | \
               (addr << 37) | (ptr << 53)
        self.emit(word)
        self.tick()
        if pathway is not None:
            self.pathway = pathway
        self.idle()

    def switch_pulse(self):
        self.emit(1 << 0)
        self.tick()
        self.idle()


def generate_program():
    """The 8b training-step choreography as a word stream (150 words:
    14 host-write + 60 pass-1 + 76 pass-2 cycles)."""
    g = ProgGen()
    g.idle()
    words = ([to_fixed(v) for v in X.flatten()]
             + [to_fixed(v) for v in W.flatten()]
             + [to_fixed(v) for v in B]
             + [to_fixed(v) for v in Y.flatten()])
    assert len(words) % N == 0
    for b in range(len(words) // N):
        g.write_beat([words[N * b + (N - 1 - i)] for i in range(N)])
    g.tick()  # trailing idle cycle after the last beat

    # pass 1: forward + loss (transition pathway)
    g.issue_read(ptr=1, addr=16, rows=4, cols=4, transpose=1)
    g.tick(9)
    g.issue_read(ptr=0, addr=0, rows=4, cols=4, pathway=0b0001111)
    g.switch_pulse()
    g.tick(2)
    g.issue_read(ptr=2, addr=32, rows=4, cols=4)
    g.tick(1)
    g.issue_read(ptr=3, addr=36, rows=4, cols=4)
    g.tick(2)
    g.issue_read(ptr=5, addr=32, rows=4, cols=4)
    g.tick(40)

    # pass 2: weight gradient G = dZ^T @ X
    g.issue_read(ptr=1, addr=0, rows=4, cols=4, transpose=0)
    g.tick(9)
    g.issue_read(ptr=0, addr=52, rows=4, cols=4, transpose=1,
                 pathway=0b0000000)
    g.switch_pulse()
    g.tick(3)
    g.issue_read(ptr=6, addr=16, rows=4, cols=4)
    g.tick(60)
    return g.prog


async def tick(dut, cycles=1):
    """Edge + 1ns settle: reads see post-edge values, drives land
    mid-cycle."""
    for _ in range(cycles):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")


@cocotb.test()
async def test_tpu_nxn_prog_n4(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    prog = generate_program()
    nxn = dut.tpu_nxn_prog_inst.tpu_nxn_ic_inst.tpu_nxn_inst

    # Reset (the program load happens with the chip held in reset).
    dut.rst.value = 1
    dut.prog_wr_en.value = 0
    dut.prog_wr_data.value = 0
    dut.run.value = 0
    dut.learning_rate_in.value = to_fixed(LR)
    await tick(dut, 2)

    # Load the program, one word per cycle.
    for w in prog:
        dut.prog_wr_data.value = w
        dut.prog_wr_en.value = 1
        await tick(dut)
    dut.prog_wr_en.value = 0
    dut.prog_wr_data.value = 0
    await tick(dut, 2)

    # Per-lane collectors on the VPU -> UB write interface (same probe
    # point as the 8b test, one more instance level down).
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

    # ---- per-lane stream checks (identical to the 8b test) ----
    for j in range(N):
        got_dz = lanes[j][0:4]
        expected_dz = [to_fixed(dZ[k][j]) for k in range(4)]
        assert got_dz == expected_dz, (
            f"VPU lane {j} pass 1: got "
            f"{[f'{from_fixed(w):+.4f}' for w in got_dz]}, expected "
            f"{[f'{from_fixed(w):+.4f}' for w in expected_dz]} "
            f"(dZ column {j})")
        got_g = lanes[j][4:8]
        expected_g = [to_fixed(G[k][j]) for k in range(4)]
        assert got_g == expected_g, (
            f"VPU lane {j} pass 2: got "
            f"{[f'{from_fixed(w):+.4f}' for w in got_g]}, expected "
            f"{[f'{from_fixed(w):+.4f}' for w in expected_g]} "
            f"(G column {j})")

    # ---- final UB image: exact match in every region ----
    for base, count, name in [(0, 16, "X"), (16, 16, "W'"), (32, 4, "B'"),
                              (36, 16, "Y"), (52, 16, "dZ"),
                              (68, 16, "G")]:
        for a in range(base, base + count):
            got = nxn.ub_inst.ub_sram.mem[a].value.integer & 0xFFFF
            want = to_fixed(GOLD[a])
            assert got == want, (
                f"{name} region: mem[{a}] = {from_fixed(got):+.4f}, "
                f"expected {GOLD[a]:+.4f}")

    print(f"tpu_nxn_prog N={N} program-driven training step OK "
          f"({len(prog)}-word loaded program; same golden as "
          f"test_tpu_nxn_ic_train_n4)")
