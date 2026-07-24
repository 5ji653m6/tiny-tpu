"""Gate test for src/tpu_nxn_ic.sv (roadmap item 8b): the FIRST
instruction-driven top — control_unit_nxn decoding a single
133+17*(N-2)-bit instruction port into tpu_nxn's field inputs. Written
by the harness author, not the agent — per tinytpu-loop README "the
gate grows with the design".

This is test_tpu_nxn_train_n4's exact end-to-end training step (same
stimulus, same numpy golden, same 84-word UB image, same per-lane dZ/G
stream checks, same cycle-by-cycle choreography) with ONE change: every
field-port drive is re-expressed as instruction words on the single
`instruction` port. Since control_unit_nxn is purely combinational and
tpu_nxn_ic is pinned to be a thin decode+instantiate shell around the
verified tpu_nxn, the choreography transfers cycle for cycle.

Instruction encoding (item-8a layout; N = SYSTOLIC_ARRAY_WIDTH = 4):
  bit  0        sys_switch_in           bit  1        ub_rd_start_in
  bit  2        ub_rd_transpose         bits 3/4      host valid lanes 0/1
  bits 5-20     ub_rd_col_size          bits 21-36    ub_rd_row_size
  bits 37-52    ub_rd_addr_in           bits 53-61    ub_ptr_select
  bits 62-77    host data lane 0        bits 78-93    host data lane 1
  bits 94-97    pathway[3:0]            bits 98-113   inv_batch_size_times_two
  bits 114-129  vpu_leak_factor         bits 130-132  pathway[6:4]
  lane k>=2 data : bit 133+16*(k-2) +: 16
  lane k>=2 valid: bit 133+16*(N-2)+(k-2)

The field-driven test held pathway/leak/inv_batch on static ports; the
instruction word has no persistence, so the Host below carries them as
phase state and composes every cycle's word as hold | command — which
is also what a real host would emit. learning_rate_in stays a separate
port (it is not in the instruction word).

Assertions (LIVE — PYTHONOPTIMIZE is empty): identical to
test_tpu_nxn_train_n4 — per-lane dZ and G streams beat-exact, and the
final UB image (X, W', B', Y, dZ, G regions) exact in every word.
"""

import os

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

N = int(os.environ.get("TPU_NXN_IC_N", "4"))
FRAC = 8

# ---- stimulus (all values exact in Q8.8) — identical to the
# field-driven train test ----
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
INV2B = 0.5  # 2/batch, batch = 4

# ---- golden model (identical to the field-driven train test) ----
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
    """Exact Q8.8 encoding; asserts the value is representable."""
    scaled = val * (1 << FRAC)
    assert scaled == int(scaled), f"{val} not exact in Q8.8"
    return int(scaled) & 0xFFFF


def from_fixed(word):
    w = word & 0xFFFF
    v = w - 0x10000 if w & 0x8000 else w
    return v / (1 << FRAC)


async def tick(dut, cycles=1):
    for _ in range(cycles):
        await RisingEdge(dut.clk)


class Host:
    """Composes instruction words. pathway/leak/inv_batch are phase
    state (a real host holds them per phase); every emitted word is
    hold | command."""

    def __init__(self, dut):
        self.dut = dut
        self.pathway = 0
        self.leak = to_fixed(LEAK)
        self.inv2b = to_fixed(INV2B)

    def hold_word(self):
        return ((self.pathway & 0xF) << 94) | (self.inv2b << 98) | \
               (self.leak << 114) | ((self.pathway >> 4) << 130)

    def emit(self, word):
        self.dut.instruction.value = self.hold_word() | word

    async def idle(self):
        self.emit(0)

    async def write_beat(self, lane_words):
        """One host-write beat: lane_words[i] = Q8.8 word for lane i."""
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
        await tick(self.dut)
        await self.idle()

    async def issue_read(self, ptr, addr, rows, cols, transpose=0,
                         pathway=None):
        word = (1 << 1) | (transpose << 2) | (cols << 5) | (rows << 21) | \
               (addr << 37) | (ptr << 53)
        self.emit(word)
        await tick(self.dut)
        if pathway is not None:
            self.pathway = pathway
        await self.idle()

    async def switch_pulse(self):
        self.emit(1 << 0)
        await tick(self.dut)
        await self.idle()


async def host_write_words(host, words):
    """words: list of Q8.8 ints; beat b lane i = words[4b + (3-i)]
    (BUG-UB-2), identical to the field-driven train test."""
    assert len(words) % N == 0
    for b in range(len(words) // N):
        await host.write_beat([words[N * b + (N - 1 - i)]
                               for i in range(N)])
    await tick(host.dut)


@cocotb.test()
async def test_tpu_nxn_ic_train_n4(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    host = Host(dut)
    nxn = dut.tpu_nxn_ic_inst.tpu_nxn_inst

    dut.rst.value = 1
    await host.idle()
    dut.learning_rate_in.value = to_fixed(LR)
    await tick(dut, 2)
    dut.rst.value = 0
    await tick(dut)

    # Per-lane collectors on the VPU -> UB write interface (same probe
    # point as the field-driven train test, one instance level deeper).
    lanes = [[] for _ in range(N)]

    async def collect():
        while True:
            await RisingEdge(dut.clk)
            for i in range(N):
                if nxn.ub_wr_valid_in[i].value.integer:
                    lanes[i].append(nxn.ub_wr_data_in[i].value.integer
                                    & 0xFFFF)

    collector = cocotb.start_soon(collect())

    # Host image: X @ 0, W @ 16, B @ 32, Y @ 36 (52 words).
    words = ([to_fixed(v) for v in X.flatten()]
             + [to_fixed(v) for v in W.flatten()]
             + [to_fixed(v) for v in B]
             + [to_fixed(v) for v in Y.flatten()])
    await host_write_words(host, words)

    # ---- pass 1: forward + loss (transition pathway) ----
    await host.issue_read(ptr=1, addr=16, rows=4, cols=4, transpose=1)
    await tick(dut, 9)  # N=4 weight walk = R+C-1 = 7 cycles + margin

    await host.issue_read(ptr=0, addr=0, rows=4, cols=4,
                          pathway=0b0001111)
    await host.switch_pulse()

    # B read: 2 cycles after switch deassert (5d-2 N=4 choreography).
    await tick(dut, 2)
    await host.issue_read(ptr=2, addr=32, rows=4, cols=4)

    # Y read: 1 cycle after the B read (trace-verified pairing with the
    # loss stage's H beats).
    await tick(dut, 1)
    await host.issue_read(ptr=3, addr=36, rows=4, cols=4)

    # bias-gradient read: first dZ beat at capture counter 1 (after the
    # value_old load), last beat at counter 7 < row+col — no clip.
    await tick(dut, 2)
    await host.issue_read(ptr=5, addr=32, rows=4, cols=4)

    await tick(dut, 40)  # dZ lands @52-67, B updated @32-35

    # ---- pass 2: weight gradient G = dZ^T @ X ----
    await host.issue_read(ptr=1, addr=0, rows=4, cols=4, transpose=0)
    await tick(dut, 9)  # X "weight" walk to the top of the array

    await host.issue_read(ptr=0, addr=52, rows=4, cols=4, transpose=1,
                          pathway=0b0000000)
    await host.switch_pulse()

    # weight-gradient read: 3 cycles after switch deassert so the G
    # beats span capture counters 1..7 (full capture, no clip).
    await tick(dut, 3)
    await host.issue_read(ptr=6, addr=16, rows=4, cols=4)

    await tick(dut, 60)
    collector.kill()

    # ---- per-lane stream checks (identical to the train test) ----
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
            got = nxn.ub_inst.ub_memory[a].value.integer & 0xFFFF
            want = to_fixed(GOLD[a])
            assert got == want, (
                f"{name} region: mem[{a}] = {from_fixed(got):+.4f}, "
                f"expected {GOLD[a]:+.4f}")

    print(f"tpu_nxn_ic N={N} instruction-driven end-to-end training "
          f"step OK (same golden as test_tpu_nxn_train_n4; per-lane "
          f"streams and full UB image verified)")
