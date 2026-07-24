"""Gate test for roadmap item 9a: TWO in-place training steps through
the instruction port of tpu_nxn_ic. Written by the harness author, not
the agent — per tinytpu-loop README "the gate grows with the design".

No new RTL: this reuses src/tpu_nxn_ic.sv (item 8b) and runs the
instruction choreography of test_tpu_nxn_ic_train_n4 TWICE back to
back with no host rewrite in between — step 2's forward pass reads
the W'/B' images that step 1 wrote back in place, so the chip
genuinely trains twice on the same data.

UB ADDRESSING FACT (trace-verified 2026-07-25): VPU result streams do
NOT have fixed destinations — each stream lands at the UB write
pointer (wr_ptr), which advances past every stream and host region
(BUG-UB-3 row-major placement). With the 52-word host image, step 1's
dZ1 lands @52 and G1 @68 (wr_ptr then 84); step 2's dZ2 lands @84 and
G2 @100. Only the gradient-descent writebacks (W', B') are in-place
(@16/@32, driven by the explicit read addresses). So step 2's pass-2
dZ read must target @84 — reading @52 would recompute G from the
stale dZ1 (this exact failure was observed: G1 values re-emitted).
W''/B'' ARE trained twice in place.

Stimulus: the 8b matrices scaled by 2, i.e. INTEGER valued. Rationale:
two composed steps multiply rounding depth; with the original
0.5-multiple stimulus, step 2's dZ2/B'' terms hit 2^-9 and Q8.8 could
not represent them exactly. With integers, every intermediate of both
steps lands at >= 2^-8 granularity, so to_fixed()'s exactness assert
holds for the whole golden. (LR = LEAK = INV2B = 0.5 as before.)

Checks (LIVE asserts — PYTHONOPTIMIZE is empty):
  - per-lane VPU streams capture exactly 16 beats: dZ1, G1, dZ2, G2
    columns, each beat-exact against the twice-iterated numpy golden;
  - the final UB image: X/Y regions untouched, W''/B'' in place, the
    step-1 dZ1/G1 regions INTACT @52/@68, and the step-2 dZ2/G2
    regions appended @84/@100 (the append itself is part of the
    check).
"""

import os

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

N = int(os.environ.get("TPU_NXN_IC_N", "4"))
FRAC = 8

# ---- stimulus: the 8b matrices x2 (integers, exact in Q8.8) ----
X = np.array([[2.0, 1.0, 0.0, -1.0],
              [1.0, -2.0, 1.0, 2.0],
              [-1.0, 1.0, 2.0, 0.0],
              [0.0, 2.0, -1.0, 1.0]])
W = np.array([[1.0, -1.0, 2.0, 1.0],
              [2.0, 1.0, -1.0, 0.0],
              [0.0, 2.0, 1.0, -2.0],
              [-1.0, 0.0, 2.0, 1.0]])
B = np.array([1.0, -1.0, 1.0, -1.0])
Y = np.array([[2.0, -1.0, 1.0, 0.0],
              [1.0, 2.0, -1.0, 1.0],
              [-1.0, 0.0, 2.0, 1.0],
              [1.0, -2.0, 0.0, 2.0]])
LR = 0.5
LEAK = 0.5
INV2B = 0.5  # 2/batch, batch = 4


def train_step(Wcur, Bcur):
    """One SGD step; returns (W_new, B_new, dZ, G)."""
    Z = X @ Wcur.T
    Hpre = Z + Bcur
    H = np.where(Hpre >= 0, Hpre, Hpre * LEAK)
    dZ = np.where(H >= 0, (H - Y) * INV2B, (H - Y) * INV2B * LEAK)
    B_new = Bcur - LR * dZ.sum(axis=0)
    G = dZ.T @ X
    W_new = Wcur - LR * G
    return W_new, B_new, dZ, G


# ---- golden: iterate twice (step 2 reads step 1's in-place W'/B') ----
W1, B1, dZ1, G1 = train_step(W, B)
W2, B2, dZ2, G2 = train_step(W1, B1)

GOLD = {}
for r in range(4):
    for c in range(4):
        GOLD[0 + 4 * r + c] = X[r][c]
        GOLD[16 + 4 * r + c] = W2[r][c]
        GOLD[36 + 4 * r + c] = Y[r][c]
        GOLD[52 + 4 * r + c] = dZ1[r][c]   # step-1 regions stay intact
        GOLD[68 + 4 * r + c] = G1[r][c]
        GOLD[84 + 4 * r + c] = dZ2[r][c]   # step-2 streams append at
        GOLD[100 + 4 * r + c] = G2[r][c]   # the UB write pointer
for i in range(4):
    GOLD[32 + i] = B2[i]


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
    """Composes instruction words — identical to the 8b test's Host
    (pathway/leak/inv_batch are phase state; every word is hold |
    command)."""

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


async def run_training_step(host, dut, dz_addr):
    """The 8b pass-1/pass-2 choreography, verbatim except dz_addr: the
    pass-2 dZ read must target where THIS step's dZ stream landed
    (streams append at the UB write pointer — step 1's dZ @52, step
    2's @84). Reads X@0, W@16, B@32, Y@36; writes dZ/G at wr_ptr and
    updates W/B in place — so a second invocation trains on the
    first's results."""
    # pass 1: forward + loss (transition pathway)
    await host.issue_read(ptr=1, addr=16, rows=4, cols=4, transpose=1)
    await tick(dut, 9)  # N=4 weight walk = R+C-1 = 7 cycles + margin

    await host.issue_read(ptr=0, addr=0, rows=4, cols=4,
                          pathway=0b0001111)
    await host.switch_pulse()

    await tick(dut, 2)  # B read: 2 cycles after switch deassert
    await host.issue_read(ptr=2, addr=32, rows=4, cols=4)

    await tick(dut, 1)  # Y read: 1 cycle after the B read
    await host.issue_read(ptr=3, addr=36, rows=4, cols=4)

    await tick(dut, 2)  # bias-gradient read pairing
    await host.issue_read(ptr=5, addr=32, rows=4, cols=4)

    await tick(dut, 40)  # dZ lands at wr_ptr, B updated @32-35

    # pass 2: weight gradient G = dZ^T @ X
    await host.issue_read(ptr=1, addr=0, rows=4, cols=4, transpose=0)
    await tick(dut, 9)  # X "weight" walk to the top of the array

    await host.issue_read(ptr=0, addr=dz_addr, rows=4, cols=4,
                          transpose=1, pathway=0b0000000)
    await host.switch_pulse()

    await tick(dut, 3)  # G beats span capture counters 1..7
    await host.issue_read(ptr=6, addr=16, rows=4, cols=4)

    await tick(dut, 60)  # G lands at wr_ptr, W updated in place @16-31


@cocotb.test()
async def test_tpu_nxn_ic_train2_n4(dut):
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

    lanes = [[] for _ in range(N)]

    async def collect():
        while True:
            await RisingEdge(dut.clk)
            for i in range(N):
                if nxn.ub_wr_valid_in[i].value.integer:
                    lanes[i].append(nxn.ub_wr_data_in[i].value.integer
                                    & 0xFFFF)

    collector = cocotb.start_soon(collect())

    # Host image: X @ 0, W @ 16, B @ 32, Y @ 36 (52 words). Loaded ONCE
    # — both steps read/update these regions in place.
    words = ([to_fixed(v) for v in X.flatten()]
             + [to_fixed(v) for v in W.flatten()]
             + [to_fixed(v) for v in B]
             + [to_fixed(v) for v in Y.flatten()])
    assert len(words) % N == 0
    for b in range(len(words) // N):
        await host.write_beat([words[N * b + (N - 1 - i)]
                               for i in range(N)])
    await tick(dut)

    # ---- step 1, then step 2 with NO host traffic in between ----
    # Step 1's dZ lands @52 (wr_ptr after the 52-word host image), its
    # G @68; step 2's dZ therefore lands @84 — its pass-2 dZ read must
    # target @84 (reading @52 recomputes G from the stale dZ1).
    await run_training_step(host, dut, dz_addr=52)
    await run_training_step(host, dut, dz_addr=84)

    collector.kill()

    # ---- per-lane streams: dZ1, G1, dZ2, G2 (16 beats per lane) ----
    for j in range(N):
        for k, (name, mat) in enumerate([("dZ1", dZ1), ("G1", G1),
                                         ("dZ2", dZ2), ("G2", G2)]):
            got = lanes[j][4 * k:4 * k + 4]
            expected = [to_fixed(mat[r][j]) for r in range(4)]
            assert got == expected, (
                f"VPU lane {j} {name}: got "
                f"{[f'{from_fixed(w):+.4f}' for w in got]}, expected "
                f"{[f'{from_fixed(w):+.4f}' for w in expected]} "
                f"(column {j})")
        assert len(lanes[j]) == 16, (
            f"VPU lane {j}: expected exactly 16 beats "
            f"(dZ1,G1,dZ2,G2), got {len(lanes[j])}")

    # ---- final UB image: both steps' regions, exact everywhere ----
    for base, count, name in [(0, 16, "X"), (16, 16, "W''"),
                              (32, 4, "B''"), (36, 16, "Y"),
                              (52, 16, "dZ1"), (68, 16, "G1"),
                              (84, 16, "dZ2"), (100, 16, "G2")]:
        for a in range(base, base + count):
            got = nxn.ub_inst.ub_memory[a].value.integer & 0xFFFF
            want = to_fixed(GOLD[a])
            assert got == want, (
                f"{name} region: mem[{a}] = {from_fixed(got):+.4f}, "
                f"expected {GOLD[a]:+.4f}")

    print(f"tpu_nxn_ic N={N} TWO in-place training steps OK "
          f"(integer stimulus, twice-iterated golden; dZ1/G1 @52/@68 "
          f"intact, dZ2/G2 appended @84/@100, W''/B'' in place)")
