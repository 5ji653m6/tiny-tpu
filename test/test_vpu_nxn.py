"""Gate test for src/vpu_nxn.sv (roadmap item 5c), exact/semantic half at
parameter N (the Makefile's N=4 target overrides the width with
iverilog -Pdump.N=4 and exports VPU_NXN_N). Written by the harness author,
not the agent — per tinytpu-loop README "the gate grows with the design".

The companion test_vpu_nxn_equiv.py anchors vpu_nxn to the legacy vpu
cycle-by-cycle at N=2 (the legacy module is the spec for ALL stage math).
THIS test proves the parameterization itself at N:

  1. PER-LANE STAGES, EXACT: bypass and bias(+leaky_relu on nonnegative
     data, where lr is the identity) pathways with natively-skewed streams
     (lane k leads lane k+1 by one cycle). Expected per-lane output
     sequences are exact integers (out = in, out = in + bias_scalar), so a
     lane-indexing or port-mapping mistake fails outright.
  2. GROUP STAGES, SEMANTIC: layernorm and softmax are 2-lane math; at N
     they must operate PER LANE PAIR (2p, 2p+1) with per-pair BUG-SKEW-1
     alignment. Two discriminators that need no fixed-point golden model:
       - PAIR ISOLATION: drive only pair 0; lanes 2..N-1 must stay SILENT
         (a mistakenly N-wide group stage would stall waiting for all N
         lanes, or bleed across pairs).
       - SKEW PRESERVATION: with all lanes streaming on the native skew,
         each lane's first valid output must be strictly later than the
         previous lane's (the re-skew registers restore the one-cycle
         column skew per pair).
     The math itself is anchored by the N=2 equivalence test.

Unlike the upstream tests, the asserts here are LIVE — PYTHONOPTIMIZE is
empty (NOASSERT is never defined), so they fire.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

N = int(os.environ.get("VPU_NXN_N", "2"))

# Pathway encoding: |sm(6)|ln(5)|gelu(4)|bias(3)|lr(2)|loss(1)|lr_d(0)|
PW_BYPASS = 0b0000000
PW_BIAS = 0b0001000
PW_BIAS_LR = 0b0001100
PW_LN = 0b0100000
PW_SM = 0b1000000

armed = False


async def tick(dut, cycles=1):
    for _ in range(cycles):
        await RisingEdge(dut.clk)


async def drive_idle(dut):
    for i in range(N):
        dut.vpu_data_in[i].value = 0
        dut.vpu_valid_in[i].value = 0
        dut.bias_scalar_in[i].value = 0
        dut.Y_in[i].value = 0
        dut.H_in[i].value = 0
    dut.vpu_data_pathway.value = 0


class Collector:
    """Valid-qualified per-lane output beat collector. Started only after
    reset has clocked the X's away (reads right after RisingEdge see the
    pre-edge state)."""

    def __init__(self, dut):
        self.dut = dut
        self.beats = [[] for _ in range(N)]   # data values per lane
        self.first_cycle = [None] * N         # cycle of each lane's 1st beat
        self.cycle = 0

    async def run(self):
        while True:
            await RisingEdge(self.dut.clk)
            if not armed:
                continue
            self.cycle += 1
            for i in range(N):
                if self.dut.vpu_nxn.vpu_valid_out[i].value.integer:
                    self.beats[i].append(
                        self.dut.vpu_nxn.vpu_data_out[i].value.integer)
                    if self.first_cycle[i] is None:
                        self.first_cycle[i] = self.cycle

    def clear(self):
        self.beats = [[] for _ in range(N)]
        self.first_cycle = [None] * N


async def stream(dut, pathway, lane_data, nbeats, lanes=None):
    """Drive nbeats on `lanes` (default all) with the native systolic skew:
    lane k's beat b is valid at cycle k + b. lane_data[k][b] is the Q8.8
    integer value. Returns after the pipeline drains."""
    if lanes is None:
        lanes = list(range(N))
    dut.vpu_data_pathway.value = pathway
    await tick(dut)
    for t in range(nbeats + N):
        for k in range(N):
            b = t - k
            if k in lanes and 0 <= b < nbeats:
                dut.vpu_data_in[k].value = lane_data[k][b]
                dut.vpu_valid_in[k].value = 1
            else:
                dut.vpu_data_in[k].value = 0
                dut.vpu_valid_in[k].value = 0
        await tick(dut)
    for k in range(N):  # idle the stream, but KEEP the scalar constants
        dut.vpu_data_in[k].value = 0
        dut.vpu_valid_in[k].value = 0
    dut.vpu_data_pathway.value = pathway  # keep routing until drained
    await tick(dut, 14)  # drain the full stage chain + re-skew registers
    dut.vpu_data_pathway.value = 0
    await tick(dut, 2)


def check_seq(col, expected, label):
    """Exact per-lane sequence comparison, then clear for the next phase."""
    for i in range(N):
        assert col.beats[i] == expected[i], (
            f"{label} lane {i}: got {col.beats[i]}, expected {expected[i]}")
    col.clear()


@cocotb.test()
async def test_vpu_nxn(dut):
    global armed
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    await drive_idle(dut)
    dut.lr_leak_factor_in.value = 26             # 0.1 in Q8.8
    dut.inv_batch_size_times_two_in.value = 128  # 0.5 in Q8.8
    biases = [0x0010 * (i + 1) for i in range(N)]
    for i in range(N):
        dut.bias_scalar_in[i].value = biases[i]
        dut.Y_in[i].value = 0x0100
        dut.H_in[i].value = 0x0080
    await tick(dut, 2)
    dut.rst.value = 0
    await tick(dut)
    armed = True

    col = Collector(dut)
    collector = cocotb.start_soon(col.run())

    nbeats = 3

    # ---- 1. bypass: out = in, exact, per lane -------------------------
    data = [[0x0100 * (k + 1) + b for b in range(nbeats)] for k in range(N)]
    await stream(dut, PW_BYPASS, data, nbeats)
    check_seq(col, data, "bypass")

    # ---- 2. bias only: out = in + bias_scalar, exact ------------------
    expected = [[v + biases[k] for v in data[k]] for k in range(N)]
    await stream(dut, PW_BIAS, data, nbeats)
    check_seq(col, expected, "bias")

    # ---- 3. bias + leaky relu on nonnegative data: lr is the identity -
    await stream(dut, PW_BIAS_LR, data, nbeats)
    check_seq(col, expected, "bias+lr (nonneg)")

    # ---- 4. group stages: skew preservation + pair isolation ----------
    for pathway, name in ((PW_LN, "layernorm"), (PW_SM, "softmax")):
        # 4a. all lanes streaming on the native skew: every lane must emit
        # nbeats, and first-valid cycles must strictly increase with lane
        # index (per-pair align + re-skew restores the column skew).
        gdata = [[0x0100 + 0x0080 * k for _ in range(nbeats)]
                 for k in range(N)]
        await stream(dut, pathway, gdata, nbeats)
        for i in range(N):
            assert len(col.beats[i]) == nbeats, (
                f"{name} lane {i}: got {len(col.beats[i])} beats, "
                f"expected {nbeats}")
            assert col.first_cycle[i] is not None
            if i > 0:
                assert col.first_cycle[i] > col.first_cycle[i - 1], (
                    f"{name}: lane {i} first beat at cycle "
                    f"{col.first_cycle[i]} not after lane {i-1} at "
                    f"{col.first_cycle[i - 1]} — re-skew broken")
        col.clear()

        # 4b. pair isolation: drive ONLY lanes 0,1. Lanes 2..N-1 must stay
        # silent — the group stage works per pair, not across all N lanes.
        await stream(dut, pathway, gdata, nbeats, lanes=[0, 1])
        assert len(col.beats[0]) == nbeats and len(col.beats[1]) == nbeats, (
            f"{name} pair isolation: pair 0 emitted "
            f"{len(col.beats[0])}/{len(col.beats[1])} beats, "
            f"expected {nbeats}/{nbeats}")
        for i in range(2, N):
            assert col.beats[i] == [], (
                f"{name} pair isolation: lane {i} emitted "
                f"{col.beats[i]} with only pair 0 driven — group stage "
                f"spans more than one lane pair")
        col.clear()

    collector.kill()
    print(f"N={N} vpu_nxn exact/semantic passes OK "
          f"(bypass, bias, bias+lr exact; ln/sm skew + pair isolation)")
