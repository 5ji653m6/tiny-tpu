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
  2. GROUP STAGES, SEMANTIC: the group stage (layernorm, softmax) is
     N-dependent (roadmap item 7b): at N=2 it is the legacy 2-lane pair
     module (bit-identity with vpu.sv); at N>2 it is a single N-wide
     layernorm_group_nxn / softmax_group_nxn instance with full de-skew
     (lane k delayed N-1-k at the input) and re-skew (lane k delayed k at
     the output). Three discriminators:
       - SKEW PRESERVATION: with all lanes streaming on the native skew,
         each lane's first valid output must be strictly later than the
         previous lane's (the re-skew restores the one-cycle column skew).
       - GROUP SEMANTICS: at N=2, driving only pair 0 leaves lanes 2..N-1
         silent (pair isolation); at N>2, driving a PARTIAL group (only
         pair 0) must stall ALL lanes — the N-wide stage's all-valid
         handshake waits for every lane.
       - VALUES (N>2): distinct constant vectors per beat through the full
         de-skew/group/re-skew path, compared against the exact group
         models from test_layernorm_group_nxn / test_softmax_group_nxn —
         a de-skew misalignment would mix beats and fail outright.
     The N=2 math itself is anchored by the N=2 equivalence test.

Unlike the upstream tests, the asserts here are LIVE — PYTHONOPTIMIZE is
empty (NOASSERT is never defined), so they fire.
"""

import math
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

FRAC_BITS = 8
LSB = 1.0 / (1 << FRAC_BITS)
LN_TOL = 12 * LSB  # matches test_layernorm_group_nxn
SM_TOL = 4 * LSB   # matches test_softmax_group_nxn

EXP_LUT = [round(256 * math.exp(-0.25 * k)) for k in range(33)]


def exp_lut_exact(e_raw):
    """The N-wide softmax leaf's piecewise-linear exp(-|e|) on a raw Q8.8
    magnitude, with the RTL's exact integer arithmetic (see
    test_softmax_group_nxn.py)."""
    if e_raw >= 2048:
        return 0
    seg = e_raw >> 6
    frac = e_raw & 0x3F
    slope = EXP_LUT[seg + 1] - EXP_LUT[seg]
    return EXP_LUT[seg] + ((slope * frac + 32) >> 6)


def softmax_group_spec(xs):
    """N-wide softmax on Q8.8 integer inputs -> float lanes."""
    m = max(xs)
    exps = [exp_lut_exact(m - x) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def layernorm_group_spec(xs):
    """N-wide layernorm on Q8.8 integer inputs -> float lanes. The mean
    replicates the RTL's truncating arithmetic shift exactly."""
    mean_raw = sum(xs) >> (len(xs).bit_length() - 1)  # truncating shift
    devs = [(x - mean_raw) / (1 << FRAC_BITS) for x in xs]
    var = sum(d * d for d in devs) / len(xs)
    std = math.sqrt(var + 1.0 / 16)
    return [d / std for d in devs]


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
    await tick(dut, N + 8)  # drain the full stage chain + re-skew
                            # registers (lane k re-skew delay = k cycles;
                            # drain must be >= N to accommodate lane N-1
                            # at arbitrary N — 14 was enough for N<=8,
                            # short for N>=16; N+8 gives headroom for the
                            # group-stage pipeline latency)
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

    # ---- 4. group stages: skew preservation + group semantics ---------
    for pathway, name in ((PW_LN, "layernorm"), (PW_SM, "softmax")):
        # 4a. all lanes streaming on the native skew: every lane must emit
        # nbeats, and first-valid cycles must strictly increase with lane
        # index (the re-skew restores the column skew).
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

        # 4b. group semantics: at N=2 the stage is the legacy pair module,
        # so driving only pair 0 is a complete group and emits; at N>2
        # the stage is a single N-wide group, so a partial group (only
        # pair 0 driven) must stall ALL lanes (all-valid handshake).
        await stream(dut, pathway, gdata, nbeats, lanes=[0, 1])
        if N == 2:
            assert (len(col.beats[0]) == nbeats
                    and len(col.beats[1]) == nbeats), (
                f"{name} pair isolation: pair 0 emitted "
                f"{len(col.beats[0])}/{len(col.beats[1])} beats, "
                f"expected {nbeats}/{nbeats}")
            for i in range(2, N):
                assert col.beats[i] == [], (
                    f"{name} pair isolation: lane {i} emitted "
                    f"{col.beats[i]} with only pair 0 driven — group "
                    f"stage spans more than one lane pair")
        else:
            for i in range(N):
                assert col.beats[i] == [], (
                    f"{name} group stall: lane {i} emitted "
                    f"{col.beats[i]} with only lanes 0,1 driven — the "
                    f"N-wide group stage must wait for ALL {N} lanes")
        col.clear()

    # ---- 4c. group stage VALUES at N=4 (de-skew/re-skew alignment) ----
    # Distinct constant vector per beat; a de-skew misalignment would mix
    # beats and diverge from the exact N-wide models far beyond tolerance.
    # (Vectors are length-4; extend when the gate grows past N=4.)
    if N == 4:
        for pathway, name, spec, vecs, tol in (
            (PW_LN, "layernorm", layernorm_group_spec,
             [(0x0300, 0x0100, -0x0100, -0x0300),
              (0x0200, 0x0200, 0x0200, 0x0200),
              (0x0080, -0x0080, 0x0040, -0x0040)], LN_TOL),
            (PW_SM, "softmax", softmax_group_spec,
             [(0x0200, 0x0100, 0x0000, -0x0100),
              (0x0080, 0x0080, 0x0080, 0x0080),
              (0x0900, 0x0000, 0x0000, 0x0000)], SM_TOL),
        ):
            vdata = [[vecs[b][k] for b in range(len(vecs))]
                     for k in range(N)]
            await stream(dut, pathway, vdata, len(vecs))
            for i in range(N):
                assert len(col.beats[i]) == len(vecs), (
                    f"{name} values lane {i}: got {len(col.beats[i])} "
                    f"beats, expected {len(vecs)}")
                for b in range(len(vecs)):
                    exp = spec(list(vecs[b]))[i]
                    got = col.beats[i][b]
                    if got >= 1 << 15:
                        got -= 1 << 16
                    got /= 1 << FRAC_BITS
                    assert abs(got - exp) <= tol, (
                        f"{name} values lane {i} beat {b}: got "
                        f"{got:.5f}, expected {exp:.5f} "
                        f"(tol {tol:.5f}) — de-skew/re-skew misalignment "
                        f"or wrong group math")
            col.clear()

    collector.kill()
    print(f"N={N} vpu_nxn exact/semantic passes OK "
          f"(bypass, bias, bias+lr exact; ln/sm skew + group semantics"
          f"{' + values' if N == 4 else ''})")
