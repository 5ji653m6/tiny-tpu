# UB Re-architecture for Routable Hardening (Item 25)

Status: DESIGN (2026-08-04). Predecessor: `docs/SRAM_INTEGRATION.md` (behavioral
2NW/6NR macro swap — timing bit-exact in sim, but unroutable in synthesis at
true N).

## 1. Why this document exists

The behavioral DFF-UB makes true-N hardening impossible. Three stacked
pathologies, all rooted in the UB's port structure (full evidence in the
tinytpu-loop README item-24 entry):

1. `repair_design` (the ONLY fanout repair in LibreLane 3.0.5 — there is no
   synthesis-level fanout control) freezes on one net at ~85-93% of nets,
   deterministic (N=8 twins froze at the same iteration).
2. `repair_timing` never converges on the unrepaired netlist (N=8: 350+
   passes, WNS asymptote ~-787 ns at 0.007 ns/iter; N=16: zero-output freeze).
3. Global routing saturates (GRT-0228/0229, usage=65535 on single gcell
   edges) from unrepaired 1,026-4,097-terminal fanout nets (~350/run).

Fanout census on the N=8 post-ABC netlist (`diag/fanout_census.py`):

- `clk` 34,156 sinks (CTS's job — fine).
- `rst` 10,139 sinks (CLEAR_ON_RESET rides rst to every UB DFF).
- ~30 ABC-created nets at exactly 4096/4097/4098 sinks + one at 7,507.
  **4096 = 8 lanes x 512 words**: ABC discovered the per-lane read-mux select
  patterns are correlated and merged them into single nets driving all 8
  lanes' mux trees. These nets do not exist pre-ABC, so RTL-level per-lane
  buffering cannot fix them; the read path itself must change.

Conclusion: end-to-end true-N hardening is gated on replacing the 6N-read /
2N-write behavioral UB with a port-reduced, banked architecture that is
routable by construction while remaining cycle/bit-exact against the
behavioral model (the golden for the entire gate suite).

## 2. Complete walk-mode characterization

All address math from `src/unified_buffer_nxn.sv` (the combinational port
assembly, lines ~395-581). Within one cycle, one read group presents up to N
addresses (one per active lane). Active-lane condition for all groups:
`t >= i && t < R + i && i < C` (lane i starts at cycle i — the systolic
skew — and streams R beats). Notation: R/C = latched row/col sizes, base =
latched pointer, t = channel time counter, i = lane.

| Group | Mode | Lane-i address at cycle t | Stride across lanes (i) | Stride across cycles (t) | Touches each element |
|---|---|---|---|---|---|
| GRP_IN (ptr 0) | untransposed | base + (t-i)*C + i | -(C-1) | +C | once |
| GRP_IN (ptr 0) | transposed | base + i*(R'-1) + (t-i)... affine | +(R'-1) | +1 | once |
| GRP_W (ptr 1) | untransposed | base + t*... - i*skip, skip=C+1 | -(C+1) | re-read | R times (weights reused per output row) |
| GRP_W (ptr 1) | transposed | ... + i*skip | +(C+1) | re-read | R times |
| GRP_B (ptr 2) bias | broadcast | ptr + i (CONSTANT) | 1 | 0 | held whole walk |
| GRP_B (ptr 7/8) residual/scale | elementwise | ptr + (t-i)*C + i | -(C-1) | +C | once |
| GRP_Y (ptr 3) | descending | same form as GRP_IN untransposed | -(C-1) | +C | once |
| GRP_H (ptr 4) | descending | same | -(C-1) | +C | once |
| GRP_G (ptr 5) grad-bias | broadcast | ptr + i (CONSTANT) | 1 | 0 | held whole walk |
| GRP_G (ptr 6) grad-weight | descending | same form | -(C-1) | +C | once |

Writes:

| Channel | Lane-i address | Stride across lanes |
|---|---|---|
| [0,N) grad writeback (weights) | grad_descent_ptr + r*w + i | +1 (CONSECUTIVE) |
| [0,N) grad writeback (biases) | grad_descent_ptr + i | +1 (CONSECUTIVE) |
| [N,2N) VPU stream | stream_base + r*w + i | +1 (CONSECUTIVE) |
| [N,2N) host load | wr_ptr++ (decrementing lane loop) | +1 (CONSECUTIVE) |

### Structural findings

F1. **Every per-cycle address set is an arithmetic progression across lanes.**
    All read strides s in {0, ±(C-1), ±(C+1)}; all write strides are +1.
    No random access anywhere in the design.

F2. **Constant-address groups exist.** ptr-2 bias and ptr-5 grad-bias read the
    same N words every cycle of their walk. They can be hoisted out of the
    SRAM entirely: read once at walk start into N shadow registers, hold.
    Removes 2 of 6 read groups from port pressure permanently.

F3. **Skewed (mod-N) banking is conflict-free within a group at full width.**
    Bank = addr mod N. At C = N the strides are N-1 == -1, N+1 == +1, 1 — all
    coprime with N, so the N active lanes land in N DISTINCT banks. Writes
    (stride 1) are always conflict-free.

F4. **Partial-width walks can collide under plain mod-N banking.**
    For C < N, stride C-1 (or C+1) collides when gcd(C-1, N) > 1 and the
    active-lane span exceeds N/gcd (e.g. N=16, C=9: stride 8, lanes 0 and 2
    both hit bank b). Mitigation options: XOR-fold swizzle
    (bank = (addr ^ (addr >> log2N)) mod N — keeps stride-1 consecutive
    writes conflict-free only if the fold is above the low bits... needs
    care), address-offset swizzle per phase, or accepting a 1-cycle skid on
    collision (changes timing — REJECTED, bit-exactness forbids). RESOLUTION
    REQUIRED before RTL; verify the swizzle against every (R, C) pair the
    gate suite exercises.

F5. **Across-group bank collisions are the real port-budget question.**
    Within one cycle, GRP_IN + GRP_W are both active every matmul cycle, and
    during overlap windows stream writes + grad writeback + operand reads can
    all fire (mid-phase next-operand reads, items 17a/18a). Mod-N banking
    does NOT separate groups (regions are runtime-assigned pointers). The
    maximum concurrent (group, bank) pressure per phase is UNKNOWN — this is
    the next diagnostic (section 4).

F6. **The mux forest is the fanout pathology.** Each of the 6N read ports is
    a 512:1 x 16-bit mux whose select is the per-lane address; ABC merges
    correlated selects across lanes into 4096-sink nets. A banked UB replaces
    each 512:1 mux with N banks of DEPTH/N:1 muxes (32:1 at N=16, DEPTH=512)
    — select fanout per net drops ~N-fold by construction, and no single net
    spans lanes.

## 3. Architecture candidates

Constraints carried from the item-24 analysis: no 2NW/6NR sky130 macro
exists; banking was previously rejected FOR SIM (cross-channel concurrency +
even-stride conflicts — F4/F5 are exactly those concerns, now quantified);
CLEAR_ON_RESET is behavioral-only (silicon needs a boot scrub — sequencer
micro-routine or host writes; small state machine, does not affect steady
state); bit/cycle-exactness vs the behavioral model is non-negotiable (the
entire 67-target gate baseline compares against it).

### (a) Banked 1RW/1R + port arbitration — REJECTED as stated
Time-multiplexing ports inserts bubbles; bit-exactness forbids. Arbitration
without bubbles = more physical ports, i.e. (e).

### (b) Compiled latch/FF register-file macro — DEFERRED
No RF compiler in sky130; a hand-built latch array is a custom-macro project
of its own. Revisit if (e) misses area targets.

### (c) Word-width folding (store N consecutive elements per wide word) —
    REJECTED
Read strides are C-dependent (runtime); only stride-1 accesses (writes, bias)
are contiguous. Folding helps writes but breaks every descending/weight walk.

### (d) Keep DFF-UB, kill CLEAR_ON_RESET + tree-buffer control nets — REJECTED
    (insufficient)
The 4096-sink monsters are ABC-merged mux selects (F6) — they don't exist in
RTL and can't be buffered by construction. Post-synthesis buffering = repair
= the hang. DFF-UB at N=16 is also ~18 mm2 / ~500k gates — unroutable even
with perfect fanout hygiene.

### (e) Skewed banked UB with replicated 1RW/1R banks — CANDIDATE
- N banks, bank = swizzle(addr), DEPTH/N words of 16 bits each.
- Each bank = K physical sky130 1rw1r replicas (write-all, read-any) giving
  K+1 read and 2 write accesses per bank per cycle (RW port can carry the
  2nd write). K sized by the F5 diagnostic; target K = 2 (matmul operand
  pair) + slack.
- Constant-address groups (F2) shadowed into N-entry register files at walk
  start — GRP_B/GRP_GB leave the port budget.
- Area at N=16, DEPTH=32768: 32768x16 bits x K replicas ~ K x 3.15M
  transistors of macro — 2 replicas ~ 6.3M ~ 3.3x better than DFF-UB, and
  ROUTABLE (hardened macros + std-cell glue only).
- Timing: bank read = the same 1-cycle sync read as the behavioral macro;
  address generation (walk math) is unchanged — the same arithmetic feeds
  bank-select + in-bank address. Bit-exact by construction; the behavioral
  sram_macro stays the sim golden under the non-SYNTH path.

### Decision
(e), pending the F4 swizzle resolution and the F5 concurrency diagnostic.

## 4. Next diagnostics (before RTL)

D1. **Per-phase group-concurrency measurement.** Instrument
    `unified_buffer_nxn.sv` (ifdef'd $display of the active-group mask +
    write-channel mask per cycle) and run the heaviest gate target
    (adaln_n16 capstone). Reduce the log to: max concurrent read groups, max
    concurrent (read+write) channels, and the joint histogram. Sets K.

D2. **Collision census for candidate swizzles.** Python model of the walk
    math (section 2 table) enumerated over every (R, C, transpose, ptr) the
    gate suite + DiT programs use; for each candidate swizzle, count
    within-group bank conflicts. Pick a swizzle with zero conflicts; if none
    exists for plain schemes, restrict stride pathologies by UB layout
    convention (pad odd strides at write time — changes ProgGen placement
    only, not RTL timing).

D3. **sky130 macro geometry at DEPTH/N.** 32768/16 = 2048 words x 16b =
    4KB/bank; the existing 1kbyte macro x 4 stacked, or a 4KB ciel macro if
    available. Bank pinout feeds the floorplan sketch.

## 5. Diagnostic results

### D1/D2 first pass — adaln_n16 capstone (N=16, 6285 active cycles)

Instrumentation: `UB_TRACE` ifdef in `unified_buffer_nxn.sv` dumps per-cycle
read windows + all read/write addresses; `diag/ub_concurrency.py` reduces.

- **Max concurrent read groups: 2** (IN+B, 360 cycles). Never 3+. Y/H/G
  (training-only channels) are idle in the inference capstone.
- **IN and W NEVER overlap** — the weight-stationary dataflow separates the
  weight-load phase from the activation stream. W alone 1998 cycles, IN
  1998, B 744.
- **Write channels never concurrent** (STRM only; GRAD idle in inference).
- **Plain mod-N banking: ZERO within-group conflicts, ZERO rr, ZERO ww**;
  rw (read+write same bank) 4704 cycles; **max bank pressure = 2** accesses
  (1R+1W) — EXACTLY a 1rw1r macro per bank, no replication (K=1).
- **Zero same-address R+W cycles** in this program (read-during-write
  semantics not exercised — must be re-checked on training traces where
  grad writeback updates W'/B' in place in the G-read region).
- XOR-fold swizzles only make things worse (within-group conflicts appear:
  stride-1 writes alias under folding); plain mod-N is the right bank map at
  N=16.

**Verdict so far**: N banks x 1rw1r, bank = addr mod N, K=1. Remaining
risks: partial-width strides (C=3 at N=4 -> stride 2, gcd 2) in the ncol
tests, and same-address R/W in the training/ktil tests (behavioral model
returns OLD data; the sky130 macro is X — if any program needs it, the
banked UB needs a same-address bypass latch or a documented old-data
guarantee).

### D1/D2 full gate sweep — VERIFIED (2026-08-04)

All 19 program-level gate targets traced (`make <t> IVERILOG='iverilog
-DUB_TRACE'` in the tinytpu-sim container, stale results.xml deleted per
run). Inference scope = every loaded-program test; training scope = the
four train targets (UB-direct tests like ncol are UB-unit-level and listed
separately).

| Scope | Tests | within-group | rr | ww | same-addr R+W | max bank pressure |
|---|---|---|---|---|---|---|
| Inference (N=4 attn/mh_attn/loopi/tiled/radd/dit/scale/adaln/prog, N=8 prog/adaln, N=16 prog/ktil/adaln) | 15 | 0 | 0 | 0 | 0 | **2 (1R+1W)** |
| Training (train_n4, ic_train_n4, ic_train2_n4, prog_train2_n4) | 4 | 0 | 8-16 | 8-16 | 5-10 | 3 |
| UB-unit (ncol_n4, ncol_streams_n4) | 2 | 4 (stride-2 at C=3) | 0 | 0 | 0 | 3 |

**THE CONTRACT (inference)**: under plain mod-N banking, every gated
inference program needs at most 1 read + 1 write per bank per cycle — one
sky130 1rw1r macro per bank, K=1, no replication, no swizzle. Bank select =
addr[log2N-1:0] (a wire slice, zero logic).

**Training is out of scope for the banked UB**: the four training tests
violate the contract (grad writeback + stream writes collide ww; G-reads
collide rr with operand reads; in-place W'/B' updates need read-old on
same-address R+W — the behavioral macro returns OLD data, a silicon macro
returns X). Training keeps the behavioral 2NW/6NR sram_macro model even on
the hardening path; the chip's goal is inference. (If training hardening is
ever wanted: K=2 replicas + a genuine 2W scheme + read-old bypass — noted,
not planned.)

**Partial-width strides**: gcd(stride, N) > 1 walks (C=3 at N=4) collide
within a group under mod-N. No gated inference program does this, but the
space of loadable programs is open. Fallback on record: prime bank count
P = next_prime(N+2) (N=8 -> 11, N=16 -> 19) — all design strides are
<= N+1 < P and none is a multiple of P, so mod-P banking is conflict-free
for EVERY (R, C, transpose) by construction (P prime, lane deltas < P).
Cost: a mod-P divider on the address path and ⌈DEPTH/P⌉-deep banks.
v1 ships mod-N; sim-time assertions (below) catch any future program that
leaves the contract.

**Sim-time contract assertions** (non-synthesis only, `$error`): two stream
writes or grad+stream write to one bank in one cycle (ww); two read ports
to one bank (rr); read and write of the same address in one cycle
(read-old semantics unavailable in silicon).

## 6. v1 architecture (decision)

`src/ub_banked.sv` — drop-in storage replacement inside `unified_buffer_nxn`
under the SYNTH_SRAM_MACROS ifdef family (behavioral 2NW/6NR sram_macro
stays the default sim golden):

- N banks; bank = addr[log2N-1:0], offset = addr >> log2N.
- Per bank: one 1rw1r macro (behavioral bank model in sim; sky130 wrapper
  in the hardening flow — 2048 words x 16b at N=16 = 4KB = 2x 2kbyte
  ciel macros; 32 UB macros total at N=16 + 12 prog_mem from item 24).
- Read crossbar: per bank, the requesting port's address is OR-selected
  (one-hot by contract); per port, data = N:1 mux of bank outputs by the
  port's bank-select bits. 6N x N:1 x 16b data muxes + N x 6N-way address
  OR-selects — ~30x less mux logic than the 512:1-per-port forest, and
  every select net is private to its small mux (fanout ~N, nothing for ABC
  to merge across lanes).
- Writes: per bank, grad channel wins over stream (matches the behavioral
  port-priority for the same-address case); ww/rr/same-address violations
  hit the sim-time assertions.
- Read-during-write different address in the same bank: native 1rw1r
  behavior (the inference worst case, 4704 cycles in adaln_n16).
- CLEAR_ON_RESET: sim keeps a behavioral clear in the bank model under
  `ifndef SYNTHESIS`; silicon needs a boot scrub (sequencer or host walks
  all banks post-reset) — called out for the hardening flow, no RTL cost.
- Timing: 1-cycle synchronous read identical to the behavioral macro;
  the walk/address generation in unified_buffer_nxn is UNCHANGED — only
  the storage block behind sr_rd_addr/sr_rd_data/sr_wr_* is swapped.
  Bit-exact by construction against the behavioral model.

Verification plan (section 5 deliverables): equivalence unit test driving
both storage blocks with identical stimulus (directed walks + randomized
(R,C,transpose,base)), then the full 67-target gate on the SYNTH path
(training targets excluded — documented scope), then N=8 hardening
(must pass repair_design and route), then N=16.

## 5. Execution plan (after D1-D3)

1. `src/ub_banked.sv` + `src/sram_bank_1rw1r.sv` wrapper, under
   `SYNTH_SRAM_MACROS` ifdef family alongside the item-24 prog_mem swap;
   behavioral sram_macro stays the default sim model.
2. Equivalence: new unit test `test/test_ub_banked.py` driving both UBs with
   identical port stimulus (directed walks + randomized (R,C,transpose)),
   comparing per-cycle outputs bit-exactly.
3. Full gate baseline (67 targets) green on the SYNTH path.
4. N=8 hardening run: must pass repair_design and route. Then N=16.
