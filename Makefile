#================DO NOT MODIFY BELOW===================== Compiler and simulator settings
IVERILOG = iverilog
VVP = vvp
COCOTB_PREFIX = $(shell cocotb-config --prefix)


COCOTB_LIBS = $(COCOTB_PREFIX)/cocotb/libs

# Windows (iverilog uses .vpl extension, no 'lib' prefix)
# Linux uses libcocotbvpi_icarus (.so); Windows uses cocotbvpi_icarus (.vpl)
ifeq ($(OS),Windows_NT)
  COCOTB_VPI_MODULE = cocotbvpi_icarus
else
  COCOTB_VPI_MODULE = libcocotbvpi_icarus
endif

SIM_BUILD_DIR = sim_build
SIM_VVP = $(SIM_BUILD_DIR)/sim.vvp

# Force bash shell on Windows so ! negation and Unix commands work.
# Guarded: setting SHELL unconditionally breaks make on Linux.
ifeq ($(OS),Windows_NT)
  SHELL = C:/Program Files/Git/bin/sh.exe
  .SHELLFLAGS = -c
endif

# Environment variables
export COCOTB_REDUCED_LOG_FMT=1
export LIBPYTHON_LOC=$(shell cocotb-config --libpython)
ifeq ($(OS),Windows_NT)
  export PYTHONPATH := $(abspath test);$(PYTHONPATH)
else
  export PYTHONPATH := $(abspath test):$(PYTHONPATH)
endif

#=============== MODIFY BELOW ======================
# ********** IF YOU HAVE A NEW VERILOG FILE, ADD IT TO THE SOURCES VARIABLE
SOURCES = src/pe.sv \
          src/leaky_relu_child.sv \
          src/leaky_relu_parent.sv \
          src/leaky_relu_derivative_child.sv \
          src/leaky_relu_derivative_parent.sv \
          src/systolic.sv \
          src/bias_child.sv \
          src/bias_parent.sv \
          src/fixedpoint.sv \
          src/control_unit.sv \
          src/unified_buffer.sv \
          src/vpu.sv \
          src/loss_parent.sv \
		  src/loss_child.sv \
		  src/tpu.sv \
		  src/gradient_descent.sv \
		  src/gelu_child.sv \
		  src/gelu_parent.sv \
		  src/layernorm_parent.sv \
		  src/softmax_parent.sv \
		  src/systolic_nxn.sv \
		  src/unified_buffer_nxn.sv \
		  src/vpu_nxn.sv \
		  src/tpu_nxn.sv

# MODIFY 1) variable next to -s 
# MODIFY 2) variable next to $(SOURCES)
# MODIFY 3) variable right of MODULE=
# MODIFY 4) file name next to mv (i.e. pe.vcd)


# Test targets
test_pe: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s pe -s dump -g2012 $(SOURCES) test/dump_pe.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_pe $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv pe.vcd waveforms/ 2>/dev/null || true

test_leaky_relu: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s leaky_relu -s dump -g2012 $(SOURCES) test/dump_leaky_relu.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_leaky_relu $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv test.vcd waveforms/ 2>/dev/null || true

test_systolic: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s systolic -s dump -g2012 $(SOURCES) test/dump_systolic.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_systolic $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv systolic.vcd waveforms/ 2>/dev/null || true

# Roadmap item 5a: parameterized NxN array (src/systolic_nxn.sv), gate test
# authored harness-side. Two targets: N=4 via iverilog -P override, and an
# N=2 equivalence run at the default parameter. SYSTOLIC_NXN_N tells the
# cocotb test which N the sim was compiled at.
test_systolic_nxn_n4: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s systolic_nxn -s dump -g2012 -Psystolic_nxn.SYSTOLIC_ARRAY_WIDTH=4 $(SOURCES) test/dump_systolic_nxn.sv
	PYTHONOPTIMIZE=$(NOASSERT) SYSTOLIC_NXN_N=4 MODULE=test_systolic_nxn $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv systolic_nxn.vcd waveforms/systolic_nxn_n4.vcd 2>/dev/null || true

test_systolic_nxn: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s systolic_nxn -s dump -g2012 $(SOURCES) test/dump_systolic_nxn.sv
	PYTHONOPTIMIZE=$(NOASSERT) SYSTOLIC_NXN_N=2 MODULE=test_systolic_nxn $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv systolic_nxn.vcd waveforms/ 2>/dev/null || true

# Roadmap item 5b: unified buffer with per-lane array read ports
# (src/unified_buffer_nxn.sv), gate tests authored harness-side.
# _equiv: legacy and nxn instances driven in lockstep at N=2, compared
# cycle-by-cycle (the legacy module is the spec). _n4: exact per-lane
# read sequences at N=4 via iverilog -P override (UB_NXN_N tells the
# cocotb test the compiled width). Also run plain at the default N=2.
test_unified_buffer_nxn_equiv: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s dump -g2012 $(SOURCES) test/dump_ub_nxn_equiv.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_unified_buffer_nxn_equiv $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv unified_buffer_nxn_equiv.vcd waveforms/ 2>/dev/null || true

test_unified_buffer_nxn: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s dump -g2012 $(SOURCES) test/dump_unified_buffer_nxn.sv
	PYTHONOPTIMIZE=$(NOASSERT) UB_NXN_N=2 MODULE=test_unified_buffer_nxn $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv unified_buffer_nxn.vcd waveforms/ 2>/dev/null || true

test_unified_buffer_nxn_n4: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s dump -g2012 -Pdump.N=4 $(SOURCES) test/dump_unified_buffer_nxn.sv
	PYTHONOPTIMIZE=$(NOASSERT) UB_NXN_N=4 MODULE=test_unified_buffer_nxn $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv unified_buffer_nxn.vcd waveforms/unified_buffer_nxn_n4.vcd 2>/dev/null || true

# Roadmap item 5c: VPU with per-lane array ports (src/vpu_nxn.sv), gate
# tests authored harness-side. _equiv: legacy and nxn instances driven in
# lockstep at N=2 across all seven pathway bits, compared cycle-by-cycle
# (the legacy module is the spec). _n4: exact per-lane sequences for the
# per-lane stages plus pairwise group-stage semantics at N=4 via iverilog
# -P override (VPU_NXN_N tells the cocotb test the compiled width). Also
# run plain at the default N=2.
test_vpu_nxn_equiv: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s dump -g2012 $(SOURCES) test/dump_vpu_nxn_equiv.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_vpu_nxn_equiv $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv vpu_nxn_equiv.vcd waveforms/ 2>/dev/null || true

test_vpu_nxn: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s dump -g2012 $(SOURCES) test/dump_vpu_nxn.sv
	PYTHONOPTIMIZE=$(NOASSERT) VPU_NXN_N=2 MODULE=test_vpu_nxn $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv vpu_nxn.vcd waveforms/ 2>/dev/null || true

test_vpu_nxn_n4: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s dump -g2012 -Pdump.N=4 $(SOURCES) test/dump_vpu_nxn.sv
	PYTHONOPTIMIZE=$(NOASSERT) VPU_NXN_N=4 MODULE=test_vpu_nxn $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv vpu_nxn.vcd waveforms/vpu_nxn_n4.vcd 2>/dev/null || true

# Roadmap item 5d-1a: N-column weight read schedule (generalized
# shared-pointer walk in src/unified_buffer_nxn.sv). Exact per-lane
# delivery at N=4 for 4x4 (R/T), 4x2, 4x3, 4x1 and 2x4T shapes, modeled
# by target semantics (per-column streams), not walk internals. The 5b
# tests anchor the C<=2 behavior bit-exactly.
test_unified_buffer_ncol_n4: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s dump -g2012 -Pdump.N=4 $(SOURCES) test/dump_unified_buffer_nxn.sv
	PYTHONOPTIMIZE=$(NOASSERT) UB_NXN_N=4 MODULE=test_unified_buffer_ncol $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv unified_buffer_nxn.vcd waveforms/unified_buffer_ncol_n4.vcd 2>/dev/null || true

test_unified_buffer_ncol_streams_n4: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s dump -g2012 -Pdump.N=4 $(SOURCES) test/dump_unified_buffer_nxn.sv
	PYTHONOPTIMIZE=$(NOASSERT) UB_NXN_N=4 MODULE=test_unified_buffer_ncol_streams $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv unified_buffer_nxn.vcd waveforms/unified_buffer_ncol_streams_n4.vcd 2>/dev/null || true

test_nn: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s nn -s dump -g2012 $(SOURCES) test/dump_nn.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_nn $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv nn.vcd waveforms/ 2>/dev/null || true

test_bias: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s bias -s dump -g2012 $(SOURCES) test/dump_bias.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_bias $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv bias.vcd waveforms/ 2>/dev/null || true

test_input_acc: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s input_acc -s dump -g2012 $(SOURCES) test/dump_input_acc.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_input_acc $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv input_acc.vcd waveforms/ 2>/dev/null || true

test_weight_acc: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s weight_acc -s dump -g2012 $(SOURCES) test/dump_weight_acc.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_weight_acc $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv weight_acc.vcd waveforms/ 2>/dev/null || true

test_cu: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s control_unit -s dump -g2012 $(SOURCES) test/dump_cu.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_cu $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv cu.vcd waveforms/ 2>/dev/null || true

test_unified_buffer: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s unified_buffer -s dump -g2012 $(SOURCES) test/dump_unified_buffer.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_unified_buffer $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv unified_buffer.vcd waveforms/ 2>/dev/null || true

# Loss module test

test_loss_child: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s loss_child -s dump -g2012 $(SOURCES) test/dump_loss_child.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_loss_child $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv loss_child.vcd waveforms/ 2>/dev/null || true

test_loss_parent: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s loss_parent -s dump -g2012 $(SOURCES) test/dump_loss_parent.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_loss_parent $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv loss_parent.vcd waveforms/ 2>/dev/null || true

# Leaky ReLU module tests
test_leaky_relu_child: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s leaky_relu_child -s dump -g2012 $(SOURCES) test/dump_leaky_relu_child.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_leaky_relu_child $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv leaky_relu_child.vcd waveforms/ 2>/dev/null || true

test_leaky_relu_parent: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s leaky_relu_parent -s dump -g2012 $(SOURCES) test/dump_leaky_relu_parent.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_leaky_relu_parent $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv leaky_relu_parent.vcd waveforms/ 2>/dev/null || true

# GELU module test (agent-authored RTL, harness-authored gate test)
test_gelu_parent: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s gelu_parent -s dump -g2012 $(SOURCES) test/dump_gelu_parent.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_gelu_parent $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv gelu_parent.vcd waveforms/ 2>/dev/null || true

# LayerNorm module test (agent-authored RTL, harness-authored gate test)
test_layernorm_parent: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s layernorm_parent -s dump -g2012 $(SOURCES) test/dump_layernorm_parent.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_layernorm_parent $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv layernorm_parent.vcd waveforms/ 2>/dev/null || true

# Softmax module test (agent-authored RTL, harness-authored gate test)
test_softmax_parent: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s softmax_parent -s dump -g2012 $(SOURCES) test/dump_softmax_parent.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_softmax_parent $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv softmax_parent.vcd waveforms/ 2>/dev/null || true

# Control unit (ISA decode) test (agent-authored RTL, harness-authored gate test)
test_control_unit: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s control_unit -s dump -g2012 $(SOURCES) test/dump_control_unit.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_control_unit $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv control_unit.vcd waveforms/ 2>/dev/null || true

# Full-chip softmax-pathway test via the widened tpu port (agent-authored
# RTL, harness-authored gate test)
test_tpu_softmax: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s tpu -s dump -g2012 $(SOURCES) test/dump_tpu_softmax.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_tpu_softmax $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv tpu_softmax.vcd waveforms/ 2>/dev/null || true

test_leaky_relu_derivative_child: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s leaky_relu_derivative_child -s dump -g2012 $(SOURCES) test/dump_leaky_relu_derivative_child.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_leaky_relu_derivative_child $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv leaky_relu_derivative_child.vcd waveforms/ 2>/dev/null || true

test_leaky_relu_derivative_parent: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s leaky_relu_derivative_parent -s dump -g2012 $(SOURCES) test/dump_leaky_relu_derivative_parent.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_leaky_relu_derivative_parent $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv leaky_relu_derivative_parent.vcd waveforms/ 2>/dev/null || true

# Bias module tests
test_bias_child: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s bias_child -s dump -g2012 $(SOURCES) test/dump_bias_child.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_bias_child $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv bias_child.vcd waveforms/ 2>/dev/null || true

test_bias_parent: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s bias_parent -s dump -g2012 $(SOURCES) test/dump_bias_parent.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_bias_parent $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv bias_parent.vcd waveforms/ 2>/dev/null || true

# Vector Processing unit test
test_vpu: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s vpu -s dump -g2012 $(SOURCES) test/dump_vpu.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_vpu $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv vpu.vcd waveforms/ 2>/dev/null || true

test_tpu: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s tpu -s dump -g2012 $(SOURCES) test/dump_tpu.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_tpu $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv tpu.vcd waveforms/ 2>/dev/null || true

test_gradient_descent: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s gradient_descent -s dump -g2012 $(SOURCES) test/dump_gradient_descent.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_gradient_descent $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv gradient_descent.vcd waveforms/ 2>/dev/null || true


# ============ DO NOT MODIFY BELOW THIS LINE ==============

# Create simulation build directory and waveforms directory
$(SIM_BUILD_DIR):
	mkdir -p $(SIM_BUILD_DIR)
	mkdir -p waveforms

# Waveform viewing
show_%: waveforms/%.vcd waveforms/%.gtkw
	gtkwave $^

# Linting
lint:
	verible-verilog-lint src/*sv --rules_config verible.rules

# Cleanup
clean:
	rm -rf waveforms/*vcd $(SIM_BUILD_DIR) test/__pycache__

.PHONY: clean

# Roadmap item 5d-2: full-chip top tpu_nxn (src/tpu_nxn.sv) integrating
# the three nxn children. Gate tests authored harness-side.
# _equiv: legacy tpu and tpu_nxn at N=2 share the full test_tpu.py
# instruction script (forward+backward+gradient descent); the two
# 128-word UB images are compared at the end (the legacy chip is the
# spec — including its beat-clip quirks). _n4: exact per-lane VPU
# output streams and UB placement for a 4x4 forward matmul (X @ W.T +
# bias, leaky_relu) with all-dyadic stimulus so the golden is exact.
test_tpu_nxn_equiv: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s dump -g2012 $(SOURCES) test/dump_tpu_nxn_equiv.sv
	PYTHONOPTIMIZE=$(NOASSERT) MODULE=test_tpu_nxn_equiv $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv tpu_nxn_equiv.vcd waveforms/ 2>/dev/null || true

test_tpu_nxn_n4: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s dump -g2012 -Pdump.N=4 $(SOURCES) test/dump_tpu_nxn.sv
	PYTHONOPTIMIZE=$(NOASSERT) TPU_NXN_N=4 MODULE=test_tpu_nxn_n4 $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv tpu_nxn.vcd waveforms/tpu_nxn_n4.vcd 2>/dev/null || true

test_tpu_nxn_train_n4: $(SIM_BUILD_DIR)
	$(IVERILOG) -o $(SIM_VVP) -s dump -g2012 -Pdump.N=4 $(SOURCES) test/dump_tpu_nxn.sv
	PYTHONOPTIMIZE=$(NOASSERT) TPU_NXN_N=4 MODULE=test_tpu_nxn_train_n4 $(VVP) -M $(COCOTB_LIBS) -m $(COCOTB_VPI_MODULE) $(SIM_VVP)
	python -c "f=open('results.xml').read();exit(1 if 'failure' in f else 0)"
	mv tpu_nxn.vcd waveforms/tpu_nxn_train_n4.vcd 2>/dev/null || true
