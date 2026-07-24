module dump();
initial begin
  // Depth 1 only: tpu top ports. Depth-0 (full subtree) dumps ~2GB VCDs
  // for full-chip sims (cf. dump_tpu.sv) and slows the sim ~25x; the gate
  // test only needs port-level visibility, and a deeper dump can be
  // re-created on demand for debug (see /data1/chia-chip-design/diag/).
  $dumpfile("waveforms/tpu_softmax.vcd");
  $dumpvars(1, tpu);
end
endmodule
