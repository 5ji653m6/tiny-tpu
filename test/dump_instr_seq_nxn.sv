module dump();
initial begin
  $dumpfile("waveforms/instr_seq_nxn.vcd");
  $dumpvars(0, instr_seq_nxn);
end
endmodule
