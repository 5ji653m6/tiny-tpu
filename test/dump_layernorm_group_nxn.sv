module dump();
initial begin
  $dumpfile("waveforms/layernorm_group_nxn.vcd");
  $dumpvars(0, layernorm_group_nxn);
end
endmodule
