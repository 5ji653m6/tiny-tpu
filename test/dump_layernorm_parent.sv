module dump();
initial begin
  $dumpfile("waveforms/layernorm_parent.vcd");
  $dumpvars(0, layernorm_parent);
end
endmodule
