module dump();
initial begin
  $dumpfile("waveforms/gelu_parent.vcd");
  $dumpvars(0, gelu_parent);
end
endmodule
