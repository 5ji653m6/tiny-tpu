module dump();
initial begin
  $dumpfile("waveforms/silu_parent.vcd");
  $dumpvars(0, silu_parent);
end
endmodule
