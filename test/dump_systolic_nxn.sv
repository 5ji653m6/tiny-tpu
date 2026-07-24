module dump();
initial begin
  $dumpfile("waveforms/systolic_nxn.vcd");
  $dumpvars(1, systolic_nxn);
end
endmodule
