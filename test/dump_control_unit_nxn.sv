module dump();
initial begin
  $dumpfile("waveforms/control_unit_nxn.vcd");
  $dumpvars(0, control_unit_nxn);
end
endmodule
