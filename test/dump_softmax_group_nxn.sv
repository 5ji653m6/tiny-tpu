module dump();
initial begin
  $dumpfile("waveforms/softmax_group_nxn.vcd");
  $dumpvars(0, softmax_group_nxn);
end
endmodule
