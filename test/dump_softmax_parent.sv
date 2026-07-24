module dump();
initial begin
  $dumpfile("waveforms/softmax_parent.vcd");
  $dumpvars(0, softmax_parent);
end
endmodule
