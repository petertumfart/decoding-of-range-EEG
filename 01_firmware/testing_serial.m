

s = serialport("COM8",115200,"Timeout",5);
s.Terminator;
configureTerminator(s,"CR");
pause(5)
read_val = []
while s.NumBytesAvailable > 0
    read_val = [read_val, readline(s)];
end

s=[];