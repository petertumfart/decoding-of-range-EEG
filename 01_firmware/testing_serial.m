

s = serialport("COM8",115200,"Timeout",5);
s.Terminator;
configureTerminator(s,"CR");

% Wait until the initial command is received:
while 1 == 1
    read_val = [];
    if s.NumBytesAvailable > 0
        read_val = [read_val, readline(s)];
        break;
    end
end

commands = ['a', 'b', 'c']; % led on, led off, read photoresistor
positions = ['l', 'r', 'c', 't', 'b']; % left, right, center, top, bottom

ret = send_serial(s, commands(3), positions(5));


s=[];