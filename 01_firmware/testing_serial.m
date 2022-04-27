s = serialport("COM8",115200,"Timeout",5);
s.Terminator;
configureTerminator(s,"CR");

pause(3);

% Wait until the initial command is received:
read_val = [];
while s.NumBytesAvailable > 0
    read_val = [read_val, readline(s)];
end

commands = ['a', 'b']; % led on, led off, read photoresistor
positions = ['l', 'r', 'c', 't', 'b']; % left, right, center, top, bottom

send_serial(s, commands(2), positions(5));
ret_val = read_serial(s, 1);



% Initialization:
send_serial(s, commands(1), positions(1));
pause(0.2)
send_serial(s, commands(2), positions(1));
send_serial(s, commands(1), positions(4));
pause(0.2)
send_serial(s, commands(2), positions(4));
send_serial(s, commands(1), positions(2));
pause(0.2)
send_serial(s, commands(2), positions(2));
send_serial(s, commands(1), positions(5));
pause(0.2)
send_serial(s, commands(2), positions(5));

send_serial(s, commands(1), positions(1));
pause(0.2)
send_serial(s, commands(1), positions(4));
pause(0.2)
send_serial(s, commands(1), positions(2));
pause(0.2)
send_serial(s, commands(1), positions(5));
pause(0.2)
send_serial(s, commands(1), positions(3));
pause(0.2)
send_serial(s, commands(2), positions(3)); 
pause(0.2)
send_serial(s, commands(1), positions(3));
pause(5)
send_serial(s, commands(2), positions(1)); 
send_serial(s, commands(2), positions(2)); 
send_serial(s, commands(2), positions(3)); 
send_serial(s, commands(2), positions(4)); 
send_serial(s, commands(2), positions(5)); 

flush(s);







s=[];