function read_val = read_serial(serial, timeout)
    start = tic;
    while 1==1
        read_val = [];
        if serial.NumBytesAvailable > 0
            read_val = [read_val, readline(serial)];
            break;
        end   
        if toc(start) > timeout
            read_val = 'x';
            break; 
        end
    end