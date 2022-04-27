function read_val = read_serial(serial, timeout)
    tic
    while 1==1
        read_val = [];
        if serial.NumBytesAvailable > 0
            read_val = [read_val, readline(serial)];
            toc
            break;
        end   
        if toc > timeout
            read_val = -1;
            break; 
        end
    end