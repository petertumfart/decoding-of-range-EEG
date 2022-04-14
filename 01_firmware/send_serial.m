function m = send_serial(serial, cmd, pos)
    str_to_write = [cmd ' ' pos];
    
    writeline(serial,str_to_write)
    tic
    while 1==1
        m = [];
        if serial.NumBytesAvailable > 0
            m = [m, readline(serial)];
            toc
            break;
        end   
    end
end