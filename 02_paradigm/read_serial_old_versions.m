function read_val = read_serial_old_versions(serial, timeout)
    start = tic;
    while 1==1
        read_val = [];
        if serial.BytesAvailable > 0
            read_val = [read_val, fgetl(serial)];
            break;
        end   
        if toc(start) > timeout
            read_val = 'x';
            break; 
        end
    end