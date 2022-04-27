function str_to_write = send_serial(serial, cmd, pos)
    str_to_write = [cmd ' ' pos];
    
    % Send command to serial:
    writeline(serial,str_to_write)
end