clear all
close all
clc

%% Defines:
classes = {'LTR-s', 'LTR-l','RTL-s', 'RTL-l', 'TTB-s', 'TTB-l', 'BTT-s', 'BTT-l'};
trials_perclass = 5;
Fs = 512; % Sampling rate

%fixed markers
markers = struct;
markers.break = {'Break'};
markers.cue = {'Cue'};
markers.start = {'Start'};
markers.classes = classes;
markers.release = {'Release'};
markers.touch = {'Touch'};

%% Setup timing of paradigm IN SECONDS
timings = struct;
% Time between indication and reach
timings.indication = 2;
% Reaching duration:
timings.reach = 5;
% Minimum pause duration
timings.min_break = 3;
% time variability of break
timings.break_variation = 1;


%% Create sound:
fs_sound = 8912;
f = 400;
dt = 1/fs_sound;
stop_time = 0.15;
t = 0:dt:stop_time;

beep_sine = sin(2*pi*f*t);


%% generic checks
N_classes = length(classes);

classes_nrTrials = repmat(trials_perclass, 1, N_classes);
N_trials = sum(classes_nrTrials);

%% Serial commands for params
cmds = ['a', 'b']; % led on, led off, read photoresistor
indicator_positions = ['l', 'l', 'r', 'r', 't', 't', 'b', 'b'];
reaching_positions = ['c', 'r', 'c', 'l', 'c', 'b', 'c', 't'];

%% Create parameter set of types of labels and number of trials per label and
% markers and cues
params = struct;

for iClass = 1: N_classes
    params(iClass).label = classes(iClass);
    params(iClass).trials = classes_nrTrials(iClass);
    params(iClass).marker = markers.classes(iClass);
    params(iClass).indicator_on = [cmds(1) ' ' indicator_positions(iClass)];
    params(iClass).indicator_off = [cmds(2) ' ' indicator_positions(iClass)];
    params(iClass).reaching_on = [cmds(1) ' ' reaching_positions(iClass)];
    params(iClass).reaching_off = [cmds(2) ' ' reaching_positions(iClass)];
    %params(iClass).cue = imread(char(strcat(folder_images,images(iClass))));
end

clearvars iClass

%% create pseudorandom order of trials
trial_labels = [];
trial_ind_on = [];
trial_ind_off = [];
trial_reach_on = [];
trial_reach_off = [];
for iClass = 1:N_classes
    trial_labels = [trial_labels; iClass*ones(params(iClass).trials,1)];
    trial_ind_on = [trial_ind_on; repmat(params(iClass).indicator_on,[5 1])];
    trial_ind_off = [trial_ind_off; repmat(params(iClass).indicator_off,[5 1])];
    trial_reach_on = [trial_reach_on; repmat(params(iClass).reaching_on,[5 1])];
    trial_reach_off = [trial_reach_off; repmat(params(iClass).reaching_off,[5 1])];
end

% check for 3 consecutive
trp = inf;
while trp ~= 0
    perm_idcs = randperm(length(trial_labels));
    trial_labels = trial_labels(perm_idcs);
    trp = 0;
    for iTrial = 4:N_trials
        if (trial_labels(iTrial) == trial_labels(iTrial-1)) && (trial_labels(iTrial-1) == trial_labels(iTrial-2))
            trp = trp + 1;
            break
        end
    end
end
trial_ind_on = trial_ind_on(perm_idcs,:);
trial_ind_off = trial_ind_off(perm_idcs,:);
trial_reach_on = trial_reach_on(perm_idcs,:);
trial_reach_off = trial_reach_off(perm_idcs,:);

%% Connect to serial:

%!!!!!! Switching between versions: 
% Matlab >= 2019b: 
% Uncomment block for matlab 2019b and greater
% Change fgetl(s) to readline(s)
% Change fprintf to writeline
% Change BytesAvailable to NumBytesAvailable
% Change read_serial_old_versions to read_serial
% Change read_val(5) to read_val{1}(5)
% https://www.mathworks.com/help/matlab/import_export/transition-your-code-to-serialport-interface.html#mw_18d02de5-6764-439f-8bc7-fca0f40522d3
% For old versions:
% Use: instrreset to release serial port when not propper closed
% Propper closing: for matlab <= 2019a: fclose(s); delete(s); clear s;



% Matlab 2019b and greater:
% s = serialport("COM8",115200,"Timeout",5);
% s.Terminator;
% configureTerminator(s,"CR");


% matlab 2019a and lower:
s = serial('COM5');
s.BaudRate = 115200;
s.Terminator = 'CR';
fopen(s);

pause(3);

% Wait until the initial command is received:
read_val = [];
while s.BytesAvailable > 0
    read_val = [read_val, fgetl(s)]; % readline(s) to fgetl(s)
end


%% Start LSL stream:
% instantiate the library
disp('Loading library...');
lib = lsl_loadlib();

% make a new stream outlet
disp('Creating a new streaminfo...');
info = lsl_streaminfo(lib,'markers-paradigm','Marker',1,0,'cf_string','markers_paradigm123');

disp('Opening an outlet...');
markers_out = lsl_outlet(info);


%% Ask for validation before starting
while(1)
    reply = input('Connect all streams and start recording in LabRecorder!!!\n Press Enter to continue:','s');
    if isempty(reply)
        break
    end
end
disp('Starting...');
pause(7);

%% Trials
t_overall = tic;
for iTrial = 1:N_trials
    trial_type = trial_labels(iTrial);
    markers_out.push_sample(markers.classes(trial_type));
    fprintf('trial: %d   class: %s \n',iTrial, char(classes(trial_type))  )
    
    %% INDICATION PERIOD:
    % Turn on the correct LED:
    fprintf(s,trial_ind_on(iTrial,:)); % writeline to sprintf
    % Get acknowledge:
    ret_val = read_serial_old_versions(s,1);
    if ret_val == 'x'
        fprintf('Error in serial communication!!!\n');
        break;
    end
    % Send start marker
    markers_out.push_sample(markers.start);
    % Wait between indication and reach and send marker when touched/released:
    t_start = tic;
    while toc(t_start) < timings.indication
        if s.BytesAvailable > 0
            read_val = {fgetl(s)};
            markers_out.push_sample(read_val);
%             if read_val(5) == '0'
%                 markers_out.push_sample(read_val);
%                 disp('Touch')
%             else
%                 markers_out.push_sample(read_val);
%                 disp('Release')
%             end
        end
    end
    disp(toc(t_start))
    
    %% REACHING PERIOD:
    % Turn off the indication LED:
    fprintf(s,trial_ind_off(iTrial,:));
    % Get acknowledge:
    ret_val = read_serial_old_versions(s,1);
    if ret_val == 'x'
        fprintf('Error in serial communication!!!\n');
        break;
    end
    % Turn on the reaching LED:
    fprintf(s,trial_reach_on(iTrial,:));
    % Get acknowledge:
    ret_val = read_serial_old_versions(s,1);
    if ret_val == 'x'
        fprintf('Error in serial communication!!!\n');
        break;
    end
    % Send reaching cue is shown marker:
    markers_out.push_sample(markers.cue);
    % Wait for reaching period and send marker when touched/released:
    t_start = tic;
    while toc(t_start) < timings.reach
        if s.BytesAvailable > 0
            read_val = {fgetl(s)};
            markers_out.push_sample(read_val);
%             if read_val(5) == '0'
%                 markers_out.push_sample(read_val);
%                 disp('Touch')
%             else
%                 markers_out.push_sample(read_val);
%                 disp('Release')
%             end
        end
    end
    disp(toc(t_start))
    
    % Turn off reaching LED:
    fprintf(s,trial_reach_off(iTrial,:));
    % Get acknowledge:
    ret_val = read_serial_old_versions(s,1);
    if ret_val == 'x'
        fprintf('Error in serial communication!!!\n');
        break;
    end
    
    %% BREAK PERIOD:
    % Signalise end with beep:
    sound(beep_sine, fs_sound);
    % Send break indication marker:
    markers_out.push_sample(markers.break);
    % Wait for break period and send marker when touched/released:
    break_time = timings.min_break + timings.break_variation * rand;
    t_start = tic;
    while toc(t_start) < break_time
        if s.BytesAvailable > 0
            read_val = {fgetl(s)};
            markers_out.push_sample(read_val);
%             if read_val(5) == '0'
%                 markers_out.push_sample(read_val);
%                 disp('Touch')
%             else
%                 markers_out.push_sample(read_val);
%                 disp('Release')
%             end
        end
    end
    disp(toc(t_start)) 
    

end
disp(toc(t_overall));
disp('End of Experiment. Please stop recording.');

%%
clear lib
pause(0.5)
clear markers_out

% For matlab <= 2019a
fclose(s)
delete(s)


clear s

