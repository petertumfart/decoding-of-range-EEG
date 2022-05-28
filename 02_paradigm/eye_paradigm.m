clear all
close all
clc

%% Defines:
classes = {'Rest', 'Horz','Vert', 'Blink'};
trials_perclass = 9;
Fs = 512; % Sampling rate

%fixed markers
markers = struct;
markers.break = {'Break'};
markers.cue = {'Cue'};
markers.start = {'Start'};
markers.classes = classes;

%% Setup timing of paradigm IN SECONDS
timings = struct;
% Time between indication and execution
timings.indication = 1;
% Execution duration:
timings.exec = 10;
% Minimum pause duration
timings.min_break = 2;
% time variability of break
timings.break_variation = 1;


%% Load sounds:
path = 'sounds\';

[y_rest, Fs_rest] = audioread([path, 'rest.mp3']);
[y_horz, Fs_horz] = audioread([path, 'horizontal.mp3']);
[y_vert, Fs_vert] = audioread([path, 'vertical.mp3']);
[y_blink, Fs_blink] = audioread([path, 'blink.mp3']);

sounds = {y_rest, y_horz, y_vert, y_blink};
Fs_sounds = (Fs_rest + Fs_horz + Fs_vert + Fs_blink)/4; 

% sound(y_rest,Fs_rest);
% sound(y_horz, Fs_horz);
% sound(y_vert, Fs_vert);
% sound(y_blink, Fs_blink);

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

%% Create parameter set of types of labels and number of trials per label and
% markers and cues
params = struct;

for iClass = 1: N_classes
    params(iClass).label = classes(iClass);
    params(iClass).trials = classes_nrTrials(iClass);
    params(iClass).marker = markers.classes(iClass);
end

clearvars iClass

%% create pseudorandom order of trials
trial_labels = [];
for iClass = 1:N_classes
    trial_labels = [trial_labels; iClass*ones(params(iClass).trials,1)];
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


%% Start LSL stream:
% instantiate the library
disp('Loading library...');
lib = lsl_loadlib();

% make a new stream outlet
disp('Creating a new streaminfo...');
info = lsl_streaminfo(lib,'markers-eye-paradigm','Marker',1,0,'cf_string','markers_paradigm2');

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
t_start = tic;
for iTrial = 1:N_trials
    trial_type = trial_labels(iTrial);
    markers_out.push_sample(markers.classes(trial_type));
    fprintf('trial: %d   class: %s \n',iTrial, char(classes(trial_type))  )
    
    %% INDICATION PERIOD:
    % Play the correct sound:
    sound(sounds{trial_labels(iTrial)}, Fs_sounds);
    % Send start marker
    markers_out.push_sample(markers.start);
    % Wait between indication and execution and send marker when touched/released:
    pause(timings.indication);
    
    %% Exectution PERIOD:
    % Beep to signalise start:
%     sound(beep_sine, fs_sound);
    % Send start of trial marker:
    markers_out.push_sample(markers.cue);
    % Wait for reaching period and send marker when touched/released:
    pause(timings.exec);
    
    %% BREAK PERIOD:
    % Signalise end with beep:
    sound(beep_sine, fs_sound);
    % Send break indication marker:
    markers_out.push_sample(markers.break);
    % Wait for break period and send marker when touched/released:
    break_time = timings.min_break + timings.break_variation * rand;
    pause(break_time);
end
disp(toc(t_start))
display('End of Experiment. Please stop recording.');

%%
clear lib
pause(0.5)
clear markers_out
