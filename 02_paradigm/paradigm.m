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

%% Setup timing of paradigm IN SECONDS
timings = struct;
% Time of the distance LED:
timings.imag_period = 2;
% Reaching duration:
timings.fb_rest = 5;
% Minimum pause duration
timings.min_break = 3;
% time variability of break
timings.break_variation = 1;


%% generic checks
N_classes = length(classes);

classes_nrTrials = repmat(trials_perclass, 1, N_classes);
N_trials = sum(classes_nrTrials);



% Connect to serial:


%% Start LSL stream:
% instantiate the library
disp('Loading library...');
lib = lsl_loadlib();

% make a new stream outlet
disp('Creating a new streaminfo...');
info = lsl_streaminfo(lib,'markers-paradigm','Marker',1,0,'cf_string','markers_paradigm123');

disp('Opening an outlet...');
markers_out = lsl_outlet(info);
