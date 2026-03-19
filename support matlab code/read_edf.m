clear;
clc;

[file, location] = uigetfile('*.edf');
filepath = fullfile(location, file);
timetable = edfread(filepath);

table = timetable2table(timetable);

ECG = vertcat(table.EKG{:});

fs = 500;

lowcut = 5.0;
highcut = 15.0;
nyq = 0.5 * fs;
low = lowcut / nyq;
high = highcut / nyq;
order = 2;
[b,a] = butter(order, [low high], 'bandpass');

ECG_filt = filtfilt(b,a, ECG);

f1 = subplot(211);
plot(ECG);

f2 = subplot(212);
plot(ECG_filt)

linkaxes([f1 f2], 'x');
