clear;
clc;

[file, location] = uigetfile('*.edf');
filepath = fullfile(location, file);
timetable = edfread(filepath);