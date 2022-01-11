% --------------------------------------------------------------------
function import_raven_Callback(hObject, eventdata, handles)
% Requires a Raven table and audio file.
% (http://www.birds.cornell.edu/brp/raven/RavenOverview.html)

%% Get the files
[ravenname,ravenpath] = uigetfile([handles.data.squeakfolder '/*.txt'],'Select Raven Log - Can Select Multiple','MultiSelect','on');
audiopath = uigetdir(ravenpath,'Select Directory Containing Corresponding Audio Files');
audiodir = [dir([audiopath '\*.wav']); ...
    dir([audiopath '\*.ogg']); ...
    dir([audiopath '\*.flac']); ...
    dir([audiopath '\*.UVD']); ...
    dir([audiopath '\*.au']); ...
    dir([audiopath '\*.aiff']); ...
    dir([audiopath '\*.aif']); ...
    dir([audiopath '\*.aifc']); ...
    dir([audiopath '\*.mp3']); ...
    dir([audiopath '\*.m4a']); ...
    dir([audiopath '\*.mp4'])];
outpath = uigetdir(ravenpath,'Select Directory To Save Output Files (WARNING: Will Overwrite)');

if ischar(ravenname)
    ravenname = {ravenname};
end
audioname = cell(size(ravenname));
for i = 1:length(ravenname)
    [datetime,~] = regexp(ravenname{i},'([0-9]{6}).*([0-9]{6})','tokens','match');
    if isempty(datetime)
        [datetime,~] = regexp(ravenname{i},'([0-9]{6}).*([0-9]{4})','tokens','match');
    end
    if ~isempty(datetime)
        audiomatch = regexp({audiodir.name},['.*' datetime{1}{1} '.*' datetime{1}{2} '.*'],'match');
        audiomatch = ~cellfun(@isempty, audiomatch);
        if ~any(audiomatch)
            audiomatch = regexp({audiodir.name},['.*' datetime{1}{1} '.*' datetime{1}{2}(1:4) '.*'],'match');
            audiomatch = ~cellfun(@isempty, audiomatch);
        end
    end
    if isempty(datetime) || length(find(audiomatch)) ~= 1
        warning('Could not automatically match all wav files to txt files - you will have to do them one-by-one')
        [ravenname,ravenpath] = uigetfile([handles.data.squeakfolder '/*.txt'],'Select Raven Log');
        %ravenTable = readtable([ravenpath ravenname], 'Delimiter', 'tab');
        [audioname, audiopath] = uigetfile({
            '*.wav;*.ogg;*.flac;*.UVD;*.au;*.aiff;*.aif;*.aifc;*.mp3;*.m4a;*.mp4' 'Audio File'
            '*.wav' 'WAVE'
            '*.flac' 'FLAC'
            '*.ogg' 'OGG'
            '*.UVD' 'Ultravox File'
            '*.aiff;*.aif', 'AIFF'
            '*.aifc', 'AIFC'
            '*.mp3', 'MP3 (it''s probably a bad idea to record in MP3'
            '*.m4a;*.mp4' 'MPEG-4 AAC'
            }, 'Select Audio File',handles.data.settings.audiofolder);
        audioname = {audioname};
        ravenname = {ravenname};
        break;
    else
        audioname{i} = audiodir(audiomatch).name;
    end
end

for i = 1:length(ravenname)
	ravenTable = readtable([ravenpath ravenname{i}], 'Delimiter', 'tab');
    audiodata = audioinfo(fullfile(audiopath, audioname{i}));
    if audiodata.NumChannels > 1
        warning('Audio file contains more than one channel. Use channel 1...')
    end
    hc = waitbar(0,'Importing Calls from Raven Log');

    % fix some compatibility issues with Raven's naming
    if ~ismember('DeltaTime_s_', ravenTable.Properties.VariableNames)
        ravenTable.DeltaTime_s_ = ravenTable.EndTime_s_ - ravenTable.BeginTime_s_;
    end

    %% Get the data from the raven file
    Box    = [ravenTable.BeginTime_s_, ravenTable.LowFreq_Hz_/1000, ravenTable.DeltaTime_s_, (ravenTable.HighFreq_Hz_ - ravenTable.LowFreq_Hz_)/1000];
    Score  = ones(height(ravenTable),1);
    Accept = ones(height(ravenTable),1);

    %% Get the classification from raven, from the variable 'Tags' or 'Annotation'
    if ismember('Tags', ravenTable.Properties.VariableNames)
        Type = categorical(ravenTable.Tags);
    elseif ismember('Annotation', ravenTable.Properties.VariableNames)
        Type = categorical(ravenTable.Annotation);
    else
        Type = categorical(repmat({'USV'}, height(ravenTable), 1));
    end

    %% Put all the variables into a table
    Calls = table(Box,Score,Accept,Type,'VariableNames',{'Box','Score','Accept','Type'});

    [~ ,FileName] = fileparts(audioname{i});
    %[FileName, PathName] = uiputfile(fullfile(handles.data.settings.detectionfolder, [name '.mat']),'Save Call File');
    save(fullfile(outpath,FileName),'Calls', 'audiodata','-v7.3');
    close(hc);
end
update_folders(hObject, eventdata, handles);
