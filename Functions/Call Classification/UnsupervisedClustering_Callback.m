function UnsupervisedClustering_Callback(hObject, eventdata, handles)
% Cluster with k-means or adaptive

finished = 0; % Repeated until
while ~finished
    choice = questdlg('Choose clustering method:','Cluster','ARTwarp','K-means (recommended)', 'Variational Autoencoder','K-means (recommended)');
    
    % Get the data
    %     [ClusteringData] = CreateClusteringData(handles, 'forClustering', true, 'save_data', true);
    %     if isempty(ClusteringData); return; end
    %     clustAssign = zeros(size(ClusteringData,1),1);
    
    
    switch choice
        case []
            return
            
        case {'K-means (recommended)', 'Variational Autoencoder'}
            FromExisting = questdlg('From existing model?','Cluster','Yes','No','No');
            switch FromExisting % Load Model
                case 'No'
                    % Get parameter weights
                    switch choice
                        case 'K-means (recommended)'
                            [ClusteringData, ~, ~, ~, spectrogramOptions] = CreateClusteringData(handles, 'forClustering', true, 'save_data', true);
                            if isempty(ClusteringData); return; end
                            clusterParameters= inputdlg({'Number of Contour Pts','Shape weight','Frequency weight', ...
                                'Relative Frequency weight','Duration weight','Parsons Resolution','Parsons weight'}, ...%,'Parsons2 weight'},
                                'Choose cluster parameters:',1,{'20','0','0','1','0','4','0'});%,'0'});
                            if isempty(clusterParameters); return; end
                            num_pts = str2double(clusterParameters{1});
                            slope_weight = str2double(clusterParameters{2});
                            freq_weight = str2double(clusterParameters{3});
                            relfreq_weight = str2double(clusterParameters{4});
                            duration_weight = str2double(clusterParameters{5});
                            RES = str2double(clusterParameters{6});
                            pc_weight = str2double(clusterParameters{7});
                            %pc2_weight = str2double(clusterParameters{8});
                            ClusteringData{:,'NumContPts'} = num_pts;
                            data = get_kmeans_data(ClusteringData, num_pts, RES, slope_weight, freq_weight, relfreq_weight, duration_weight, pc_weight);%, pc2_weight);
                        case 'Variational Autoencoder'
                            [encoderNet, decoderNet, options, ClusteringData] = create_VAE_model(handles);
                            data = extract_VAE_embeddings(encoderNet, options, ClusteringData);
                    end
                    
                    % Make a k-means model and return the centroids
                    C = get_kmeans_centroids(data);
                    if isempty(C); return; end
                    
                case 'Yes'
                    [FileName,PathName] = uigetfile(fullfile(handles.data.squeakfolder,'Clustering Models','*.mat'));
                    if isnumeric(FileName); return;end
                    switch choice
                        case 'K-means (recommended)'
                            spectrogramOptions = [];
                            load(fullfile(PathName,FileName),'C','num_pts',...
                                'RES','freq_weight','relfreq_weight','slope_weight',...
                                'duration_weight','pc_weight',...%'pc2_weight',...
                                'clusterName','spectrogramOptions');
                            ClusteringData = CreateClusteringData(handles, 'forClustering', true, 'spectrogramOptions', spectrogramOptions, 'save_data', true);
                            if isempty(ClusteringData); return; end
                            % Set number of contour pts to default 12 if it
                            % didn't load as a variable
                            if exist('num_pts','var') ~= 1
                                num_pts = 12;
                            end
                            if exist('RES','var') ~= 1
                                RES = 1;
                            end
                            if exist('pc_weight','var') ~= 1
                                pc_weight = 0;
                            end
                            ClusteringData{:,'NumContPts'} = num_pts;
                            data = get_kmeans_data(ClusteringData, num_pts, RES, slope_weight, freq_weight, relfreq_weight, duration_weight, pc_weight);%, pc2_weight);
                        case 'Variational Autoencoder'
                            C = [];
                            load(fullfile(PathName,FileName),'C','encoderNet','decoderNet','options');
                            [ClusteringData] = CreateClusteringData(handles, 'spectrogramOptions', options.spectrogram, 'scale_duration', options.maxDuration, 'freqRange', options.freqRange, 'save_data', true);
                            if isempty(ClusteringData); return; end
                            data = extract_VAE_embeddings(encoderNet, options, ClusteringData);
                            
                            % If the model was created through create_tsne_Callback, C won't exist, so make it.
                            if isempty(C)
                                C = get_kmeans_centroids(data);
                            end
                    end
            end
            
            [clustAssign,D] = knnsearch(C,data,'Distance','euclidean');
            
            ClusteringData.DistToCen = D;
            ClusteringData.ClustAssign = clustAssign;
            
            %% Save contour used in ClusteringData
            contourfreqsl = cellfun(@(x) {imresize(x',[1 num_pts+1])}, ClusteringData.xFreq,'UniformOutput',0);
            contourtimesl = cellfun(@(x) {imresize(x,[1 num_pts+1])}, ClusteringData.xTime,'UniformOutput',0);
            contourfreq = cellfun(@(x) {imresize(x',[1 num_pts])}, ClusteringData.xFreq,'UniformOutput',0);
            contourtime = cellfun(@(x) {imresize(x,[1 num_pts])}, ClusteringData.xTime,'UniformOutput',0);
            
            ClusteringData(:,'xFreq_Contour_Sl') = contourfreqsl;
            ClusteringData(:,'xTime_Contour_Sl') = contourtimesl;
            ClusteringData(:,'xFreq_Contour') = contourfreq;
            ClusteringData(:,'xTime_Contour') = contourtime;
            
            % Save the cluster assignments
            saveChoice =  questdlg('Save Extracted Contours with Cluster Assignments?','Save cluster assignments','Yes','No','No');
            switch saveChoice
                case 'Yes'
                    CDBU = ClusteringData;
                    ClusteringData{:,'ClustAssign'} = clustAssign;
                    pind = regexp(char(ClusteringData{1,'Filename'}),'\');
                    pind = pind(end);
                    pname = char(ClusteringData{1,'Filename'});
                    pname = pname(1:pind);
                    [FileName,PathName] = uiputfile(fullfile(pname,'Extracted Contours.mat'),'Save contours with cluster assignments');
                    if FileName ~= 0
                        save(fullfile(PathName,FileName),'ClusteringData','-v7.3');
                    end
                    ClusteringData = CDBU;
                    clear CDBU
                case 'No'
            end
            
            % Save PC?
            saveChoice =  questdlg('Save Extracted Contours with Parsons Code?','Save PC','Yes','No','No');
            switch saveChoice
                case 'Yes'
                    CDBU = ClusteringData;
                    ReshapedX   = cell2mat(cellfun(@(x) imresize(x',[1 num_pts+1]) ,ClusteringData.xFreq,'UniformOutput',0));
                    slope       = diff(ReshapedX,1,2);
                    MX          = (max(slope,[],'all')/(RES+1))*RES;
                    pc          = round(slope.*(RES/MX));
                    pc(pc>RES)  = RES;
                    pc(pc<-RES) = -RES;
                    ClusteringData{:,'Parsons'} = pc;
                    pind = regexp(char(ClusteringData{1,'Filename'}),'\');
                    pind = pind(end);
                    pname = char(ClusteringData{1,'Filename'});
                    pname = pname(1:pind);
                    [FileName,PathName] = uiputfile(fullfile(pname,'Extracted Contours.mat'),'Save contours with Parsons Code');
                    if FileName ~= 0
                        save(fullfile(PathName,FileName),'ClusteringData','-v7.3');
                    end
                    ClusteringData = CDBU;
                    clear CDBU
                case 'No'
            end
                        
            %% Centroid contours
            if relfreq_weight > 0 && sum([slope_weight, freq_weight, duration_weight, pc_weight]) == 0
                % Generate relative frequencies
                allrelfreq = ClusteringData.xFreq_Contour;
                allrelfreq = cell2mat(allrelfreq);
                allrelfreq = allrelfreq-allrelfreq(:,1);
                allrelfreq = zscore(allrelfreq,0,'all');
                
                minylim = min(allrelfreq,[],'all');
                maxylim = max(allrelfreq,[],'all');
                
                % Make the figure
                figure('Color','w','Position',[50,50,800,800]);
                montTile = tiledlayout('flow','TileSpacing','none');
                
                for i = unique(clustAssign)'
                    thisclust = allrelfreq(ClusteringData.ClustAssign == i,:);
                    thiscent = C(i,num_pts+1:2*num_pts);
                
                    maxcont = max(thisclust,[],1);
                    mincont = min(thisclust,[],1);
                    
                    nexttile
                    plot(1:num_pts,thiscent,1:num_pts,maxcont,'r--',1:num_pts,mincont,'r--')
                    ylim([minylim maxylim])
                end
                
                title(montTile, 'Centroid Contours with Max and Min Call Variation')
            end
            
            %% Sort the calls by how close they are to the cluster center
            [~,idx] = sort(D);
            clustAssign = clustAssign(idx);
            ClusteringData = ClusteringData(idx,:);
            
            %% Make a montage with the top calls in each class
            try
                % Find the median call length
                [~, i] = unique(clustAssign,'sorted');
                maxlength = cellfun(@(spect) size(spect,2), ClusteringData.Spectrogram(i));
                maxlength = round(prctile(maxlength,75));
                maxBandwidth = cellfun(@(spect) size(spect,1), ClusteringData.Spectrogram(i));
                maxBandwidth = round(prctile(maxBandwidth,75));
                
                % Make the figure
                figure('Color','w','Position',[50,50,800,800]);
%                 ax_montage = axes(f_montage);
                % Make the image stack
                %montageI = [];
                montTile = tiledlayout('flow','TileSpacing','none');
                for i = unique(clustAssign)'
                    index = find(clustAssign==i,1);
                    tmp = ClusteringData.Spectrogram{index,1};
                    tmp = padarray(tmp,[0,max(maxlength-size(tmp,2),0)],'both');
                    tmp = rescale(tmp,1,256);
                    %montageI(:,:,i) = floor(imresize(tmp,[maxBandwidth,maxlength]));
                    
                    nexttile
                    image(imtile(floor(imresize(tmp,[maxBandwidth,maxlength])), inferno, 'BackgroundColor', 'w', 'GridSize',[1 1]))
                    title(num2str(i))
                    axis off
                end
%                 image(ax_montage, imtile(montageI, inferno, 'BackgroundColor', 'w', 'BorderSize', 2, 'GridSize',[5 NaN]))
%                 axis(ax_montage, 'off')
                title(montTile, 'Closest call to each cluster center')
            catch
                disp('For some reason, I couldn''t make a montage of the call exemplars')
            end
            
            %Undo sort
            clustAssign(idx) = clustAssign;
            ClusteringData(idx,:) = ClusteringData;
            
        case 'ARTwarp'
            ClusteringData = CreateClusteringData(handles, 'forClustering', true, 'save_data', true);
            if isempty(ClusteringData); return; end
            FromExisting = questdlg('From existing model?','Cluster','Yes','No','No');
            switch FromExisting% Load Art Model
                case 'No'
                    %% Get settings
                    prompt = {'Matching Threshold:','Duplicate Category Merge Threshold:','Outlier Threshold','Learning Rate:','Interations:','Shape Importance','Frequency Importance','Duration Importance'};
                    dlg_title = 'ARTwarp';
                    num_lines = [1 50];
                    defaultans = {'5','2.5','8','0.001','5','4','1','1'};
                    settings = inputdlg(prompt,dlg_title,num_lines,defaultans);
                    if isempty(settings)
                        return
                    end
                    %% Cluster
                    try
                        [ARTnet, clustAssign] = ARTwarp2(ClusteringData.xFreq,settings);
                    catch ME
                        disp(ME)
                    end
                    
                case 'Yes'
                    [FileName,PathName] = uigetfile(fullfile(handles.data.squeakfolder,'Clustering Models','*.mat'));
                    load(fullfile(PathName,FileName),'ARTnet','settings');
                    if exist('ARTnet', 'var') ~= 1
                        warndlg('ARTnet model could not be found. Is this file a trained ARTwarp2 model?')
                        continue
                    end
                    
            end
            [clustAssign] = GetARTwarpClusters(ClusteringData.xFreq,ARTnet,settings);
    end
    
    %     data = freq;
    %         epsilon = 0.0001;
    % mu = mean(data);
    % data = data - mean(data)
    % A = data'*data;
    % [V,D,~] = svd(A);
    % whMat = sqrt(size(data,1)-1)*V*sqrtm(inv(D + eye(size(D))*epsilon))*V';
    % Xwh = data*whMat;
    % invMat = pinv(whMat);
    %
    % data = Xwh
    %
    % data  = (freq-mean(freq)) ./ std(freq)
    % [clustAssign, C]= kmeans(data,10,'Distance','sqeuclidean','Replicates',10);
    
    
    %% Assign Names
    % If the
    if strcmp(choice, 'K-means (recommended)') && strcmp(FromExisting, 'Yes')
        clustAssign = categorical(clustAssign, 1:size(C,1), cellstr(clusterName));
    end
    
    % Standardize clustering GUI image axes?
    saveChoice =  questdlg('Standardize clustering GUI image axes?','Standardize axes','Yes','No','No');
    switch saveChoice
        case 'Yes'
            CDBU = ClusteringData;
            %ClusteringData{:,'StandSpec'} = ClusteringData{:,'Spectrogram'};
            if length(unique(ClusteringData.TimeScale)) > 1
                warning('%s\n%s\n%s',...
                    'WARNING: It looks like the spectrograms in this collection were not run consistently.',...
                    'This may be because you are loading multiple Extracted Contours that were run separately.',...
                    'Recommend running the original detection mats instead or the Clustering GUI images may look weird.')
                bProceed = questdlg('Do you wish to proceed anyway?','Yes','No','No');
                if strcmp(bProceed,'No')
                    error('You chose to stop.')
                end
            end
            CDDurs = cell2mat(cellfun(@(x) size(x,2),ClusteringData.Spectrogram,'UniformOutput',false)).*ClusteringData.TimeScale;
            %resz = max(cell2mat(cellfun(@size,ClusteringData.Spectrogram,'UniformOutput',false)));
            pad = [zeros(size(CDDurs,1),1) max(CDDurs)-CDDurs];
            pad = floor(pad./ClusteringData.TimeScale);
            pad = num2cell(pad,2);
            ClusteringData.Spectrogram = cellfun(@(x,y) padarray(x, y, 255, 'post'),ClusteringData.Spectrogram,pad,'UniformOutput',false);
            [~, clusterName, rejected, finished, clustAssign] = clusteringGUI(clustAssign, ClusteringData);
            ClusteringData = CDBU;
            clear CDBU CDDurs pad
        case 'No'
            [~, clusterName, rejected, finished, clustAssign] = clusteringGUI(clustAssign, ClusteringData);%, ...
            %[str2double(handles.data.settings.detectionSettings{3}) str2double(handles.data.settings.detectionSettings{2})]);
    end
    
    
end
%% Update Files
% Save the clustering model
if FromExisting(1) == 'N'
    switch choice
        case 'K-means (recommended)'
            pind = regexp(char(ClusteringData{1,'Filename'}),'\');
            pind = pind(end);
            pname = char(ClusteringData{1,'Filename'});
            pname = pname(1:pind);
            [FileName, PathName] = uiputfile(fullfile(pname, 'K-Means Model.mat'), 'Save clustering model');
            %[FileName, PathName] = uiputfile(fullfile(handles.data.squeakfolder, 'Clustering Models', 'K-Means Model.mat'), 'Save clustering model');
            if ~isnumeric(FileName)
                save(fullfile(PathName, FileName), 'C', 'num_pts','RES','freq_weight',...
                    'relfreq_weight', 'slope_weight', 'duration_weight', 'pc_weight',... % 'pc2_weight',
                    'clusterName', 'spectrogramOptions');
            end
        case 'ARTwarp'
            [FileName, PathName] = uiputfile(fullfile(handles.data.squeakfolder, 'Clustering Models', 'ARTwarp Model.mat'), 'Save clustering model');
            if ~isnumeric(FileName)
                save(fullfile(PathName, FileName), 'ARTnet', 'settings');
            end
        case 'Variational Autoencoder'
            [FileName, PathName] = uiputfile(fullfile(handles.data.squeakfolder, 'Clustering Models', 'Variational Autoencoder Model.mat'), 'Save clustering model');
            if ~isnumeric(FileName)
                save(fullfile(PathName, FileName), 'C', 'encoderNet', 'decoderNet', 'options', 'clusterName');
            end
    end
end

% Save the clusters
saveChoice =  questdlg('Update files with new clusters?','Save clusters','Yes','No','No');
switch saveChoice
    case 'Yes'
        UpdateCluster(ClusteringData, clustAssign, clusterName, rejected)
        update_folders(hObject, eventdata, handles);
        if isfield(handles,'current_detection_file')
            loadcalls_Callback(hObject, eventdata, handles, true)
        end
    case 'No'
        return
end
end

%% Dyanamic Time Warping
% for use as a custom distance function for pdist, kmedoids
function D = dtw2(ZI,ZJ)
D = zeros(size(ZJ,1),1);
for i = 1:size(ZJ,1)
    D(i) = dtw(ZI,ZJ(i,:),3);
end
end

function data = get_kmeans_data(ClusteringData, num_pts, RES, slope_weight, freq_weight, relfreq_weight, duration_weight, pc_weight)%, pc2_weight)
% Parameterize the data for kmeans
ReshapedX   = cell2mat(cellfun(@(x) imresize(x',[1 num_pts+1]) ,ClusteringData.xFreq,'UniformOutput',0));
slope       = diff(ReshapedX,1,2);
%MX          = quantile(slope,0.9,'all');
%MX          = 2*std(slope,0,'all');
%MX          = max(slope,[],'all');
MX          = (max(slope,[],'all')/(RES+1))*RES;
pc          = round(slope.*(RES/MX));
pc(pc>RES)  = RES;
pc(pc<-RES) = -RES;
slope       = zscore(slope,0,'all');
freq        = cell2mat(cellfun(@(x) imresize(x',[1 num_pts]) ,ClusteringData.xFreq,'UniformOutput',0));
relfreq     = freq-freq(:,1);

% MX2         = (max(relfreq,[],'all')/(RES+1))*RES;
% pc2          = round(relfreq.*(RES/MX2));
% pc2(pc2>RES)  = RES;
% pc2(pc2<-RES) = -RES;

freq        = zscore(freq,0,'all');
relfreq     = zscore(relfreq,0,'all');
duration    = repmat(ClusteringData.Duration,[1 num_pts]);
duration    = zscore(duration,0,'all');
pc          = zscore(pc,0,'all');
% pc2       = zscore(pc2,0,'all');

data = [
    freq     .*  freq_weight+.001,...
    relfreq     .*  relfreq_weight+.001,...
    slope    .*  slope_weight+.001,...
    duration .*  duration_weight+.001,...
    pc       .*  pc_weight+0.001...
%     pc2       .*  pc2_weight+0.001,...
    ];
end

function C = get_kmeans_centroids(data)
% Make a k-means model and return the centroids
optimize = questdlg('Optimize Cluster Number?','Cluster Optimization','Elbow Optimized','Elbow w/ Min Clust Size','User Defined','Elbow Optimized');
C = [];
switch optimize
    case 'Elbow Optimized'
        opt_options = inputdlg({'Max Clusters','Replicates'},'Cluster Optimization',[1 50; 1 50],{'100','3'});
        if isempty(opt_options); return; end
        
        %Cap the max clusters to the number of samples.
        if size(data,1) < str2double(opt_options{1})
            opt_options{1} = num2str(size(data,1));
        end
        [~,C] = kmeans_opt(data, str2double(opt_options{1}), 0, str2double(opt_options{2}));
        
    case 'Elbow w/ Min Clust Size'
        opt_options = inputdlg({'Max Clusters','Replicates','Min Clust Size'},'Cluster Optimization',[1 50; 1 50; 1 50],{'100','3','1'});
        if isempty(opt_options); return; end
        
        %Cap the max clusters to the number of samples.
        if size(data,1) < str2double(opt_options{1})
            opt_options{1} = num2str(size(data,1));
        end
        [IDX,C] = kmeans_opt(data, str2double(opt_options{1}), 0, str2double(opt_options{2}));
        Celb = C;
        [GC,~] = groupcounts(IDX);
        numcl = length(GC);
        while min(GC) < str2double(opt_options{3})
            numcl = numcl - 1;
            [IDX,C] = kmeans(data,numcl,'Distance','sqeuclidean','Replicates',str2double(opt_options{2}));
            [GC,~] = groupcounts(IDX);
        end
        if numcl == 1
            warning('Unable to find more than one cluster >= the min cluster size. Proceeding with basic elbow-optimized method.')
            C = Celb;
        end
        
    case 'User Defined'
        opt_options = inputdlg({'# of Clusters','Replicates'},'Cluster with k-means',[1; 1],{'15','10'});
        %k = inputdlg({'Choose number of k-means:'},'Cluster with k-means',1,{'15'});
        if isempty(opt_options); return; end
        k = str2double(opt_options{1});
        nReps = str2double(opt_options{2});
        [~, C] = kmeans(data,k,'Distance','sqeuclidean','Replicates',nReps);
end
end