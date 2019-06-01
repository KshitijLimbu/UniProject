function PlotAllImages(Y, titlename, dimensions)
% plot all images in dataset
% image giben in Y
% title in 'titlename'
% maximim plot dimesions is given by dimensions


% get dimensions
[N D]=size(Y);

% limit number of plots to dimensions * dimensions
if(N > dimensions ^ 2) 
    N = dimensions ^ 2;
end

% display N data sets
figure
hold on
subplot(1,1,1);
%bigTitle(titlename);

colormap gray; % want grey scaledim
imageSide = sqrt(D); % for each image
% for each image
for n=1:N
  subplot(dimensions,dimensions,n);   % do a sub-plot per image 
  % reshape image data so that nice imageSideximageSide image appears
  imagesc(reshape(Y(n,:)',imageSide,imageSide)); 
  axis off;
end
hold off

