
%%
% 模型输出的ply文件所在文件夹
ply_dir = 'D:\oo\color\test_outputs\0901\grnet_test\149\';
% 场景编号
fname = '1006';
% pcread 读取ply文件
ptc_o = pcread([ply_dir fname '_d.ply']);       % dense
ptc_os = pcread([ply_dir fname '_s.ply']);      % sparse
ptc_i = pcread([ply_dir  fname '_input.ply']);  % input
figure("Name", "Input"); pcshow(ptc_i)
figure("Name", "Output Sparse"); pcshow(ptc_os)
figure("Name", "Output Dense"); pcshow(ptc_o)
%%
% num_sample = 1;
% idx = 16384*(num_sample-1)+1:16384*(num_sample);
% points_i = ptc_i.Location; points_i(:, 1) = rescale(points_i(:, 1), 0, 1); points_i(:, 2) = rescale(points_i(:, 2), 0, 1); points_i(:, 3) = rescale(points_i(:, 3), 0, 1);
% points_o = ptc_o.Location(idx,:);% points_o(:, 1) = rescale(points_o(:, 1), 0, 1); 
% points_o(:, 2) = rescale(points_o(:, 2), 0, 1); points_o(:, 3) = rescale(points_o(:, 3), 0, 1);
% figure('Name', 'Input'); pcshow(points_i);
% figure('Name', 'Output'); pcshow(points_o);
%%
% points_i = ptc_i.Location; %points_o = ptc_o.Location(1:16384,:);
% num_sample = 3;
% points_o = ptc_o.Location(16384*(num_sample-1)+1:16384*(num_sample),:);
% 
% % colmin = [0, 0, 0]; colmax = [1000, 1, 260];
% % points_i = rescale(points_i, 'InputMin', colmin, 'InputMax', colmax);
% % points_o = rescale(points_o, 'InputMin', colmin, 'InputMax', colmax);
% figure('Name', 'Input'); pcshow(points_i);
% figure('Name', 'Output'); pcshow(points_o);
% %%
% prefix = 'D:\oo\color\datas\old\mat\';
% idx = '80';
% load([prefix 'gt\' idx '.mat'], 'gt');
% load([prefix 'preds\' idx '.mat'], 'preds');
% 
% % gt1 = squeeze(gt(1,:,:));
% gt1 = gt;
% gt1(:,1) = rescale(gt1(:,1), 1, 1000);
% gt1(:,2) = rescale(gt1(:,2), 1, 346);
% gt1(:,3) = rescale(gt1(:,3), 1, 260);
% figure("Name", "gt"); pcshow(gt1)
% 
% % pred1 = squeeze(preds(1,:,:));
% pred1 = preds;
% pred1(:,1) = rescale(pred1(:,1), 1, 1000);
% pred1(:,2) = rescale(pred1(:,2), 1, 346);
% pred1(:,3) = rescale(pred1(:,3), 1, 260);
% figure("Name", "output"); pcshow(pred1)
% %%
% points_i = ptc_i.Location;
% points_i(:,1) = round(rescale(points_i(:,1), 1, 200));
% points_i(:,2) = round(rescale(points_i(:,2), 1, 222));
% points_i(:,3) = round(rescale(points_i(:,3), 1, 124));
% 
% ind = sub2ind([124,222], points_i(:,3), points_i(:,2));
% A = accumarray([points_i(:, 3) points_i(:, 2)], 1, [124, 222]);
% % A(A>10)=10;
% % figure; imagesc(flipud(A));
% gt_xy = points_i(:,2:end);
% %%
% points_o = ptc_o.Location;
% points_o(:,1) = round(rescale(points_o(:,1), 1, 200));
% points_o(:,2) = round(rescale(points_o(:,2), 1, 222));
% points_o(:,3) = round(rescale(points_o(:,3), 1, 124));
% pred_xy = points_o(:,2:end);
% %%
% [X, Y] = meshgrid(1:222, 1:124);
% sample_points = cat(1, reshape(X, 1, []), reshape(Y, 1, []));
% %%
% sample_points = repmat(reshape(sample_points, 1, 2,[]), 16384, 1, 1);
% %%
% input_p0 = sum(exp(-vecnorm(sample_points - gt_xy, 2, 2).^2 / 0.01), 1);
% output_p0 = sum(exp(-vecnorm(sample_points - pred_xy, 2, 2).^2 / 0.01), 1);
% %%
% sqrt(mean((input_p0 - output_p0).^2))
% %%
% B = reshape(input_p0, [124, 222]);
% figure;imagesc(flipud(B));
% C = reshape(output_p0, [124, 222]);
% figure;imagesc(flipud(C))