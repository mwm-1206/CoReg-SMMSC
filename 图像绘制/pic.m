%% ================= 批量绘图配置区 =================
clear; clc; close all;

% 1. 【在这里手动填写数据集名称】
% 格式：{'名字1', '名字2', ...}
dataset_names = {'mfeat','UCI_3view','bbcsport_2view','NGs','WebKB'}; 

% 2. Excel 文件后缀
% 假设你的文件是 "数据集名_GridSearch_Results.xlsx"
file_suffix = '_GridSearch_Results.xlsx'; 
sheetName = 'Sheet1'; 

% 3. 需要遍历的 p 值
p_list = [50, 100, 200, 500];

% 4. 图片保存文件夹
output_folder = 'Batch_Figures2';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% ================= 开始批量处理 =================

for d_idx = 1:length(dataset_names)
    current_dataset = dataset_names{d_idx};
    
    % 拼接读取的文件名
    full_filename = [current_dataset, file_suffix];
    
    fprintf('------------------------------------------------\n');
    fprintf('正在处理: %s (文件: %s)\n', current_dataset, full_filename);
    
    % 读取 Excel
    try
        opts = detectImportOptions(full_filename);
        opts.VariableNamingRule = 'preserve'; 
        T = readtable(full_filename, opts, 'Sheet', sheetName);
    catch ME
        fprintf(2, '【跳过】无法读取 %s。错误: %s\n', full_filename, ME.message);
        continue; 
    end
    
    % 遍历 p 值
    for p_idx = 1:length(p_list)
        target_p = p_list(p_idx);
        
        if ~ismember(target_p, T.p)
            fprintf('  -> 警告: 没找到 p=%d，跳过。\n', target_p);
            continue;
        end
        
        fprintf('  -> 正在绘制 p=%d ...\n', target_p);
        
        % 调用绘图函数
        try
            plot_single_graph(T, target_p, current_dataset, output_folder);
        catch ME
            fprintf(2, '  -> 绘图失败 p=%d。错误: %s\n', target_p, ME.message);
        end
    end
end

fprintf('------------------------------------------------\n');
fprintf('全部完成！请去 "%s" 文件夹查看图片。\n', output_folder);


%% ================= 子函数：绘图逻辑 =================
function plot_single_graph(full_table, p_val, dataset_name, save_dir)
    % 1. 数据筛选
    subT = full_table(full_table.p == p_val, :);
    
    % 2. 准备矩阵
    betas = sort(unique(subT.Beta));
    gammas = sort(unique(subT.Gamma));
    num_beta = length(betas);
    num_gamma = length(gammas);
    
    Z_data = zeros(num_gamma, num_beta);
    
    for i = 1:height(subT)
        r = find(gammas == subT.Gamma(i));
        c = find(betas == subT.Beta(i));
        if ~isempty(r) && ~isempty(c)
            Z_data(r, c) = subT.ACC(i);
        end
    end
    
    % 3. Z轴截断逻辑
    min_acc = min(Z_data(:));
    z_base = min_acc - 0.2;
    Z_plot = Z_data - z_base;
    Z_plot(Z_plot < 0) = 0; 
    
    % 4. 创建图形 (不可见模式)
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [0, 0, 1000, 800]);
    
    h = bar3(Z_plot, 0.6);
    
    % 颜色插值
    for k = 1:length(h)
        h(k).CData = h(k).ZData;
        h(k).FaceColor = 'interp';
        h(k).EdgeColor = [0.1 0.1 0.1]; 
        h(k).LineWidth = 0.5;
    end
    colormap(jet);
    
    % 5. 坐标轴
    ax = gca;
    
    set(ax, 'XTick', 1:num_beta);
    set(ax, 'XTickLabel', string(betas));
    xlabel('Beta (\beta)', 'FontSize', 12, 'FontWeight', 'bold');
    
    set(ax, 'YTick', 1:num_gamma);
    set(ax, 'YTickLabel', string(gammas));
    ylabel('Gamma (\gamma)', 'FontSize', 12, 'FontWeight', 'bold');
    
    % Z轴标签还原
    current_zticks = get(ax, 'ZTick');
    new_zlabels = current_zticks + z_base;
    set(ax, 'ZTickLabel', arrayfun(@(x) sprintf('%.2f', x), new_zlabels, 'UniformOutput', false));
    
    z_max_plot = 1.0 - z_base;
    zlim([0, z_max_plot]);
    zlabel('ACC', 'FontSize', 12, 'FontWeight', 'bold');
    
    % 6. 标题
    % title_str = sprintf('%s (p=%d)', dataset_name, p_val);
    % title(title_str, 'FontSize', 16, 'Interpreter', 'none'); 
    
    view(-50, 30);
    grid on; box on;
    
    % ================================================
    % 【修复核心】绘制并还原 Colorbar 的真实数值
    % ================================================
    cb = colorbar;
    cb.Label.String = 'ACC';
    
    % 获取当前 Colorbar 上的默认刻度 (0 到 0.2...)
    current_cbticks = cb.Ticks; 
    % 将其加上 z_base 还原为真实的 ACC 刻度 (0.77 到 0.97...)
    new_cblabels = current_cbticks + z_base; 
    % 重新赋值给 Colorbar 的标签
    cb.TickLabels = arrayfun(@(x) sprintf('%.2f', x), new_cblabels, 'UniformOutput', false);

    
    % 7. 正确生成路径
    filename_pure = sprintf('%s_p%d.png', dataset_name, p_val);
    save_path = fullfile(save_dir, filename_pure);
    
    saveas(fig, save_path);
    
    close(fig);
end