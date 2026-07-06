%% 增强版 PatchMatch 纹理合成 - 修复质量 + 速度优化
%% 改进:
%%   1. RGB三通道联合匹配（单NNF + 三通道代价计算）
%%   2. 投票重建（patch voting）代替逐像素复制
%%   3. 双向传播（正向+反向交替扫描）
%%   4. regionfill 初始化 + PatchMatch 精修
%%   5. 向量化代价计算（去掉双层嵌套循环）
%%   6. 指数收缩随机搜索（预计算偏移表）
%%   7. 只遍历 hole 像素，跳过全图扫描

params = struct();
params.brightness_threshold = 75;
params.line_color = 'black';
params.patch_size = 11;
params.num_iterations = 8;
params.search_radius = 20;
params.edge_threshold_low = 0.01;
params.edge_threshold_high = 0.08;

fill_lines_patchmatch('2.jpg', '2_fixed_v21.jpg', params);

function fill_lines_patchmatch(img_path, output_path, params)
    if nargin < 3, params = struct(); end
    if ~isfield(params, 'line_color'), params.line_color = 'black'; end
    if ~isfield(params, 'brightness_threshold')
        if strcmp(params.line_color, 'black')
            params.brightness_threshold = 80;
        else
            params.brightness_threshold = 200;
        end
    end
    fn_defaults = {'edge_threshold_low',0.02; 'edge_threshold_high',0.1; ...
        'min_edge_length',30; 'max_line_width',15; 'line_length_threshold',3; ...
        'patch_size',9; 'num_iterations',5; 'search_radius',50};
    for i = 1:size(fn_defaults,1)
        if ~isfield(params, fn_defaults{i,1})
            params.(fn_defaults{i,1}) = fn_defaults{i,2};
        end
    end
    img = imread(img_path);
    if size(img,3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
        img = repmat(img, [1 1 3]);
    end
    mask = detect_lines_by_edges(img_gray, params);
    img_filled = patchmatch_inpaint_v2(img, mask, params);
    display_results_lines(img, img_gray, mask, img_filled, params);
    if nargin >= 2 && ~isempty(output_path)
        imwrite(img_filled, output_path);
        fprintf('Result saved to: %s\n', output_path);
    end
end

function mask = detect_lines_by_edges(img_gray, params)
    if nargin < 2, params = struct(); end
    mask = false(size(img_gray));
    fn_def = {'brightness_threshold',100; 'edge_threshold_low',0.02; ...
        'edge_threshold_high',0.1; 'min_edge_length',30; ...
        'max_line_width',15; 'line_color','black'};
    for i = 1:size(fn_def,1)
        if ~isfield(params, fn_def{i,1})
            params.(fn_def{i,1}) = fn_def{i,2};
        end
    end
    [h,w] = size(img_gray);
    edges = edge(img_gray, 'canny', [params.edge_threshold_low params.edge_threshold_high]);
    se = strel('line', 3, 0);
    edges_d = imdilate(edges, se);
    se = strel('line', 3, 90);
    edges_d = imdilate(edges_d, se);
    edges_d = imerode(edges_d, se);
    theta = -90:0.5:89.5;
    [H,T,R] = hough(edges_d, 'Theta', theta);
    peaks = houghpeaks(H, 200, 'Threshold', ceil(0.15*max(H(:))), 'NHoodSize', [15 15]);
    lines = houghlines(edges_d, T, R, peaks, ...
        'MinLength', params.min_edge_length, 'FillGap', 20);
    h_idx = []; v_idx = [];
    for k = 1:length(lines)
        a = abs(lines(k).theta);
        if a < 10 || abs(a-180) < 10, h_idx(end+1)=k;
        elseif abs(a-90) < 15, v_idx(end+1)=k; end
    end
    fprintf('Hough: %d lines\n', length(lines));
    if strcmp(params.line_color, 'black')
        bright_mask = img_gray < params.brightness_threshold;
    else
        bright_mask = img_gray > params.brightness_threshold;
    end
    %% Combine brightness + edge proximity
    mask = bright_mask & imdilate(edges, strel("disk",2));
    %% Multi-angle closing to connect broken stripes
    mask = imclose(mask, strel("line", 18, 0));
    mask = imclose(mask, strel("line", 18, 90));
    mask = imclose(mask, strel("line", 18, 45));
    mask = imclose(mask, strel("line", 18, 135));
    %% Add Hough line regions (expands mask, does not restrict)
    if ~isempty(lines)
        hl = false(h,w);
        for k = 1:length(lines)
            p1 = lines(k).point1; p2 = lines(k).point2;
            for t = 0:0.5:1
                cx = round(p1(1)+t*(p2(1)-p1(1)));
                cy = round(p1(2)+t*(p2(2)-p1(2)));
                y0 = max(1,cy-3); y1 = min(h,cy+3);
                x0 = max(1,cx-3); x1 = min(w,cx+3);
                hl(y0:y1,x0:x1) = true;
            end
        end
        mask = mask | (hl & bright_mask);
    end
    %% Cleanup
    mask = imfill(mask, "holes");
    mask = bwareaopen(mask, 30);
    mask = imdilate(mask, strel("disk",2));
    mask = imclose(mask, strel("disk",1));
end

function img_filled = patchmatch_inpaint_v2(img, mask, params)
    %% Criminisi exemplar-based inpainting
    %% Fills from boundary inward, copying entire known patches
    
    if nargin < 3, params = struct(); end
    if ~isfield(params,'patch_size'), params.patch_size=9; end
    if ~isfield(params,'search_radius'), params.search_radius=50; end
    
    [h, w, c] = size(img);
    ps = params.patch_size;
    hp = floor(ps / 2);
    sr = params.search_radius;
    
    fprintf('Criminisi: patch=%d, sr=%d, img=%dx%d\n', ps, sr, h, w);
    drawnow('update');
    
    img_w = double(img);
    initial_mask = mask;  %% save original hole mask
    mask_w = mask;  %% true = hole
    conf = double(~mask_w);  %% confidence: 1 for known, 0 for hole
    
    total_hole = sum(mask_w(:));
    remaining = total_hole;
    last_pct = 0;
    iter = 0;
    

    t_start = tic;
    
    while remaining > 0 && iter < 5000
        iter = iter + 1;
        
        %% 1. Find fill front (hole boundary)
        front = imdilate(mask_w, strel('disk', 1)) & mask_w;
        front = front & ~imerode(mask_w, strel('disk', 1));
        [fy, fx] = find(front);
        
        if isempty(fy), break; end
        
        %% Only consider boundary pixels whose full patch fits in the image
        valid = fy >= hp+1 & fy <= h-hp & fx >= hp+1 & fx <= w-hp;
        fy = fy(valid); fx = fx(valid);
        if isempty(fy), break; end
        
        %% 2. Compute priorities: prefer vertical (top/bottom) filling
        %% by adding a directional bias based on boundary normal
        n_f = length(fy);
        P = zeros(n_f, 1);
        C_vals = zeros(n_f, 1);
        D = bwdist(~mask_w);
        [dnx, dny] = gradient(D);
        
        for i = 1:n_f
            y = fy(i); x = fx(i);
            patch_mask = mask_w(y-hp:y+hp, x-hp:x+hp);
            n_known = sum(~patch_mask(:));
            n_total = numel(patch_mask);
            C = n_known / n_total;
            C_vals(i) = C;
            if C == 0, P(i) = 0; continue; end
            
            %% Directional bias: prefer vertical boundaries (up/down)
            nx = dnx(y, x); ny = dny(y, x);
            nmag = sqrt(nx^2 + ny^2);
            if nmag > 0, nx = nx/nmag; ny = ny/nmag; end
            vert_bias = abs(ny);  %% 1 = purely vertical, 0 = purely horizontal
            
            %% Bias strength: top/bottom get ~3x priority over left/right
            P(i) = C * (1 + 2.0 * vert_bias) + 0.001;
        end
        
        %% 3. Select best boundary pixel
        [~, best_i] = max(P);
        by = fy(best_i); bx = fx(best_i);
        
        %% 4. Find best matching source patch (fully known)
        ty1 = by-hp; ty2 = by+hp;
        tx1 = bx-hp; tx2 = bx+hp;
        target_patch = img_w(ty1:ty2, tx1:tx2, :);
        target_known = ~mask_w(ty1:ty2, tx1:tx2);
        
        s_yr = max(1+hp, by-sr); s_yr_end = min(h-hp, by+sr);
        s_xr = max(1+hp, bx-sr); s_xr_end = min(w-hp, bx+sr);
        
        n_search = (s_yr_end-s_yr+1) * (s_xr_end-s_xr+1);
        max_search = 5000;
        
        if n_search > max_search
            r_idx = randperm(n_search, max_search);
            [sy_grid, sx_grid] = meshgrid(s_yr:s_yr_end, s_xr:s_xr_end);
            sy_list = sy_grid(r_idx); sx_list = sx_grid(r_idx);
        else
            [sy_grid, sx_grid] = meshgrid(s_yr:s_yr_end, s_xr:s_xr_end);
            sy_list = sy_grid(:); sx_list = sx_grid(:);
        end
        
        best_cost = inf;
        best_sy = 0; best_sx = 0;
        
        for ci = 1:length(sy_list)
            sy = sy_list(ci); sx = sx_list(ci);
            if mask_w(sy, sx), continue; end
            
            %% Source patch must be fully known
            if any(mask_w(sy-hp:sy+hp, sx-hp:sx+hp), 'all')
                continue;
            end
            
            src_patch = img_w(sy-hp:sy+hp, sx-hp:sx+hp, :);
            
            diff = target_patch - src_patch;
            d2 = sum(diff.^2, 3);
            n_known = sum(target_known(:));
            if n_known < 3, continue; end
            
            cost = sum(d2(target_known)) / n_known;
            if cost < best_cost
                best_cost = cost; best_sy = sy; best_sx = sx;
            end
        end
        
        %% 5. Copy source patch to target
        if best_sy > 0
            src_patch = img_w(best_sy-hp:best_sy+hp, best_sx-hp:best_sx+hp, :);
            for ci = 1:c
                tmp = src_patch(:, :, ci);
                ch = img_w(ty1:ty2, tx1:tx2, ci);
                filled = mask_w(ty1:ty2, tx1:tx2);
                ch(filled) = tmp(filled);
                img_w(ty1:ty2, tx1:tx2, ci) = ch;
            end
            
            %% Update confidence (use actual C value, not priority)
            C_star = C_vals(best_i);
            for pi = ty1:ty2
                for pj = tx1:tx2
                    if mask_w(pi, pj)
                        conf(pi, pj) = C_star;
                    end
                end
            end
            
            %% Update mask
            mask_w(ty1:ty2, tx1:tx2) = false;
            remaining = sum(mask_w(:));

        else
            mask_w(by, bx) = false;
            conf(by, bx) = 0;
            remaining = sum(mask_w(:));
        end
        
        %% Progress
        pct = (total_hole - remaining) / total_hole * 100;
        if pct - last_pct >= 2 || remaining == 0
            elapsed = toc(t_start);
            eta = elapsed / max(pct, 1) * (100 - pct);
            fprintf('  Criminisi: %.0f%%%% done (%d iters, %d px left, %.1fs, ETA %.1fs)\n', ...
                pct, iter, remaining, elapsed, eta);
            drawnow('update');
            last_pct = pct;
        end
    end
    
    img_filled = uint8(max(0, min(255, img_w)));
    %% Final feather: smooth boundary of entire filled region to remove seams
    filled_region = initial_mask & ~mask_w;  %% pixels that were hole, now filled
    feather = imdilate(filled_region, strel("disk", 1)) & filled_region;
    for ci = 1:c
        ch = img_w(:, :, ci);
        ch_smooth = imgaussfilt(ch, 2.0);
        ch(feather) = ch_smooth(feather);
        img_w(:, :, ci) = ch;
    end
    
    fprintf('  Criminisi done: %d iterations, %.1fs\n', iter, toc(t_start));
end
function display_results_lines(img, img_gray, mask, img_filled, params)
    figure('Position', [0 0 1600 900]);
    subplot(2,3,1); imshow(img); title('Original Image');
    edges = edge(img_gray, 'canny', [0.02 0.1]);
    subplot(2,3,2); imshow(edges); title('Edge Detection');
    subplot(2,3,3); imshow(mask); title('Detected Lines');
    subplot(2,3,4); imshow(img_filled); title('PatchMatch v2 Filled');
    [h,w] = size(img_gray);
    cy = round(h/2); cx = round(w/2); cs = 100;
    crop_y = max(1,cy-cs):min(h,cy+cs);
    crop_x = max(1,cx-cs):min(w,cx+cs);
    comp = [img(crop_y,crop_x,:) img_filled(crop_y,crop_x,:)];
    subplot(2,3,5); imshow(comp); title('Comparison');
    diff_img = abs(double(img)-double(img_filled));
    diff_img = uint8(diff_img / max(diff_img(:)) * 255);
    subplot(2,3,6); imshow(diff_img); title('Difference Map');
    fprintf('line color: %s\n', params.line_color);
    fprintf('patch: %dx%d\n', params.patch_size, params.patch_size);
    fprintf('iters: %d\n', params.num_iterations);
end
