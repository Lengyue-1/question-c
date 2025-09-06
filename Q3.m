clear;close all;clc;
%这里定义参数，就不多说了
P_M1_0 = [20000; 0; 2000];
P_FY1_0 = [17800; 0; 1800];
v_M1_mag = 300;
g = 9.8;
R_T = 7; 
H_T = 10;
C_T = [0; 200; 0];
R_cloud = 10;
problem_constants.P_M1_0 = P_M1_0;
problem_constants.P_FY1_0 = P_FY1_0;
problem_constants.v_M1_mag = v_M1_mag;
problem_constants.g = g;
problem_constants.R_T = R_T;
problem_constants.H_T = H_T;
problem_constants.C_T = C_T;
problem_constants.R_cloud = R_cloud;
% 明确我们要优化的决策变量 (共8个)：
% x(1): theta_FY1  - 无人机飞行方向 (弧度)
% x(2): v_FY1      - 无人机飞行速度 (米/秒)
% x(3): t_fly_1    - 第一次投放前的飞行时间
% x(4): t_fuse_1   - 第一个烟幕弹的引信时间
% x(5): t_fly_2    - 第二次投放前的飞行时间
% x(6): t_fuse_2   - 第二个烟幕弹的引信时间
% x(7): t_fly_3    - 第三次投放前的飞行时间
% x(8): t_fuse_3   - 第三个烟幕弹的引信时间
nvars = 8; 
theta_FY1_min = 0;      theta_FY1_max = 0.2;  % 方向角范围
v_FY1_min = 70;         v_FY1_max = 140;% 速度范围
t_fly_min = 0;          t_fly_max = 10;% 飞行时间范围
t_fuse_min = 0;         t_fuse_max = 10;% 引信时间范围
lb = [theta_FY1_min, v_FY1_min, t_fly_min, t_fuse_min, t_fly_min, t_fuse_min, t_fly_min, t_fuse_min];
ub = [theta_FY1_max, v_FY1_max, t_fly_max, t_fuse_max, t_fly_max, t_fuse_max, t_fly_max, t_fuse_max];
% 确保烟幕弹的投放时间间隔足够长。（题目的死约束，相邻烟雾弹之间要隔1s投放）
A = [0 0  1  0 -1  0  0  0;  % 对应 t_fly_1 - t_fly_2 <= -1
     0 0  0  0  1  0 -1  0];  % 对应 t_fly_2 - t_fly_3 <= -1
b = [-1; -1];
fitfun = @(x) fitness_function_Q3(x, problem_constants);%%遗传算法参数与之前一样，我就不多赘述了
options = optimoptions('ga', ...
    'PopulationSize', 100, ... 
    'MaxGenerations', 50, ...
    'FunctionTolerance', 1e-4, ... 
    'ConstraintTolerance', 1e-4, ...
    'Display', 'iter', ... 
    'CrossoverFraction',0.9,...
    'MutationFcn', {@mutationuniform, 0.1}, ...
    'PlotFcn', {@gaplotbestf}, ...  
    'UseParallel', true);  
[x_optimal, fval] = ga(fitfun, nvars, A, b, [], [], lb, ub, [], options);
theta_FY1_opt = x_optimal(1);
v_FY1_opt = x_optimal(2);
t_fly_all_opt = [x_optimal(3), x_optimal(5), x_optimal(7)]; % 按第1,2,3枚的顺序
t_fuse_all_opt = [x_optimal(4), x_optimal(6), x_optimal(8)];
v_FY1_opt_vec = v_FY1_opt * [cos(theta_FY1_opt); sin(theta_FY1_opt); 0];
P_drops = zeros(3, 3); % 每列存储一个点的[X;Y;Z]坐标
P_dets = zeros(3, 3);
for k = 1:3
    t_fly = t_fly_all_opt(k);
    t_fuse = t_fuse_all_opt(k);
    P_drops(:, k) = P_FY1_0 + v_FY1_opt_vec * t_fly;
    P_dets(:, k) = P_drops(:, k) + v_FY1_opt_vec * t_fuse + [0; 0; -0.5 * g] * t_fuse^2;
end
% 遗传算法默认是找最小值，我们的目标是最大化遮蔽时间。
% 所以适应度函数返回的是负分。现在需要把结果变回正的。
max_shielding_time_total = -fval;
[~, individual_times_sorted, detonation_order] = analyze_shielding_time(x_optimal, problem_constants);
individual_times = zeros(1, 3);
original_order_map(detonation_order) = 1:3;
individual_times = individual_times_sorted(original_order_map);
fprintf('飞行方向: %.3f 度\n', rad2deg(theta_FY1_opt));
fprintf('飞行速度:     %.2f m/s\n', v_FY1_opt);
for k = 1:3
    fprintf('第 %d 枚烟幕弹\n', k);
    fprintf('投放前飞行时间:  %.3f s\n', k, t_fly_all_opt(k));
    fprintf('烟幕弹引信时间: %.3f s\n', k, t_fuse_all_opt(k));
    fprintf('投放点坐标 (X,Y,Z):      (%.2f, %.2f, %.2f) m\n', P_drops(1, k), P_drops(2, k), P_drops(3, k));
    fprintf('起爆点坐标 (X,Y,Z):      (%.2f, %.2f, %.2f) m\n', P_dets(1, k), P_dets(2, k), P_dets(3, k));
    fprintf('[独立有效时长贡献]:      %.3f s\n', individual_times(k));
end
fprintf('\n找到的最大有效遮蔽时长: %.3f 秒\n', max_shielding_time_total);

% 提取最优参数用于动画
theta_FY1_opt = x_optimal(1); v_FY1_opt = x_optimal(2);
t_fly_all_opt_unsorted = [x_optimal(3), x_optimal(5), x_optimal(7)];
t_fuse_all_opt_unsorted = [x_optimal(4), x_optimal(6), x_optimal(8)];
t_det_all_opt_unsorted = t_fly_all_opt_unsorted + t_fuse_all_opt_unsorted;
[t_det_all_opt_sorted, sort_idx] = sort(t_det_all_opt_unsorted);
t_fly_all_opt_sorted = t_fly_all_opt_unsorted(sort_idx);

% 准备动画时间轴
dt_anim = 0.05; % 动画步长，值越小越流畅，但会慢
T_end = max(t_det_all_opt_sorted) + 21; % 动画总时长，到最后一个烟幕失效
time_vec = 0:dt_anim:T_end;

% 预计算所有物体的完整轨迹
u_M1 = ([0;0;0] - P_M1_0) / norm([0;0;0] - P_M1_0); v_M1 = v_M1_mag * u_M1;
P_M1_traj = P_M1_0 + v_M1 * time_vec;
v_FY1_opt_vec = v_FY1_opt * [cos(theta_FY1_opt); sin(theta_FY1_opt); 0];
P_FY1_traj = P_FY1_0 + v_FY1_opt_vec * time_vec;

P_bomb_t_anim = @(t, t_fly) (P_FY1_0 + v_FY1_opt_vec * t) + [0;0;-0.5*g] .* (t - t_fly).^2 .* (t >= t_fly);
P_cloud_t_anim = @(t, t_fly, t_det) P_bomb_t_anim(min(t, t_det), t_fly) + [0;0;-3] .* (t - t_det) .* (t >= t_det);
P_cloud1_traj = P_cloud_t_anim(time_vec, t_fly_all_opt_sorted(1), t_det_all_opt_sorted(1));
P_cloud2_traj = P_cloud_t_anim(time_vec, t_fly_all_opt_sorted(2), t_det_all_opt_sorted(2));
P_cloud3_traj = P_cloud_t_anim(time_vec, t_fly_all_opt_sorted(3), t_det_all_opt_sorted(3));

% 创建3D场景
figure('Name', '最优策略动态仿真 (三烟幕带视锥)', 'NumberTitle', 'off', 'Position', [100 100 1400 900]);
ax = axes;
hold(ax, 'on'); grid(ax, 'on'); axis(ax, 'equal'); view(3);
xlabel('X (米)'); ylabel('Y (米)'); zlabel('Z (米)');
xlim([-1000 21000]); zlim([0 2500]);

% 绘制静态物体和轨迹线
plot3(ax, 0, 0, 0, 'kx', 'MarkerSize', 15, 'LineWidth', 3, 'DisplayName', '假目标');
[Xc, Yc, Zc] = cylinder(R_T, 20);
surf(ax, Xc*R_T+C_T(1), Yc*R_T+C_T(2), Zc*H_T+C_T(3), 'FaceColor', 'b', 'EdgeColor', 'none', 'FaceAlpha', 0.6, 'DisplayName', '真目标');
plot3(ax, P_M1_traj(1,:), P_M1_traj(2,:), P_M1_traj(3,:), 'r--', 'LineWidth', 1.5, 'DisplayName', 'M1 轨迹');
plot3(ax, P_FY1_traj(1, time_vec <= max(t_fly_all_opt_sorted)), ...
          P_FY1_traj(2, time_vec <= max(t_fly_all_opt_sorted)), ...
          P_FY1_traj(3, time_vec <= max(t_fly_all_opt_sorted)), 'c--', 'LineWidth', 1.5, 'DisplayName', 'FY1 轨迹 (最优)');

% 创建动态物体的图形句柄
h_M1 = plot3(ax, nan, nan, nan, 'r^', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
h_FY1 = plot3(ax, nan, nan, nan, 'co', 'MarkerSize', 8, 'MarkerFaceColor', 'c');
[Xs, Ys, Zs] = sphere(20);
h_cloud1 = surf(ax, nan(size(Xs)), 'FaceColor', '#77AC30', 'EdgeColor', 'none', 'FaceAlpha', 0.4); % 绿色
h_cloud2 = surf(ax, nan(size(Xs)), 'FaceColor', '#0072BD', 'EdgeColor', 'none', 'FaceAlpha', 0.4); % 蓝色
h_cloud3 = surf(ax, nan(size(Xs)), 'FaceColor', '#D95319', 'EdgeColor', 'none', 'FaceAlpha', 0.4); % 橙色
h_cone = patch(ax, 'XData', [], 'YData', [], 'ZData', [], 'FaceColor', 'g', 'EdgeColor', 'none', 'FaceAlpha', 0.2);

K_T = [ C_T(1), C_T(1), C_T(1)+R_T, C_T(1)-R_T;
        C_T(2)-R_T, C_T(2)+R_T, C_T(2), C_T(2);
        C_T(3), C_T(3), C_T(3), C_T(3); ];
K_T = [K_T, K_T + [0;0;H_T]]; num_key_points = size(K_T, 2);

legend(ax, 'Location', 'northeast'); is_shielded_log = false(1, length(time_vec));

% 动画主循环
for i = 1:length(time_vec)
    t_current = time_vec(i);
    set(h_M1, 'XData', P_M1_traj(1,i), 'YData', P_M1_traj(2,i), 'ZData', P_M1_traj(3,i));
    if t_current <= max(t_fly_all_opt_sorted), set(h_FY1, 'XData', P_FY1_traj(1,i), 'YData', P_FY1_traj(2,i), 'ZData', P_FY1_traj(3,i)); end
    
    P_M1_current = P_M1_traj(:, i); status_str = "未遮蔽"; cone_color = 'r';

    % 控制烟幕云显示
    is_cloud1_active = t_current >= t_det_all_opt_sorted(1) && t_current < t_det_all_opt_sorted(1) + 20;
    is_cloud2_active = t_current >= t_det_all_opt_sorted(2) && t_current < t_det_all_opt_sorted(2) + 20;
    is_cloud3_active = t_current >= t_det_all_opt_sorted(3) && t_current < t_det_all_opt_sorted(3) + 20;
    set(h_cloud1, 'Visible', is_cloud1_active);
    set(h_cloud2, 'Visible', is_cloud2_active);
    set(h_cloud3, 'Visible', is_cloud3_active);
    
    if is_cloud1_active, P_c1 = P_cloud1_traj(:,i); set(h_cloud1, 'XData', P_c1(1)+R_cloud*Xs, 'YData', P_c1(2)+R_cloud*Ys, 'ZData', P_c1(3)+R_cloud*Zs); end
    if is_cloud2_active, P_c2 = P_cloud2_traj(:,i); set(h_cloud2, 'XData', P_c2(1)+R_cloud*Xs, 'YData', P_c2(2)+R_cloud*Ys, 'ZData', P_c2(3)+R_cloud*Zs); end
    if is_cloud3_active, P_c3 = P_cloud3_traj(:,i); set(h_cloud3, 'XData', P_c3(1)+R_cloud*Xs, 'YData', P_c3(2)+R_cloud*Ys, 'ZData', P_c3(3)+R_cloud*Zs); end
    
    % 进行联合遮蔽判断
    idx1 = []; idx2 = []; idx3 = [];
    if is_cloud1_active, idx1 = get_occluded_indices(P_M1_current, P_c1, K_T, R_cloud); end
    if is_cloud2_active, idx2 = get_occluded_indices(P_M1_current, P_c2, K_T, R_cloud); end
    if is_cloud3_active, idx3 = get_occluded_indices(P_M1_current, P_c3, K_T, R_cloud); end
    if length(union(union(idx1, idx2), idx3)) == num_key_points
        is_shielded_log(i) = true; status_str = "有效遮蔽"; cone_color = 'g';
    end
    
    % 绘制遮蔽锥 (选择一个主导云来绘制，例如第一个激活的云)
    P_cloud_for_cone = [];
    if is_cloud1_active, P_cloud_for_cone = P_c1;
    elseif is_cloud2_active, P_cloud_for_cone = P_c2;
    elseif is_cloud3_active, P_cloud_for_cone = P_c3;
    end
    
    if ~isempty(P_cloud_for_cone)
        V_axis = P_cloud_for_cone - P_M1_current;
        dist_missile_cloud = norm(V_axis);
        
        if dist_missile_cloud <= R_cloud
            set(h_cone, 'Visible', 'off'); % 导弹在云里，不画锥
        else
            alpha = asin(R_cloud / dist_missile_cloud);
            V_axis_norm = V_axis / dist_missile_cloud;
            if abs(dot(V_axis_norm, [0;0;1])) > 0.99, arbitrary_vec = [0; 1; 0]; else, arbitrary_vec = [0; 0; 1]; end
            ortho1 = cross(V_axis_norm, arbitrary_vec); ortho1 = ortho1 / norm(ortho1);
            ortho2 = cross(V_axis_norm, ortho1);
            
            cone_base_dist = norm(C_T - P_M1_current) + 2*H_T; % 与代码1一致的锥底距离
            cone_base_radius = cone_base_dist * tan(alpha);
            P_cone_base_center = P_M1_current + cone_base_dist * V_axis_norm;
            theta_circle = linspace(0, 2*pi, 40);
            p_circle = P_cone_base_center + cone_base_radius * (ortho1 * cos(theta_circle) + ortho2 * sin(theta_circle));
            
            cone_vertices = [P_M1_current, p_circle];
            cone_faces = [ones(1, 39); 2:40; 3:40, 2]';
            set(h_cone, 'Vertices', cone_vertices', 'Faces', cone_faces, 'FaceColor', cone_color, 'Visible', 'on');
        end
    else
        set(h_cone, 'Visible', 'off');
    end

    % 更新标题信息
    shielded_time = sum(is_shielded_log(1:i)) * dt_anim;
    title(ax, sprintf('最优策略仿真 | 时间: %.2f 秒 | 状态: %s | 累计遮蔽时长: %.2f 秒', t_current, status_str, shielded_time));
    drawnow;
end

%% 
% 后面的函数部分保持不变

function neg_T_effective = fitness_function_Q3(x, constants)
    [T_total, ~, ~] = analyze_shielding_time(x, constants);
    neg_T_effective = -T_total;
end

function [T_total, T_individual_sorted, det_order] = analyze_shielding_time(x, constants)
    theta_FY1 = x(1); v_FY1 = x(2);
    t_fly_all = [x(3), x(5), x(7)];
    t_fuse_all = [x(4), x(6), x(8)];

    P_M1_0 = constants.P_M1_0; P_FY1_0 = constants.P_FY1_0; v_M1_mag = constants.v_M1_mag;
    g = constants.g; R_T = constants.R_T; H_T = constants.H_T; C_T = constants.C_T; R_cloud = constants.R_cloud;
    u_M1 = ([0;0;0] - P_M1_0) / norm([0;0;0] - P_M1_0);
    v_M1 = v_M1_mag * u_M1;
    P_M1_t = @(t) P_M1_0 + v_M1 * t;

    v_FY1_vec = v_FY1 * [cos(theta_FY1); sin(theta_FY1); 0];
    P_FY1_t = @(t) P_FY1_0 + v_FY1_vec * t;
    t_det_unsorted = t_fly_all + t_fuse_all;
    [t_det_sorted, det_order] = sort(t_det_unsorted);
    t_fly_sorted = t_fly_all(det_order);
    P_bomb_t = @(t, t_fly) P_FY1_t(t) + [0;0;-0.5*g] .* (t - t_fly).^2 .* (t >= t_fly);
    P_cloud_t = @(t, t_fly, t_det) P_bomb_t(min(t, t_det), t_fly) + [0;0;-3] .* (t - t_det) .* (t >= t_det);

    P_cloud1_t = @(t) P_cloud_t(t, t_fly_sorted(1), t_det_sorted(1));
    P_cloud2_t = @(t) P_cloud_t(t, t_fly_sorted(2), t_det_sorted(2));
    P_cloud3_t = @(t) P_cloud_t(t, t_fly_sorted(3), t_det_sorted(3));
    
    K_T = [ C_T(1), C_T(1), C_T(1)+R_T, C_T(1)-R_T;
            C_T(2)-R_T, C_T(2)+R_T, C_T(2), C_T(2);
            C_T(3), C_T(3), C_T(3), C_T(3); ];
    K_T = [K_T, K_T + [0;0;H_T]];
    num_key_points = size(K_T, 2);
    dt_sim = 0.01;%%步长
    t_sim_start = t_det_sorted(1);
    t_sim_end = min(t_det_sorted(3) + 20, t_det_sorted(1) + 25);

    if isempty(t_sim_start) || (t_sim_end <= t_sim_start)
        T_total = 0; T_individual_sorted = [0, 0, 0]; return; 
    end
    sim_time_vec = t_sim_start : dt_sim : t_sim_end;
    
    shielded_steps_total = 0;
    shielded_steps_individual = [0, 0, 0];
    
    for t = sim_time_vec
        P_M1_current = P_M1_t(t);
        idx1 = []; idx2 = []; idx3 = [];

        if t >= t_det_sorted(1) && t < t_det_sorted(1) + 20
            idx1 = get_occluded_indices(P_M1_current, P_cloud1_t(t), K_T, R_cloud);
            if length(unique(idx1)) == num_key_points
                shielded_steps_individual(1) = shielded_steps_individual(1) + 1;
            end
        end
        if t >= t_det_sorted(2) && t < t_det_sorted(2) + 20
            idx2 = get_occluded_indices(P_M1_current, P_cloud2_t(t), K_T, R_cloud);
            if length(unique(idx2)) == num_key_points
                shielded_steps_individual(2) = shielded_steps_individual(2) + 1;
            end
        end
        if t >= t_det_sorted(3) && t < t_det_sorted(3) + 20
            idx3 = get_occluded_indices(P_M1_current, P_cloud3_t(t), K_T, R_cloud);
            if length(unique(idx3)) == num_key_points
                shielded_steps_individual(3) = shielded_steps_individual(3) + 1;
            end
        end

        if length(union(union(idx1, idx2), idx3)) == num_key_points
            shielded_steps_total = shielded_steps_total + 1;
        end
    end
    
    T_total = shielded_steps_total * dt_sim;
    T_individual_sorted = shielded_steps_individual * dt_sim;
end

function occluded_indices = get_occluded_indices(P_M1, P_cloud, K_T, R_cloud)
    V_axis = P_cloud - P_M1;
    dist_missile_cloud = norm(V_axis);

    if dist_missile_cloud <= R_cloud
        occluded_indices = 1:size(K_T, 2); return;
    end
    
    alpha = asin(R_cloud / dist_missile_cloud);
    occluded_indices = [];
    
    for k = 1:size(K_T, 2)
        W = K_T(:, k) - P_M1;
        cos_beta_k = dot(V_axis, W) / (dist_missile_cloud * norm(W));
        beta_k = acos(max(min(cos_beta_k, 1), -1));
        
        if beta_k <= alpha
            occluded_indices = [occluded_indices, k];
        end
    end
end
