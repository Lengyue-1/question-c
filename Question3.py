import numpy as np
from deap import base, creator, tools, algorithms
import warnings

warnings.filterwarnings("ignore")


# -------------------------- 1. 定义常量参数（与原代码一致） --------------------------
# 初始位置（m）：M1导弹、FY1无人机
P_M1_0 = np.array([20000, 0, 2000])
P_FY1_0 = np.array([17800, 0, 1800])
# 物理参数：M1速度（m/s）、重力加速度（m/s²）
v_M1_mag = 300
g = 9.8
# 真目标参数：半径（m）、高度（m）、中心坐标（m）
R_T = 7
H_T = 10
C_T = np.array([0, 200, 0])
# 烟幕云参数：半径（m）、有效时长（s）
R_cloud = 10
cloud_life = 20

# 打包常量（便于函数传递）
problem_constants = {
    "P_M1_0": P_M1_0,
    "P_FY1_0": P_FY1_0,
    "v_M1_mag": v_M1_mag,
    "g": g,
    "R_T": R_T,
    "H_T": H_T,
    "C_T": C_T,
    "R_cloud": R_cloud,
    "cloud_life": cloud_life,
}

# 决策变量配置（8个变量：theta_FY1, v_FY1, t_fly1, t_fuse1, t_fly2, t_fuse2, t_fly3, t_fuse3）
nvars = 8
lb = np.array([0, 70, 0, 0, 0, 0, 0, 0])  # 变量下界
ub = np.array([0.2, 140, 10, 10, 10, 10, 10, 10])  # 变量上界


# 约束条件：相邻烟幕投放间隔≥1s（t_fly2 ≥ t_fly1+1，t_fly3 ≥ t_fly2+1）
def check_constraint(individual):
    t_fly1, t_fly2, t_fly3 = individual[2], individual[4], individual[6]
    return (t_fly2 >= t_fly1 + 1) and (t_fly3 >= t_fly2 + 1)


# -------------------------- 2. 核心功能函数（与原代码一致） --------------------------
def get_occluded_indices(P_M1, P_cloud, K_T, R_cloud):
    """计算真目标关键点中被烟幕云遮挡的索引"""
    V_axis = P_cloud - P_M1
    dist_missile_cloud = np.linalg.norm(V_axis)

    # 导弹在烟幕云内：所有关键点均被遮挡
    if dist_missile_cloud <= R_cloud:
        return list(range(K_T.shape[1]))

    # 计算遮挡角度阈值
    alpha = np.arcsin(R_cloud / dist_missile_cloud)
    occluded_indices = []

    # 逐个判断关键点是否在遮挡锥内
    for k in range(K_T.shape[1]):
        W = K_T[:, k] - P_M1
        dist_M1_key = np.linalg.norm(W)
        if dist_M1_key < 1e-6:  # 避免除以0
            occluded_indices.append(k)
            continue

        # 计算导弹-关键点与导弹-烟幕云的夹角
        cos_beta_k = np.dot(V_axis, W) / (dist_missile_cloud * dist_M1_key)
        cos_beta_k = np.clip(cos_beta_k, -1, 1)  # 防止数值溢出
        beta_k = np.arccos(cos_beta_k)

        # 角度≤阈值则遮挡
        if beta_k <= alpha:
            occluded_indices.append(k)

    return occluded_indices


def analyze_shielding_time(x, constants):
    """分析总遮蔽时间与单个烟幕遮蔽时间"""
    # 解析决策变量
    theta_FY1, v_FY1 = x[0], x[1]
    t_fly_all = np.array([x[2], x[4], x[6]])
    t_fuse_all = np.array([x[3], x[5], x[7]])

    # 解析常量
    P_M1_0 = constants["P_M1_0"]
    P_FY1_0 = constants["P_FY1_0"]
    v_M1_mag = constants["v_M1_mag"]
    g = constants["g"]
    C_T = constants["C_T"]
    R_T = constants["R_T"]
    R_cloud = constants["R_cloud"]
    cloud_life = constants["cloud_life"]

    # 计算导弹（M1）和无人机（FY1）速度向量
    u_M1 = (np.array([0, 0, 0]) - P_M1_0) / np.linalg.norm(np.array([0, 0, 0]) - P_M1_0)
    v_M1 = v_M1_mag * u_M1
    v_FY1_vec = v_FY1 * np.array([np.cos(theta_FY1), np.sin(theta_FY1), 0])

    # 烟幕起爆时间排序（按起爆先后）
    t_det_unsorted = t_fly_all + t_fuse_all
    det_order = np.argsort(t_det_unsorted)
    t_det_sorted = t_det_unsorted[det_order]
    t_fly_sorted = t_fly_all[det_order]

    # 位置计算函数
    def P_M1_t(t):
        return P_M1_0 + v_M1 * t

    def P_FY1_t(t):
        return P_FY1_0 + v_FY1_vec * t

    def P_bomb_t(t, t_fly):
        if t < t_fly:
            return P_FY1_t(t)
        else:
            delta_t = t - t_fly
            return (
                P_FY1_t(t_fly)
                + v_FY1_vec * delta_t
                + np.array([0, 0, -0.5 * g * delta_t**2])
            )

    def P_cloud_t(t, t_fly, t_det):
        p_bomb = P_bomb_t(min(t, t_det), t_fly)
        if t >= t_det:
            return p_bomb + np.array([0, 0, -3 * (t - t_det)])
        return p_bomb

    # 真目标关键点（底面4点+顶面4点）
    K_T_bottom = np.array(
        [
            [C_T[0], C_T[0], C_T[0] + R_T, C_T[0] - R_T],
            [C_T[1] - R_T, C_T[1] + R_T, C_T[1], C_T[1]],
            [C_T[2], C_T[2], C_T[2], C_T[2]],
        ]
    )
    K_T_top = K_T_bottom + np.array([[0, 0, 0, 0], [0, 0, 0, 0], [H_T, H_T, H_T, H_T]])
    K_T = np.hstack([K_T_bottom, K_T_top])
    num_key_points = K_T.shape[1]

    # 仿真时间配置
    dt_sim = 0.01
    t_sim_start = t_det_sorted[0]
    t_sim_end = min(t_det_sorted[-1] + cloud_life, t_det_sorted[0] + 25)
    if t_sim_end <= t_sim_start:
        return 0.0, np.array([0.0, 0.0, 0.0]), det_order
    sim_time_vec = np.arange(t_sim_start, t_sim_end + dt_sim, dt_sim)

    # 统计遮蔽时间
    shielded_steps_total = 0
    shielded_steps_individual = np.zeros(3)

    for t in sim_time_vec:
        P_M1_current = P_M1_t(t)
        idx1, idx2, idx3 = [], [], []

        # 第1个烟幕（排序后）
        if t >= t_det_sorted[0] and t < t_det_sorted[0] + cloud_life:
            P_c1 = P_cloud_t(t, t_fly_sorted[0], t_det_sorted[0])
            idx1 = get_occluded_indices(P_M1_current, P_c1, K_T, R_cloud)
            if len(set(idx1)) == num_key_points:
                shielded_steps_individual[0] += 1

        # 第2个烟幕（排序后）
        if t >= t_det_sorted[1] and t < t_det_sorted[1] + cloud_life:
            P_c2 = P_cloud_t(t, t_fly_sorted[1], t_det_sorted[1])
            idx2 = get_occluded_indices(P_M1_current, P_c2, K_T, R_cloud)
            if len(set(idx2)) == num_key_points:
                shielded_steps_individual[1] += 1

        # 第3个烟幕（排序后）
        if t >= t_det_sorted[2] and t < t_det_sorted[2] + cloud_life:
            P_c3 = P_cloud_t(t, t_fly_sorted[2], t_det_sorted[2])
            idx3 = get_occluded_indices(P_M1_current, P_c3, K_T, R_cloud)
            if len(set(idx3)) == num_key_points:
                shielded_steps_individual[2] += 1

        # 联合遮蔽判断
        all_occluded = list(set(idx1 + idx2 + idx3))
        if len(all_occluded) == num_key_points:
            shielded_steps_total += 1

    # 步数转时间
    T_total = shielded_steps_total * dt_sim
    T_individual_sorted = shielded_steps_individual * dt_sim

    return T_total, T_individual_sorted, det_order


def fitness_function_Q3(x, constants):
    """适应度函数：返回负遮蔽时间（用于遗传算法最小化优化）"""
    T_total, _, _ = analyze_shielding_time(x, constants)
    return -T_total


# -------------------------- 3. 遗传算法配置与运行 --------------------------
def init_ga():
    """初始化DEAP遗传算法框架"""
    # 删除已存在的适应度/个体类（避免重复定义报错）
    if "FitnessMin" in dir(creator):
        del creator.FitnessMin
    if "Individual" in dir(creator):
        del creator.Individual

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 最小化目标
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # 生成个体：每个变量在[lb[i], ub[i]]内均匀采样
    for i in range(nvars):
        toolbox.register(f"attr_var{i}", np.random.uniform, lb[i], ub[i])
    # 组合变量为个体
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        [getattr(toolbox, f"attr_var{i}") for i in range(nvars)],
        n=1,
    )
    # 生成种群
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 遗传算法操作（匹配原MATLAB参数）
    def evaluate(individual):
        # 不满足约束的个体：适应度设为极大值（淘汰）
        if not check_constraint(individual):
            return (1e18,)
        # 满足约束：计算适应度
        return (fitness_function_Q3(individual, problem_constants),)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 交叉操作
    toolbox.register(
        "mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1
    )  # 变异操作
    toolbox.register("select", tools.selTournament, tournsize=3)  # 选择操作

    return toolbox


def run_ga(toolbox, pop_size=100, n_gen=50, cx_pb=0.9, mut_pb=0.1):
    """运行遗传算法，返回最优个体与最大遮蔽时间"""
    # 初始化种群
    pop = toolbox.population(n=pop_size)
    # 计算初始种群适应度
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # 进化过程（打印迭代信息）
    print("=" * 60)
    print("遗传算法迭代过程")
    print("=" * 60)
    print(f"{'Gen':<4}{'Best Fitness':<15}{'Best Shielding Time (s)':<20}")
    print("-" * 60)

    for gen in range(n_gen):
        # 选择子代
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # 交叉操作
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < cx_pb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 变异操作（并裁剪到变量边界）
        for mutant in offspring:
            if np.random.random() < mut_pb:
                toolbox.mutate(mutant)
                for i in range(nvars):
                    mutant[i] = np.clip(mutant[i], lb[i], ub[i])
                del mutant.fitness.values

        # 重新计算无效个体的适应度
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 更新种群
        pop[:] = offspring

        # 打印当前代最优结果
        best_ind = tools.selBest(pop, 1)[0]
        best_fitness = best_ind.fitness.values[0]
        best_shield_time = -best_fitness  # 还原为正的遮蔽时间
        print(f"{gen+1:<4}{best_fitness:<15.4f}{best_shield_time:<20.3f}")

    # 返回最优个体与最大遮蔽时间
    best_ind = tools.selBest(pop, 1)[0]
    return np.array(best_ind), -best_ind.fitness.values[0]


# -------------------------- 4. 主逻辑（运行优化+输出结果） --------------------------
if __name__ == "__main__":
    # 步骤1：初始化并运行遗传算法
    toolbox = init_ga()
    x_optimal, max_shielding_time_total = run_ga(
        toolbox,
        pop_size=100,  # 种群大小（匹配MATLAB）
        n_gen=50,  # 迭代次数（匹配MATLAB）
        cx_pb=0.9,  # 交叉概率（匹配MATLAB）
        mut_pb=0.1,  # 变异概率（匹配MATLAB）
    )

    # 步骤2：解析最优参数
    theta_FY1_opt = x_optimal[0]  # 无人机方向角（弧度）
    v_FY1_opt = x_optimal[1]  # 无人机速度（m/s）
    t_fly_all_opt = x_optimal[[2, 4, 6]]  # 3次投放前飞行时间
    t_fuse_all_opt = x_optimal[[3, 5, 7]]  # 3次引信时间
    v_FY1_opt_vec = v_FY1_opt * np.array(
        [np.cos(theta_FY1_opt), np.sin(theta_FY1_opt), 0]
    )

    # 计算投放点与起爆点坐标
    P_drops = np.zeros((3, 3))  # 投放点（3个点×3个坐标）
    P_dets = np.zeros((3, 3))  # 起爆点（3个点×3个坐标）
    for k in range(3):
        t_fly = t_fly_all_opt[k]
        t_fuse = t_fuse_all_opt[k]
        P_drops[:, k] = P_FY1_0 + v_FY1_opt_vec * t_fly  # 投放点
        P_dets[:, k] = (
            P_drops[:, k]
            + v_FY1_opt_vec * t_fuse
            + np.array([0, 0, -0.5 * g * t_fuse**2])
        )  # 起爆点

    # 计算单个烟幕的独立遮蔽时间
    _, T_individual_sorted, det_order = analyze_shielding_time(
        x_optimal, problem_constants
    )
    original_order_map = np.zeros(3, dtype=int)
    original_order_map[det_order] = np.arange(3)
    individual_times = T_individual_sorted[original_order_map]

    # 步骤3：打印最优结果
    print("\n" + "=" * 60)
    print("最优参数与遮蔽时间结果")
    print("=" * 60)
    print(f"无人机飞行方向: {np.rad2deg(theta_FY1_opt):.3f} 度")
    print(f"无人机飞行速度:     {v_FY1_opt:.2f} m/s")
    print("\n" + "-" * 40)
    for k in range(3):
        print(f"\n第 {k+1} 枚烟幕弹")
        print(f"  投放前飞行时间:  {t_fly_all_opt[k]:.3f} s")
        print(f"  引信时间:        {t_fuse_all_opt[k]:.3f} s")
        print(
            f"  投放点坐标 (X,Y,Z):  ({P_drops[0,k]:.2f}, {P_drops[1,k]:.2f}, {P_drops[2,k]:.2f}) m"
        )
        print(
            f"  起爆点坐标 (X,Y,Z):  ({P_dets[0,k]:.2f}, {P_dets[1,k]:.2f}, {P_dets[2,k]:.2f}) m"
        )
        print(f"  独立有效遮蔽时长:    {individual_times[k]:.3f} s")
    print("\n" + "-" * 40)
    print(f"最终最大有效遮蔽总时长: {max_shielding_time_total:.3f} 秒")
