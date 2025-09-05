import math
import random


# -------------------------- 1. 辅助函数：物理模型与遮蔽判定 --------------------------
def line_sphere_intersection(A, B, C, r):
    """判断线段AB是否与球体（中心C，半径r）相交"""
    AB = (B[0] - A[0], B[1] - A[1], B[2] - A[2])
    AC = (C[0] - A[0], C[1] - A[1], C[2] - A[2])
    a = sum(x**2 for x in AB)
    if a < 1e-8:  # A、B重合，判断是否在球内
        return sum(x**2 for x in AC) <= r**2 + 1e-8
    b = 2 * sum(AB[i] * AC[i] for i in range(3))
    c = sum(x**2 for x in AC) - r**2
    discriminant = b**2 - 4 * a * c
    if discriminant < -1e-8:
        return False
    discriminant = max(discriminant, 0)
    sqrt_d = math.sqrt(discriminant)
    s1 = (-b - sqrt_d) / (2 * a)
    s2 = (-b + sqrt_d) / (2 * a)
    return (s1 >= -1e-8 and s1 <= 1 + 1e-8) or (s2 >= -1e-8 and s2 <= 1 + 1e-8)

'''
def is_shielded(time, theta, v_f, t_release, Δt_delay):
    """判断time时刻是否实现有效遮蔽"""
    # 1. 计算导弹M1位置
    sqrt101 = math.sqrt(101)
    M1 = (20000 - (3000 / sqrt101) * time, 0.0, 2000 - (300 / sqrt101) * time)

    # 2. 计算烟幕云团中心
    t_det = t_release + Δt_delay
    P_d = (
        17800 + v_f * t_det * math.cos(theta),
        v_f * t_det * math.sin(theta),
        1800 - 4.9 * Δt_delay**2,  # g=9.8，0.5*g=4.9
    )
    P_c = (P_d[0], P_d[1], P_d[2] - 3 * (time - t_det))  # 3m/s下沉

    # 3. 真目标采样点（覆盖关键位置）
    target_points = [
        (0, 200, 0),
        (0, 200, 10),  # 上下底面圆心
        # (7, 200, 0),
        (0, 207, 0),
        # (-7, 200, 0),
        (0, 193, 0),  # 下底面圆周
        # (7, 200, 10),
        (0, 207, 10),
        # (-7, 200, 10),
        (0, 193, 10),  # 上底面圆周
        # (7, 200, 5),
        # (0, 207, 5),
        # (-7, 200, 5),
        # (0, 193, 5),  # 侧面中点
    ]

    # 4. 判定是否有线段相交
    for Tp in target_points:
        if line_sphere_intersection(M1, Tp, P_c, 10):
            return True
    return False
'''

def is_shielded(time, theta, v_f, t_release, Δt_delay):
    """判断time时刻是否实现有效遮蔽（基于真目标周围四个关键点位）"""
    # 1. 计算导弹M1位置
    sqrt101 = math.sqrt(101)
    M1 = (20000 - (3000 / sqrt101) * time, 0.0, 2000 - (300 / sqrt101) * time)

    # 2. 计算烟幕云团中心
    t_det = t_release + Δt_delay
    P_d = (
        17800 + v_f * t_det * math.cos(theta),
        v_f * t_det * math.sin(theta),
        1800 - 4.9 * Δt_delay**2,  # g=9.8，0.5*g=4.9
    )
    P_c = (P_d[0], P_d[1], P_d[2] - 3 * (time - t_det))  # 3m/s下沉

    # 3. 真目标位置及周围四个关键点位（源自Q1_ai_modify的判定逻辑）
    target_true = (0.0, 200.0, 0.0)  # 真目标坐标

    # 计算垂直于视线方向的单位向量（xy平面内）
    # 视线方向向量（导弹到真目标）
    los_dir_x = target_true[0] - M1[0]
    los_dir_y = target_true[1] - M1[1]
    # 垂直方向向量（xy平面内旋转90度）
    perp_dir_x = -los_dir_y
    perp_dir_y = los_dir_x
    # 单位化垂直向量
    perp_len = math.hypot(perp_dir_x, perp_dir_y)
    if perp_len < 1e-8:
        dir_x, dir_y = 0.0, 1.0  # 避免除以零，使用默认方向
    else:
        dir_x = perp_dir_x / perp_len
        dir_y = perp_dir_y / perp_len

    # 生成四个关键点位（围绕真目标）
    d = 7.0  # 水平距离
    h = 10.0  # 垂直高度
    target_points = [
        (
            target_true[0] + d * dir_x,
            target_true[1] + d * dir_y,
            target_true[2],
        ),  # 右侧点
        (
            target_true[0] - d * dir_x,
            target_true[1] - d * dir_y,
            target_true[2],
        ),  # 左侧点
        (
            target_true[0] + d * dir_x,
            target_true[1] + d * dir_y,
            target_true[2] + h,
        ),  # 右上点
        (
            target_true[0] - d * dir_x,
            target_true[1] - d * dir_y,
            target_true[2] + h,
        ),  # 左上点
    ]

    # 4. 判定是否有线段相交（任一关键点被遮蔽即判定为有效遮蔽）
    for Tp in target_points:
        if line_sphere_intersection(M1, Tp, P_c, 10):
            return True
    return False


def calc_fitness(individual):
    """计算个体适应度（有效遮蔽时长）"""
    theta, v_f, t_release, Δt_delay = individual
    t_m_arrive = 67  # 导弹到达假目标时间（≈20099.75/300）
    tau_step = 0.01  # 数值积分步长

    # 约束检查：不满足则适应度为0
    if not (0 <= theta < 2 * math.pi):
        return 0.0
    if not (70 <= v_f <= 140):
        return 0.0
    if not (0 <= t_release <= t_m_arrive):
        return 0.0
    if not (0.1 <= Δt_delay <= 18.84):  # 避免烟幕弹/云团落地
        return 0.0
    t_det = t_release + Δt_delay
    if t_det > t_m_arrive:
        return 0.0

    # 积分区间：[t_det, min(t_det+20, t_m_arrive)]
    tau_start = t_det
    tau_end = min(t_det + 20, t_m_arrive)
    if tau_start >= tau_end:
        return 0.0

    # 数值积分计算遮蔽时长
    total_time = 0.0
    tau = tau_start
    while tau <= tau_end:
        if is_shielded(tau, theta, v_f, t_release, Δt_delay):
            total_time += tau_step
        tau += tau_step
    # 处理最后一步不足步长的情况
    if tau - tau_step < tau_end < tau:
        remaining = tau_end - (tau - tau_step)
        if is_shielded(tau_end, theta, v_f, t_release, Δt_delay):
            total_time += remaining
    return total_time


# -------------------------- 2. 遗传算法核心操作 --------------------------
def init_population(pop_size=100):
    """初始化种群"""
    population = []
    for _ in range(pop_size):
        theta = random.uniform(0, 2 * math.pi)
        v_f = random.uniform(70, 140)
        t_release = random.uniform(0, 67)
        Δt_delay = random.uniform(0.1, 18.84)
        population.append([theta, v_f, t_release, Δt_delay])
    return population


def selection(population, fitnesses, tour_size=3):
    """锦标赛选择"""
    new_pop = []
    pop_size = len(population)
    for _ in range(pop_size):
        candidates = random.sample(range(pop_size), tour_size)
        best_idx = max(candidates, key=lambda i: fitnesses[i])
        new_pop.append(population[best_idx].copy())
    return new_pop


def crossover(population, pc=0.8):
    """算术交叉"""
    pop_size = len(population)
    offspring = []
    for i in range(0, pop_size, 2):
        p1 = population[i]
        p2 = population[i + 1] if i + 1 < pop_size else p1.copy()
        if random.random() < pc:
            α = random.uniform(0, 1)
            c1 = [α * p1[j] + (1 - α) * p2[j] for j in range(4)]
            c2 = [(1 - α) * p1[j] + α * p2[j] for j in range(4)]
            offspring.extend([c1, c2])
        else:
            offspring.extend([p1.copy(), p2.copy()])
    return offspring[:pop_size]  # 保证种群规模一致


def mutate(population, pm=0.1):
    """变异操作"""
    for ind in population:
        # 变异方向角theta（±0.2π）
        if random.random() < pm:
            ind[0] += (random.random() - 0.5) * 0.4 * math.pi
            ind[0] = ind[0] % (2 * math.pi)
        # 变异速度v_f（±10m/s）
        if random.random() < pm:
            ind[1] += (random.random() - 0.5) * 20
            ind[1] = max(70, min(140, ind[1]))
        # 变异投放时刻t_release（±5s）
        if random.random() < pm:
            ind[2] += (random.random() - 0.5) * 10
            ind[2] = max(0, min(67, ind[2]))
        # 变异起爆延迟Δt_delay（±1s）
        if random.random() < pm:
            ind[3] += (random.random() - 0.5) * 2
            ind[3] = max(0.1, min(18.84, ind[3]))
    return population


def elitism(pop, fit, new_pop, new_fit):
    """精英保留：用当前最优替换新种群最差"""
    best_idx = max(range(len(fit)), key=lambda i: fit[i])
    best_ind = pop[best_idx].copy()
    best_fit = fit[best_idx]
    worst_idx = min(range(len(new_fit)), key=lambda i: new_fit[i])
    if best_fit > new_fit[worst_idx]:
        new_pop[worst_idx] = best_ind
        new_fit[worst_idx] = best_fit
    return new_pop, new_fit


# -------------------------- 3. 遗传算法主函数 --------------------------
def genetic_algorithm(pop_size=100, generations=200):
    """遗传算法主流程"""
    # 初始化
    random.seed(42)  # 固定种子保证可重复
    pop = init_population(pop_size)
    fit = [calc_fitness(ind) for ind in pop]
    best_fit = max(fit)
    best_ind = pop[fit.index(best_fit)].copy()

    # 迭代优化
    for gen in range(generations):
        # 选择→交叉→变异
        selected = selection(pop, fit)
        crossed = crossover(selected)
        mutated = mutate(crossed)
        # 计算新种群适应度
        new_fit = [calc_fitness(ind) for ind in mutated]
        # 精英保留
        new_pop, new_fit = elitism(pop, fit, mutated, new_fit)
        # 更新种群与最优解
        pop, fit = new_pop, new_fit
        current_best = max(fit)
        if current_best > best_fit:
            best_fit = current_best
            best_ind = pop[fit.index(current_best)].copy()
        # 每10代输出进度
        if (gen + 1) % 10 == 0:
            print(f"第{gen+1:3d}代 | 最优遮蔽时长：{best_fit:.2f}s")

    # 输出最终结果
    theta, v_f, t_release, Δt_delay = best_ind
    t_det = t_release + Δt_delay
    P_release = (
        17800 + v_f * t_release * math.cos(theta),
        v_f * t_release * math.sin(theta),
        1800.0,
    )
    P_det = (
        17800 + v_f * t_det * math.cos(theta),
        v_f * t_det * math.sin(theta),
        1800 - 4.9 * Δt_delay**2,
    )

    print("\n==================== 最优投放策略 ====================")
    print(f"1. 无人机飞行方向：{math.degrees(theta):.2f}°（与x轴正方向夹角）")
    print(f"2. 无人机飞行速度：{v_f:.2f} m/s")
    print(f"3. 烟幕弹投放时刻：{t_release:.2f} s")
    print(f"4. 烟幕弹起爆延迟：{Δt_delay:.2f} s")
    print(
        f"5. 投放点坐标：({P_release[0]:.2f}, {P_release[1]:.2f}, {P_release[2]:.2f}) m"
    )
    print(f"6. 起爆点坐标：({P_det[0]:.2f}, {P_det[1]:.2f}, {P_det[2]:.2f}) m")
    print(f"7. 有效遮蔽时长：{best_fit:.2f} s")
    print(f"8. 烟幕有效时间窗：[{t_det:.2f}, {min(t_det+20, 67):.2f}] s")
    return best_ind, best_fit


# -------------------------- 4. 运行算法 --------------------------
if __name__ == "__main__":
    best_strategy, max_shield_time = genetic_algorithm(
        pop_size=100, generations=200  # 种群规模  # 迭代次数
    )
