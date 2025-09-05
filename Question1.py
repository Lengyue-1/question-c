# Physics-based simulation scaffold for the smoke-screen problem (A题)
# - Defines missiles, UAVs, smoke grenades, and coverage check
# - Runs Problem 1 scenario by default (FY1: v=120 m/s toward fake_target, release at 1.5 s, explode 3.6 s later)
#
# You can reuse/extend this for Problems 2–5 by adding more UAVs/grenades and tweaking parameters.

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ----------------------
# Constants & Parameters
# ----------------------
g = 9.8  # m/s^2
missile_speed = 300.0  # m/s (constant)
uav_speed_min, uav_speed_max = 70.0, 140.0  # m/s
cloud_sink_speed = 3.0  # m/s (downward, along -z)
cloud_effective_radius = 10.0  # m (within 10 m of center gives effective遮蔽)
cloud_effective_duration = 20.0  # s (post-explosion)


# ----------------------
# Initial positions
# ----------------------
M1_start = np.array([20000.0, 0.0, 2000.0])
M2_start = np.array([19000.0, 600.0, 2100.0])
M3_start = np.array([18000.0, -600.0, 1900.0])

target_true = np.array([0.0, 200.0, 0.0])
target_fake = np.array([0.0, 0.0, 0.0])

FY1_start = np.array([17800.0, 0.0, 1800.0])
FY2_start = np.array([12000.0, 1400.0, 1400.0])
FY3_start = np.array([6000.0, -3000.0, 700.0])
FY4_start = np.array([11000.0, 2000.0, 1800.0])
FY5_start = np.array([13000.0, -2000.0, 1300.0])


# ----------------------
# Utility geometry funcs
# ----------------------
def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v.copy()
    return v / n


def line_point_distance_segment(p0: np.ndarray, p1: np.ndarray, c: np.ndarray) -> float:
    """
    Distance from point c to line segment p0->p1 in R^3.
    """

    v = p1 - p0
    w = c - p0
    vv = np.dot(v, v)
    if vv == 0:
        return np.linalg.norm(c - p0)
    t = np.clip(np.dot(w, v) / vv, 0.0, 1.0)
    proj = p0 + t * v
    return float(np.linalg.norm(c - proj))


# ----------------------
# Entities
# ----------------------
@dataclass
class Missile:
    name: str
    start: np.ndarray
    target_direction: np.ndarray  # unit vector pointing to fake target
    speed: float = missile_speed

    def position(self, t: float) -> np.ndarray:
        return self.start + self.target_direction * (self.speed * t)

    def time_to_reach_point(self, point: np.ndarray) -> float:
        # only correct if moving straight line exactly toward 'point'
        dist = np.linalg.norm(point - self.start)
        return dist / self.speed


@dataclass
class UAV:
    name: str
    start: np.ndarray
    heading_unit: np.ndarray  # fixed once assigned
    speed: float  # in [70, 140]

    def position(self, t: float) -> np.ndarray:
        return self.start + self.heading_unit * (self.speed * t)


@dataclass
class SmokeGrenade:
    uav: UAV
    release_time: float  # t_{i,j}
    fuse_delay: float  # T_{i,j}
    # cached values (computed after init when needed)
    _release_pos: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _release_vel: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _explode_time: Optional[float] = field(default=None, init=False, repr=False)
    _explode_pos: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def release_pos(self) -> np.ndarray:
        if self._release_pos is None:
            self._release_pos = self.uav.position(self.release_time)
        return self._release_pos

    def release_vel(self) -> np.ndarray:
        if self._release_vel is None:
            # initial velocity equals UAV velocity at release
            self._release_vel = self.uav.heading_unit * self.uav.speed
        return self._release_vel

    def explode_time(self) -> float:
        if self._explode_time is None:
            self._explode_time = self.release_time + self.fuse_delay
        return self._explode_time

    def explode_pos(self) -> np.ndarray:
        if self._explode_pos is None:
            # ballistic motion from release to explode
            t = self.fuse_delay
            p0 = self.release_pos()
            v0 = self.release_vel()
            # gravity affects z only
            p = np.array(
                [
                    p0[0] + v0[0] * t,
                    p0[1] + v0[1] * t,
                    p0[2] + v0[2] * t - 0.5 * g * t * t,
                ]
            )
            self._explode_pos = p
        return self._explode_pos

    def cloud_center(self, t: float) -> Optional[np.ndarray]:
        """
        Center of the smoke cloud at time t (if active).
        Active in [explode_time, explode_time + cloud_effective_duration].
        After explosion, cloud sinks downward with 3 m/s (negative z direction).
        """

        te = self.explode_time()
        if t < te or t > te + cloud_effective_duration:
            return None
        dt = t - te
        p = self.explode_pos().copy()
        p[2] -= cloud_sink_speed * dt
        return p


# ----------------------
# Build default scenario (Problem 1)
# ----------------------
# Missile M1 toward fake target
M1_dir = unit(target_fake - M1_start)
missile_M1 = Missile("M1", M1_start, M1_dir)


# FY1 flies toward fake target at 120 m/s (fixed heading)
FY1_heading = unit(target_fake + (0, 0, 1800) - FY1_start)
FY1 = UAV("FY1", FY1_start, FY1_heading, speed=120.0)

# Smoke grenade from FY1: release at 1.5 s, explode 3.6 s after
grenade_1 = SmokeGrenade(FY1, release_time=1.5, fuse_delay=3.6)


# ----------------------
# Simulation helpers
# ----------------------
def compute_coverage_time_for_missile(
    missile: Missile,
    grenades: List[SmokeGrenade],
    target_true: np.ndarray,
    t0: float,
    t1: float,
    dt: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Returns (total_covered_time, time_grid, covered_mask)
    covered_mask[t_idx] = True if any cloud intersects LoS from missile to true target.
    """

    times = np.arange(t0, t1 + 1e-9, dt)
    covered = np.zeros_like(times, dtype=bool)

    for k, t in enumerate(times):

        mpos = missile.position(t)
        # line of sight endpoints
        p0 = mpos
        p1 = target_true

        # check all active clouds
        for g in grenades:
            c = g.cloud_center(t)
            if c is None:
                continue
            dir = unit((c[1] - p1[1], p1[0] - c[0], 0))
            print(dir)
            c_1 = p1 + 7 * dir
            c_2 = p1 - 7 * dir
            c_3 = c_1 + (0, 0, 10)
            c_4 = c_2 + (0, 0, 10)
            print(c_1, c_2, c_3, c_4)

            d1 = line_point_distance_segment(p0, c_1, c)
            d2 = line_point_distance_segment(p0, c_2, c)
            d3 = line_point_distance_segment(p0, c_3, c)
            d4 = line_point_distance_segment(p0, c_4, c)

            if (
                d1 <= cloud_effective_radius
                and d2 <= cloud_effective_radius
                and d3 <= cloud_effective_radius
                and d4 <= cloud_effective_radius
            ):
                covered[k] = True
                break

    total_time = covered.sum() * dt

    return total_time, times, covered


# ----------------------
# Run Problem 1 simulation
# ----------------------
# Simulate until M1 reaches fake target (straight line)
t_end = np.linalg.norm(target_fake - M1_start) / missile_speed
dt = 0.01  # 100 Hz temporal resolution
total_cov, time_grid, covered_mask = compute_coverage_time_for_missile(
    missile_M1, [grenade_1], target_true, t0=0.0, t1=t_end, dt=dt
)

print(f"[Problem 1] Estimated effective遮蔽时长 for M1: {total_cov:.2f} s")

# ----------------------
# Plot (single 3D figure, no explicit colors per instructions)
# ----------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Missile trajectory (to t_end)
T = np.linspace(0, t_end, 200)
M_traj = np.array([missile_M1.position(t) for t in T])
ax.plot(M_traj[:, 0], M_traj[:, 1], M_traj[:, 2], label=missile_M1.name)

# UAV FY1 trajectory (enough time for visualization)
T_uav = np.linspace(0, min(t_end, 60.0), 200)
FY1_traj = np.array([FY1.position(t) for t in T_uav])
ax.plot(FY1_traj[:, 0], FY1_traj[:, 1], FY1_traj[:, 2], label=FY1.name)

# Plot cloud centers at several snapshots during effectiveness
snapshots = np.linspace(
    grenade_1.explode_time(), grenade_1.explode_time() + cloud_effective_duration, 6
)
for ts in snapshots:
    c = grenade_1.cloud_center(ts)
    if c is None:
        continue
    # draw sphere surface (coarse for speed)
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 12)
    xs = cloud_effective_radius * np.outer(np.cos(u), np.sin(v)) + c[0]
    ys = cloud_effective_radius * np.outer(np.sin(u), np.sin(v)) + c[1]
    zs = cloud_effective_radius * np.outer(np.ones_like(u), np.cos(v)) + c[2]
    ax.plot_surface(xs, ys, zs, alpha=0.15, linewidth=0)

# Mark true and fake targets
ax.scatter(
    target_true[0],
    target_true[1],
    target_true[2],
    marker="o",
    s=50,
    label="True Target",
)
ax.scatter(
    target_fake[0],
    target_fake[1],
    target_fake[2],
    marker="^",
    s=50,
    label="Fake Target",
)

# Also mark explosion point
exp_pos = grenade_1.explode_pos()
ax.scatter(exp_pos[0], exp_pos[1], exp_pos[2], marker="x", s=60, label="Explosion")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Problem 1: M1, FY1, and Smoke Cloud (snapshots)")
ax.legend()

plt.show()

# ----------------------
# Also show a 2D coverage timeline for quick sanity check (separate figure)
# ----------------------
fig2 = plt.figure()
plt.plot(time_grid, covered_mask.astype(int))
plt.xlabel("Time (s)")
plt.ylabel("Coverage (1=True, 0=False)")
plt.title("Line-of-sight Coverage vs Time (M1 → True Target)")
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.show()
