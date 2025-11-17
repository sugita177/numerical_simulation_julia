using Plots

# --------------------
# 1. ヘルパー関数
# --------------------

# eps softening parameter (重力計算の特異点を避けるための緩和パラメータ)
const G = 1.0
const EPS = 1e-1


# 粒子の加速度を計算する関数
function calculate_acceleration(masses, positions)
    """
    全ての粒子の相互作用による加速度を計算する。

    Args:
        masses : 各粒子の質量 (N_body,)
        positions : 各粒子の位置 (N_body, 2) (x, y)

    Returns:
       array : 各粒子の加速度 (N_body, 2) (ax, ay)
    """
    N_body = length(masses)
    # 加速度を格納する配列を初期化 (ax, ay)
    accelerations = zeros(N_body, 2)

    # 各粒子 i について、他の全ての粒子 j からの重力を合計
    for i = 1:N_body, j = 1:N_body
        if i != j
            # 粒子 i から j への相対位置ベクトル (rx, ry)
            # r_vector = pos_j - pos_i
            r_vector = positions[j, :] - positions[i, :]

            # 距離の二乗: r^2 = (xj - xi)^2 + (yj - yi)^2
            r_squared = sum(r_vector.^2)

            # 距離: r
            r = sqrt(r_squared + EPS^2)  # ソフトニング込みの距離

            # 粒子 i にかかる、粒子 j からの力 F_ij = G * mi * mj / r^2 * (r_vector / r)
            # 粒子 i にかかる加速度 a_i = F_ij / m_i = G * m_j / r^3 * r_vector
            
            # スカラー加速度の絶対値 |a_i| = G * m_j / (r_squared + EPS**2)
            accel_abs = G * masses[j] / (r^3)

            # 加速度ベクトル (ax, ay) = accel_abs * r_vector
            accel_vector = accel_abs * r_vector

            # 総力（総加速度）に加算
            accelerations[i, :] .+= accel_vector
        end
    end
    return accelerations
end


# --------------------
# 2. ルンゲ・クッタ (RK4) のステップ関数
# --------------------

# 状態ベクトルの導関数 f(y) = (v, a) の計算
function derivative(pos, vel, masses)
    # pos: (N_body, 2), vel: (N_body, 2)
    accelerations = calculate_acceleration(masses, pos)  # (N_body, 2)
    # 導関数 (dy/dt) は (d_pos/dt, d_vel/dt) = (vel, accel)
    # np.hstack で (N_body, 4) に結合
    return hcat(vel, accelerations)
end

function runge_kutta_step(del_t, masses, current_state)
    """
    ルンゲ・クッタ法（4次）で1ステップ時間の積分を行う。

    Args:
        del_t (float): タイムステップ Delta t
        masses (array): 粒子の質量 (N_body,)
        current_state (array): 現在の状態変数 (N_body, 4) (x, y, vx, vy)

    Returns:
        array: 次の時間の状態変数 (N_body, 4)
    """
    # 状態変数の分解
    positions = current_state[:, 1:2]  # (x, y)
    velocities = current_state[:, 3:4]  # (vx, vy)

    # K1
    deriv1 = derivative(positions, velocities, masses)
    k1 = del_t * deriv1

    # K2
    state2 = current_state .+ k1 / 2.0
    deriv2 = derivative(state2[:, 1:2], state2[:, 3:4], masses)
    k2 = del_t * deriv2

    # K3
    state3 = current_state .+ k2 / 2.0
    deriv3 = derivative(state3[:, 1:2], state3[:, 3:4], masses)
    k3 = del_t * deriv3

    # K4
    state4 = current_state .+ k3
    deriv4 = derivative(state4[:, 1:2], state4[:, 3:4], masses)
    k4 = del_t * deriv4

    # 次の状態を計算
    next_state = current_state .+ (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return next_state
end