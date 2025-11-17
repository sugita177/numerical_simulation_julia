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