using Plots
gr()

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


# --------------------
# 3. シミュレーション設定と実行
# --------------------

# 時間設定
const t_0:: Float64 = 0.0
const t_f:: Float64 = 100.0
const del_t:: Float64 = 1e-2
const N_steps:: Int = round((t_f - t_0) / del_t)

# 粒子設定 (N_body)
const N_body:: Int = 3  # 粒子の数

# 質量 (np.ndarray)
masses = [100.0, 150.0, 100.0]

# 初期条件 (N_body, 4) の配列として定義: (x, y, vx, vy)
initial_conditions = zeros(N_body, 4)

# 粒子1 (x, y, vx, vy)
initial_conditions[1, :] .= [-10.0, 0.0, 0.5, 4.0]
# 粒子2 (x, y, vx, vy)
initial_conditions[2, :] .= [0.0, 0.0, 0.0, 0.0]
# 粒子3 (x, y, vx, vy)
initial_conditions[3, :] .= [10.0, 0.0, -0.5, -3.99]

# 結果格納用の配列 (N_steps, N_body, 4)
# (時間ステップ, 粒子番号, 状態変数(x, y, vx, vy))
history = zeros(N_steps, N_body, 4)
history[1, :, :] = initial_conditions

# 現在の状態変数 (N_body, 4)
current_state = copy(initial_conditions)

# シミュレーション実行
t_list = zeros(N_steps)
t_list[1] = t_0

for i in 1:(N_steps - 1)
    # RK4 ステップの実行
    global current_state = runge_kutta_step(del_t, masses, current_state)

    # 結果の格納
    history[i+1, :, :] = current_state
    t_list[i+1] = t_list[i] + del_t

    println("$(i)ステップが実行されました。")
end


# --------------------
# 4. アニメーションの作成
# --------------------

# プロットの初期設定
# x, y 軸の表示範囲を固定することで、粒子が画面外に出るのを防ぎ、見やすくします。
# 粒子の初期位置から大まかな範囲を設定。必要に応じて調整してください。
# 例: x_min = -15.0, x_max = 15.0, y_min = -15.0, y_max = 15.0
x_coords = history[:, :, 1] # 全ステップの全粒子のX座標
y_coords = history[:, :, 2] # 全ステップの全粒子のY座標

# プロット範囲を自動的に調整する代わりに、固定することもできます
plot_xlim = (minimum(x_coords) - 5, maximum(x_coords) + 5)
plot_ylim = (minimum(y_coords) - 5, maximum(y_coords) + 5)

const PLOT_SKIP = 100 # 100ステップごとに1回プロット (10,000/100 = 100フレームに削減)

# アニメーションオブジェクトの初期化
anim = @animate for i in 1:PLOT_SKIP:N_steps
    # 現在の時刻 t とステップ i のデータを取得
    current_time = t_list[i]
    current_positions = history[i, :, 1:2] # 現在ステップの全粒子の (x, y)

    # 粒子の位置 (x_i, y_i) を取得
    x = current_positions[:, 1]
    y = current_positions[:, 2]

    # プロット
    plot(x, y,
         seriestype = :scatter, # 散布図としてプロット
         markercolor = [:red, :blue, :green], # 粒子ごとに色を割り当てる (N_body の数に合わせる)
         markersize = masses ./ maximum(masses) .* 8 .+ 2, # 質量に応じてマーカーサイズを調整
         legend = false, # 凡例は表示しない
         xlims = plot_xlim, # x軸の範囲を固定
         ylims = plot_ylim, # y軸の範囲を固定
         aspect_ratio = :equal, # アスペクト比を1:1に固定して、歪みをなくす
         title = "N-Body Simulation (t = $(round(current_time, digits=2)))", # タイトルに現在の時刻を表示
         xlabel = "X Position",
         ylabel = "Y Position"
        )
    
    # オプション: 軌跡を表示する場合（全ての粒子）
    # for p_idx in 1:N_body
    #     # history[1:i, p_idx, 1] は、開始から 'i' ステップ目までの粒子のX座標
    #     plot!(history[1:i, p_idx, 1], history[1:i, p_idx, 2], 
    #           linecolor = [:red, :blue, :green][p_idx], 
    #           linewidth = 0.5, 
    #           alpha = 0.7,
    #           seriestype = :path, 
    #           legend = false)
    # end
end

# display(anim)

# アニメーションをGIFファイルとして保存
# fps (frames per second) はフレームレート
gif(anim, "nbody_simulation.gif", fps = 30)

println("アニメーション 'nbody_simulation.gif' が作成されました。")
