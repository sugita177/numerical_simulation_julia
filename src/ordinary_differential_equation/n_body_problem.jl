using PythonPlot
using PythonCall
using Printf
using Statistics

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
# 3.5. 全運動範囲の計算
# --------------------

# 全x座標の最小値と最大値
x_min_all = minimum(history[:, :, 1])
x_max_all = maximum(history[:, :, 1])

# 全y座標の最小値と最大値
y_min_all = minimum(history[:, :, 2])
y_max_all = maximum(history[:, :, 2])

# xとyの最大変動幅
range_x = x_max_all - x_min_all
range_y = y_max_all - y_min_all

# アスペクト比1:1を維持し、かつ全体が収まるように調整
max_range = max(range_x, range_y)

# センタリング
center_x = (x_min_all + x_max_all) / 2
center_y = (y_min_all + y_max_all) / 2

# 新しい固定表示範囲 (余白として 10% を追加)
padding = max_range * 0.1 / 2.0 
fixed_lim = (max_range / 2.0) + padding 

const X_LIM = (center_x - fixed_lim, center_x + fixed_lim)
const Y_LIM = (center_y - fixed_lim, center_y + fixed_lim)

# --------------------
# 4. アニメーション出力
# -------------------- 

# PythonPlotのpyplotとanimationモジュールを取得
pygui(true) # GUIバックエンドを有効に
const plt = pyimport("matplotlib.pyplot")
const animation = pyimport("matplotlib.animation")

# --- アニメーション設定 ---
const INTERVAL = 10 
const SKIP_FRAMES = 10 # 描画フレーム数 = N_steps / 1000
const TRAIL_LENGTH = N_steps # 軌跡として表示する過去のステップ数
const N_frames = floor(Int, N_steps / SKIP_FRAMES)
const COLORS = ["b", "r", "g"]

# --- プロットの初期化 ---
fig, ax = plt.subplots(figsize=(8, 8)) 
ax.set_title("N-Body Simulation (RK4)")
ax.set_xlabel("x")
ax.set_ylabel("y")
# 軸の範囲を固定値に設定
ax.set_xlim(X_LIM[1], X_LIM[2]) 
ax.set_ylim(Y_LIM[1], Y_LIM[2]) 
ax.set_aspect("equal") 
ax.grid(true)

# 粒子本体 (Marker) のプロットオブジェクトを初期化
plot_objects = []
# 軌跡 (Trail) のプロットオブジェクトを初期化
trail_objects = []

for i in 1:N_body
    marker_size = 1 + sqrt(masses[i])
    current_color = COLORS[i]
    
    # 粒子本体 (点)
    line, = ax.plot(
        [], [], "o", 
        color=current_color,
        ms=marker_size, 
        zorder=5, 
        label="Mass $(i): $(masses[i])"
    )    
    push!(plot_objects, line)

    # 軌跡 (線)
    trail, = ax.plot(
        [], [], "-", 
        color=current_color,
        alpha=0.5, 
        lw=1.5, 
        zorder=0
    )
    push!(trail_objects, trail)
end

ax.legend(loc="upper right")

# 現在の時間を表示するためのテキストオブジェクト
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

# すべての描画オブジェクトを一つの配列にまとめる
all_artists = vcat(plot_objects, trail_objects, [time_text])


# --- 初期化関数 (init_func) ---
function init_func()
    for line in plot_objects
        line.set_data([], [])
    end
    for trail in trail_objects
        trail.set_data([], [])
    end
    time_text.set_text("")
    return (all_artists...,)
end


# --- フレーム描画関数 (update_func) ---
function update_func(frame_index)
    current_step = frame_index * SKIP_FRAMES + 1
    start_step = max(1, current_step - TRAIL_LENGTH) 

    # 各粒子について位置を更新
    for i in 1:N_body
        # 1. 粒子本体の位置更新
        x_pos = history[current_step, i, 1]
        y_pos = history[current_step, i, 2]
        plot_objects[i].set_data([x_pos], [y_pos])
        
        # 2. 軌跡データの取得と更新
        x_trail = history[start_step:current_step, i, 1]
        y_trail = history[start_step:current_step, i, 2]
        trail_objects[i].set_data(x_trail, y_trail)
    end
    
    # 時間表示を更新
    current_time = t_list[current_step]
    time_text.set_text(@sprintf("Time = %.2f", current_time))

    # 更新されたオブジェクトをすべて返す
    return (all_artists...,)
end


# --- アニメーションの生成と表示 ---

# FuncAnimationオブジェクトを作成
anim = animation.FuncAnimation(
    fig,
    update_func,
    init_func=init_func,
    frames=N_frames,
    interval=INTERVAL,
    blit=true, # 軸が固定のため、blit=trueで高速化
    repeat=false
)

plt.show()