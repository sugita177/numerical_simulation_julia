using PythonPlot
using PythonCall
using Printf
using Statistics


# 初期値を返す
function get_intial_state_sin(x_data::AbstractVector{<:Real}, U::Real; wave_width_ratio::Real=0.2)
    """
    中央部にサインカーブを持つ初期状態を返す
    """
    Nx = length(x_data)
    u0 = zeros(Nx)
    x_center = (x_data[1] + x_data[end]) / 2.0
    L = x_data[end] - x_data[1]

    # 波を配置する範囲 (例: 全長の 20%)
    wave_width = L * wave_width_ratio
    x_start = x_center - wave_width / 2.0
    x_end = x_center + wave_width / 2.0

    # 該当するインデックスを特定
    idx = findall(x -> x_start <= x <= x_end, x_data)

    # サイン波の生成 (ブロードキャストで一括計算)
    arg = π * (x_data[idx] .- x_start) / wave_width
    u0[idx] = U .* sin.(arg)

    return u0
end
# -----------------------------------


# 固定端の境界条件を返す
function get_fixed_boundary_conditions()
    # 固定端 (ディリクレ条件: u=0)
    next_u_left = 0.0
    next_u_right = 0.0
    return next_u_left, next_u_right
end


# 自由端の境界条件を返す
function get_free_boundary_conditions(current_u::AbstractVector{<:Real},
                                      prev_u::AbstractVector{<:Real},
                                      coeff::Real)
    # 自由端 (ノイマン条件: du/dx=0)
    # 左端 (インデックス 0) の更新: u_0^{n+1} = 2u_0^n - u_0^{n-1} + 2C^2 (u_1^n - u_0^n)
    next_u_left = 2.0 * current_u[1] - prev_u[1] +
        2.0 * coeff * (current_u[2] - current_u[1])

    # 右端 (インデックス Nx-1) の更新:
    # u_{N-1}^{n+1} = 2u_{N-1}^n - u_{N-1}^{n-1} + 2C^2 (u_{N-2}^n - u_{N-1}^n)
    next_u_right = 2.0 * current_u[end] - prev_u[end] +
        2.0 * coeff * (current_u[end-1] - current_u[end])
    return next_u_left, next_u_right
end


# 定数
const T_I = 0.0
const T_F = 100.0
const X_MIN = 0.0
const X_MAX = 10.0
const DEL_T = 0.01
const DEL_X = 0.05
const V = 1.0      # 波の速さ

# 差分法係数 (Courant数の二乗 C^2)
const COEFF = (V * DEL_T / DEL_X)^2

# 境界条件の指定
const BOUNDARY_CONDITION = "free"  # "fixed" or "free"

# クーラン条件 (C <= 1) の確認 (ここでは C^2 <= 1)
if COEFF > 1.0
    C = sqrt(COEFF)
    @warn "クーラン数 C が1を超えています ($(round(C, digits=2)))。計算が発散する可能性があります。"
end

const NT = Int64(round((T_F - T_I) / DEL_T))
const NX = Int64(round((X_MAX - X_MIN) / DEL_X)) + 1

U_LIST = zeros(Float64, (NT+2, NX))
X_DATA = range(X_MIN, X_MAX, length=NX) |> collect  # x座標のデータを作成

# メイン処理
# 初期化
const U_AMPLITUDE = 1.0
U_LIST[1, :] = get_intial_state_sin(X_DATA, U_AMPLITUDE)
U_LIST[2, :] = get_intial_state_sin(X_DATA, U_AMPLITUDE)

# 更新処理
for i in 1:NT
    t = i * DEL_T

    @views U_LIST[i+2, 2:NX-1] = 2.0 .* U_LIST[i+1, 2:NX-1] .- U_LIST[i, 2:NX-1] .+
        COEFF .* (U_LIST[i+1, 3:NX] .- 2.0 .* U_LIST[i+1, 2:NX-1] .+ U_LIST[i+1, 1:NX-2])

    # 境界条件
    if BOUNDARY_CONDITION == "fixed"
        U_LIST[i+2, 1], U_LIST[i+2, NX] = get_fixed_boundary_conditions()
    elseif BOUNDARY_CONDITION == "free"
        U_LIST[i+2, 1], U_LIST[i+2, NX] =
            get_free_boundary_conditions(U_LIST[i+1, :], U_LIST[i, :], COEFF)
    else
        throw(DomainError(BOUNDARY_CONDITION,
        "無効な境界条件タイプです。'fixed' または 'free' を指定してください。"))
    end
end

println("シミュレーションが完了しました。全ステップ数: $(NT)")

# アニメーションの作成
# PythonPlotのpyplotとanimationモジュールを取得
# pygui(true) # GUIバックエンドを有効に
# pyimport("matplotlib").pyplot.switch_backend("TkAgg")
const plt = pyimport("matplotlib.pyplot")
const animation = pyimport("matplotlib.animation")
# 間引きレート
const SKIP_STEPS = 5
const N_FRAMES = Int(NT ÷ SKIP_STEPS)

fig, ax = plt.subplots(figsize=(8.0, 6.0))
ax.set_xlabel("x")
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylabel("u")
ax.set_ylim(-U_AMPLITUDE - 0.1, U_AMPLITUDE + 0.1)
ax.grid(true)
ax.set_title("Wave equation in 1 Dimension")

line, = ax.plot(Float64[], Float64[], color="b", label="Amplitude")


function init()
    line.set_data(Float64[], Float64[])
    return (line,)
end

# --- 更新関数 ---
# i はフレームインデックス (0 から N_FRAMES-1)
function update(i)
    data_index = i * SKIP_STEPS + 1
    u_data = U_LIST[data_index, :]
    line.set_data(X_DATA, u_data)

    # 現在の時刻を計算
    current_time_step = data_index - 1 # 1-based index を 0-based time step に戻す
    t = current_time_step * DEL_T

    ax.set_title("Wave equation in 1D (t = $(round(t, digits=2)))")

    return (line,)
end


ani = animation.FuncAnimation(
    fig,
    update,
    frames=N_FRAMES,  # 間引き後のフレーム数を使用
    init_func=init,
    interval=20,  # 指定した数値のms ごとに更新 (アニメーションの速度を調整)
    blit=false
)

ax.legend()

plt.show(block=true)
