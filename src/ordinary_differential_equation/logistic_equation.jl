using Plots # プロット用パッケージ

# 1. ロジスティック方程式の定義
# 微分方程式: dN/dt = r * N * (1 - N/K)
# N: 個体数, t: 時間, r: 増殖率, K: 環境収容力
function logistic_eq(N, r, K)
    return r * N * (1 - N/K)
end

# 2. 厳密解の定義
# 初期個体数をN0として、時間tにおける厳密解 N(t)
function exact_solution(t, N0, r, K)
    return K / (1 + (K/N0 - 1) * exp(-r * t))
end

# 3. 4次のルンゲクッタ法 (RK4) の実装
function solve_rk4(f, N0, r, K, tspan, h)
    # f: 右辺関数 (logistic_eq)
    # N0: 初期値
    # r, K: パラメータ
    # tspan: [t_start, t_end]
    # h: 刻み幅
    
    t_start, t_end = tspan
    times = collect(t_start:h:t_end)
    N_values = Float64[] # 計算結果を格納する配列
    
    N = N0 # 現在のN
    push!(N_values, N)
    
    for t in times[1:end-1]
        # RK4ステップ
        k1 = h * f(N, r, K)
        k2 = h * f(N + k1/2, r, K)
        k3 = h * f(N + k2/2, r, K)
        k4 = h * f(N + k3, r, K)
        
        N = N + (k1 + 2*k2 + 2*k3 + k4) / 6
        push!(N_values, N)
    end
    
    return times, N_values
end

# 4. パラメータ設定と計算実行
# --- パラメータ ---
r = 0.5   # 増殖率
K = 100.0 # 環境収容力
N0 = 10.0 # 初期個体数 (t=0)
tspan = (0.0, 50.0) # 時間範囲 [t_start, t_end]
h = 0.5   # 刻み幅 (hを小さくすると精度が向上しますが、計算時間が増えます)

# --- 数値計算 (RK4) の実行 ---
times_rk4, N_rk4 = solve_rk4(logistic_eq, N0, r, K, tspan, h)

# --- 厳密解の計算 ---
times_exact = range(tspan[1], stop=tspan[2], length=500) # より滑らかなプロットのために細かい時間点を使用
N_exact = [exact_solution(t, N0, r, K) for t in times_exact]

# 5. 結果のプロット
# --- プロット設定 ---
p = plot(
    times_exact, N_exact, 
    label = "exact_solution", 
    linewidth = 3, 
    linecolor = :blue,
    title = "numerical_solution and exact solution of logistic equation",
    xlabel = "time t",
    ylabel = "population N(t)",
    legend = :bottomright
)

# --- 数値解の追加プロット ---
plot!(
    p, 
    times_rk4, N_rk4, 
    label = "RK4 (h=$h)", 
    marker = :circle, 
    markersize = 3, 
    linecolor = :red
)

display(p)

# プロットが表示された後、Enterキーが押されるまでプログラムを停止する
println("Press Enter to close the plot...")
readline()