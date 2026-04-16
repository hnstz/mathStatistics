import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def l1_loss_func(params, x, y):
    a, b = params
    return np.sum(np.abs(y - (a + b * x)))

def compute_deviations(a_est, b_est, a_true=2.0, b_true=2.0):
    delta_a = np.abs(a_true - a_est)
    pct_err_a = (delta_a / np.abs(a_true)) * 100
    
    delta_b = np.abs(b_true - b_est)
    pct_err_b = (delta_b / np.abs(b_true)) * 100
    
    return delta_a, pct_err_a, delta_b, pct_err_b

def estimate_coefficients(x, y):
    b_ols, a_ols = np.polyfit(x, y, 1)
    
    res = opt.minimize(l1_loss_func, [a_ols, b_ols], args=(x, y))
    a_lad, b_lad = res.x
    
    return (a_ols, b_ols), (a_lad, b_lad)

def simulate_dataset(size=20, a_true=2.0, b_true=2.0, seed=4):
    x = np.linspace(-1.8, 2.0, size)
    np.random.seed(seed)
    noise = np.random.normal(0, 1, size=size)
    y = a_true + b_true * x + noise
    return x, y

def inject_anomalies(y):
    y_corrupted = y.copy()
    y_corrupted[0] += 10
    y_corrupted[-1] -= 10
    return y_corrupted

def display_comparison_table(title, model_params):
    (a_ols, b_ols), (a_lad, b_lad) = model_params
    
    print(f"\n{title}")
    print("-" * 80)
    print(f"{'Метод':<6} | {'a':<8} | {'delta_a':<8} | {'delta_a, %':<10} | {'b':<8} | {'delta_b':<8} | {'delta_b, %':<10}")
    print("-" * 80)
    
    da_ols, da_p_ols, db_ols, db_p_ols = compute_deviations(a_ols, b_ols)
    print(f"МНК    | {a_ols:<8.3f} | {da_ols:<8.3f} | {da_p_ols:<10.3f} | {b_ols:<8.3f} | {db_ols:<8.3f} | {db_p_ols:<10.3f}")
    
    da_lad, da_p_lad, db_lad, db_p_lad = compute_deviations(a_lad, b_lad)
    print(f"МНМ    | {a_lad:<8.3f} | {da_lad:<8.3f} | {da_p_lad:<10.3f} | {b_lad:<8.3f} | {db_lad:<8.3f} | {db_p_lad:<10.3f}")
    print("-" * 80)

def visualize_models(x, y, model_params, title, has_outliers=False):
    (a_ols, b_ols), (a_lad, b_lad) = model_params
    
    plt.figure(figsize=(9, 6), dpi=100)
    
    plt.scatter(x, y, c='#475569', alpha=0.8, edgecolors='#1e293b', s=55, label='Наблюдения', zorder=5)
    
    if has_outliers:
        plt.scatter([x[0], x[-1]], [y[0], y[-1]], c='#ef4444', marker='*', edgecolors='#991b1b', s=250, zorder=6, label='Аномальные выбросы')
        
    plt.plot(x, 2 + 2*x, color='#10b981', linestyle=':', linewidth=2.5, label='Истинная зависимость', zorder=4)
    plt.plot(x, a_ols + b_ols*x, color='#f59e0b', linestyle='--', linewidth=2.5, label='МНК ($L_2$-регрессия)', zorder=4)
    plt.plot(x, a_lad + b_lad*x, color='#3b82f6', linestyle='-', linewidth=2.5, label='МНМ ($L_1$-регрессия)', zorder=4)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15, color='#1e293b')
    plt.xlabel('Значения признака (X)', fontsize=12, color='#475569')
    plt.ylabel('Целевая переменная (Y)', fontsize=12, color='#475569')
    
    plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=10, loc='best', borderpad=1)
    
    plt.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#94a3b8')
    ax.spines['bottom'].set_color('#94a3b8')
    
    plt.tight_layout()

def run_experiment():
    x, y_base = simulate_dataset()
    y_anomalies = inject_anomalies(y_base)

    params_base = estimate_coefficients(x, y_base)
    params_anomalies = estimate_coefficients(x, y_anomalies)

    display_comparison_table("ВЫБОРКА БЕЗ ВЫБРОСОВ", params_base)
    display_comparison_table("ВЫБОРКА С ВЫБРОСАМИ", params_anomalies)
    
    visualize_models(x, y_base, params_base, 'Сравнение методов регрессии: Идеальные условия')
    visualize_models(x, y_anomalies, params_anomalies, 'Сравнение методов регрессии: Влияние выбросов', has_outliers=True)

    plt.show()

if __name__ == '__main__':
    run_experiment()