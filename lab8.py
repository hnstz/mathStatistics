import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def get_confidence_intervals(data, alpha=0.05):
    n = len(data)
    mean = np.mean(data)
    s = np.std(data, ddof=1)
    var = s**2
    
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_mean = t_crit * (s / np.sqrt(n))
    mean_ci = (mean - margin_mean, mean + margin_mean)
    
    chi2_low = stats.chi2.ppf(alpha/2, df=n-1)
    chi2_high = stats.chi2.ppf(1 - alpha/2, df=n-1)
    
    var_ci = ((n-1)*var / chi2_high, (n-1)*var / chi2_low)
    
    return mean, var, mean_ci, var_ci


def evaluate_variance_equality(sample1, sample2, alpha=0.05):
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)
    
    f_stat = max(var1, var2) / min(var1, var2)
    
    df1 = len(sample1) - 1 if var1 > var2 else len(sample2) - 1
    df2 = len(sample2) - 1 if var1 > var2 else len(sample1) - 1
    
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    
    is_equal = f_stat < f_crit
    return f_stat, f_crit, is_equal


def visualize_lab_results(samples, labels, results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#f0f0f0')
    
    colors = ['#008080', '#FF8C00'] # Теал и Оранжевый
    
    for i, (data, label) in enumerate(zip(samples, labels)):
        mean, var, m_ci, v_ci = results[i]
        
        axes[i].scatter(range(len(data)), data, color=colors[i], alpha=0.5, s=15, label='Данные')
        
        axes[i].axhline(mean, color='black', linestyle='-', lw=2, label=f'Среднее: {mean:.2f}')
        
        axes[i].fill_between([0, len(data)], m_ci[0], m_ci[1], color=colors[i], alpha=0.2, label='CI для μ')
        
        axes[i].set_title(f"Выборка {label} (n={len(data)})\nCI для σ²: [{v_ci[0]:.2f}, {v_ci[1]:.2f}]", 
                          fontweight='bold', pad=15)
        axes[i].legend(loc='upper right', fontsize=9)
        axes[i].grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("Интервальные оценки параметров распределения", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def run_lab_experiment():
    rng = np.random.default_rng(42) 
    alpha = 0.05
    
    s1 = rng.normal(0, 1, 20)
    s2 = rng.normal(0, 1, 100)
    
    samples = [s1, s2]
    labels = ["N=20", "N=100"]
    all_results = []

    print("┌" + "─"*78 + "┐")
    print(f"│ {'РЕЗУЛЬТАТЫ ИНТЕРВАЛЬНОГО ОЦЕНИВАНИЯ (alpha = {alpha})':^76} │")
    print("├" + "─"*10 + "┬" + "───────" + "┬" + "─────────────" + "┬" + "───────" + "┬" + "───────────────────────" + "┤")
    print(f"│ {'Выборка':^8} │ {'x_avg':^5} │ {'CI для μ':^11} │ {'s^2':^5} │ {'CI для σ^2':^20} │")
    print("├" + "─"*10 + "┼" + "───────" + "┼" + "─────────────" + "┼" + "───────" + "┼" + "───────────────────────" + "┤")

    for i, data in enumerate(samples):
        res = get_confidence_intervals(data, alpha)
        all_results.append(res)
        m, v, m_ci, v_ci = res
        print(f"│ {labels[i]:^8} │ {m:>5.2f} │ [{m_ci[0]:>4.2f}, {m_ci[1]:>4.2f}] │ {v:>5.2f} │ [{v_ci[0]:>7.2f}, {v_ci[1]:>7.2f}] │")
    
    print("└" + "─"*10 + "┴" + "───────" + "┴" + "─────────────" + "┴" + "───────" + "┴" + "───────────────────────" + "┘")

    f_stat, f_crit, h0_accepted = evaluate_variance_equality(s1, s2, alpha)
    
    print("\n" + "="*50)
    print("ПРОВЕРКА ГИПОТЕЗЫ О РАВЕНСТВЕ ДИСПЕРСИЙ (F-ТЕСТ)")
    print("="*50)
    print(f"F-статистика: {f_stat:.4f}")
    print(f"F-критическое: {f_crit:.4f}")
    print(f"Результат: {'H0 ПРИНЯТА (σ1² = σ2²)' if h0_accepted else 'H0 ОТКЛОНЕНА (σ1² ≠ σ2²)'}")
    print("="*50)

    visualize_lab_results(samples, labels, all_results)

if __name__ == "__main__":
    run_lab_experiment()