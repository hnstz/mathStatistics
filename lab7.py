import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def perform_chi_square_analysis(data_points, alpha_level=0.05):
    n_obs = len(data_points)
    
    mean_est = np.mean(data_points)
    std_est = np.std(data_points, ddof=0)
    
    num_bins = int(np.round(1 + 3.322 * np.log10(n_obs)))
    if num_bins < 4:
        num_bins = 4

    observed_counts, edges = np.histogram(data_points, bins=num_bins)
    
    theoretical_edges = edges.copy()
    theoretical_edges[0] = -np.inf
    theoretical_edges[-1] = np.inf
    
    probabilities = np.zeros(num_bins)
    for i in range(num_bins):
        probabilities[i] = stats.norm.cdf(theoretical_edges[i+1], loc=mean_est, scale=std_est) - \
                           stats.norm.cdf(theoretical_edges[i], loc=mean_est, scale=std_est)
                 
    expected_counts = n_obs * probabilities
    
    chi2_val = np.sum((observed_counts - expected_counts)**2 / expected_counts)
    
    deg_freedom = num_bins - 1 
    if deg_freedom > 0:
        crit_val = stats.chi2.ppf(1 - alpha_level, deg_freedom)
    else:
        crit_val = 0.0
            
    return mean_est, std_est, chi2_val, crit_val, deg_freedom, edges, observed_counts, expected_counts


def render_distribution_chart(data, title_text, subplot_ax):
    m, s, stat, crit, df, b_edges, obs, exp = perform_chi_square_analysis(data)
    
    visual_bins = b_edges.copy()
    visual_bins[0] = min(data) - 0.5
    visual_bins[-1] = max(data) + 0.5
    
    subplot_ax.hist(data, bins=visual_bins, edgecolor='white', alpha=0.7, 
                    density=True, color='#008080', label='Эмп. данные')
    
    x_range = np.linspace(visual_bins[0], visual_bins[-1], 500)
    pdf_values = stats.norm.pdf(x_range, loc=m, scale=s)
    subplot_ax.plot(x_range, pdf_values, color='#FF8C00', lw=2.5, 
                    label=f'N({m:.2f}, {s:.2f})')
    
    for boundary in b_edges[1:-1]:
        subplot_ax.axvline(boundary, color='gray', linestyle=':', alpha=0.4)
        
    subplot_ax.set_title(f"{title_text}\n$\chi^2_{{набл}}$={stat:.2f} | $\chi^2_{{крит}}$={crit:.2f}", 
                         fontsize=10, fontweight='bold')
    subplot_ax.legend(fontsize=8)
    subplot_ax.grid(axis='y', linestyle='--', alpha=0.3)


def run_statistical_experiment():
    gen = np.random.default_rng(6)
    target_alpha = 0.05
    
    datasets = [
        ("Нормальное N(0,1)", gen.normal(0, 1, 100)),
        ("Равномерное", gen.uniform(-np.sqrt(3), np.sqrt(3), 20)),
        ("Лапласа", gen.laplace(0, 1/np.sqrt(2), 20))
    ]

    print("┌" + "─"*95 + "┐")
    print(f"│ {'АНАЛИЗ СООТВЕТСТВИЯ ЗАКОНУ НОРМАЛЬНОГО РАСПРЕДЕЛЕНИЯ (alpha = {target_alpha})':^93} │")
    print("├" + "─"*22 + "┬" + "───" + "┬" + "─────" + "┬" + "─────" + "┬" + "────────" + "┬" + "────────" + "┬" + "─────────────" + "┤")
    print(f"│ {'Тип данных':<20} │ {'n':^3} │ {'μ^':^5} │ {'σ^':^5} │ {'χ² набл':^8} │ {'χ² крит':^8} │ {'Вердикт':^11} │")
    print("├" + "─"*22 + "┼" + "───" + "┼" + "─────" + "┼" + "─────" + "┼" + "────────" + "┼" + "────────" + "┼" + "─────────────" + "┤")
    
    for label, points in datasets:
        mu, sigma, chi_stat, critical, _, _, _, _ = perform_chi_square_analysis(points, target_alpha)
        
        is_normal = "ПРИНЯТА" if chi_stat < critical else "ОТКЛОНЕНА"
        
        print(f"│ {label:<20} │ {len(points):>3} │ {mu:>5.2f} │ {sigma:>5.2f} │ {chi_stat:>8.2f} │ {critical:>8.2f} │ {is_normal:^11} │")

    print("└" + "─"*22 + "┴" + "───" + "┴" + "─────" + "┴" + "─────" + "┴" + "────────" + "┴" + "────────" + "┴" + "─────────────" + "┘")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#f0f0f0')
    
    for idx, (label, points) in enumerate(datasets):
        render_distribution_chart(points, label, axes[idx])
    
    plt.suptitle("Визуализация проверки гипотез о нормальности", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_statistical_experiment()