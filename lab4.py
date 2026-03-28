import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_empirical_cdf(dist_name, samples_dict, sizes, dist_obj, limits):
    x_min, x_max = limits
    fig, axes = plt.subplots(1, len(sizes), figsize=(15, 5))
    
    for idx, n in enumerate(sizes):
        ax = axes[idx]
        current_sample = samples_dict[n]
        sorted_x = np.sort(current_sample)
        
        # Обработка границ для Коши
        if dist_name == "Коши":
            plot_min = min(x_min, sorted_x[0])
            plot_max = max(x_max, sorted_x[-1])
        else:
            plot_min, plot_max = x_min, x_max

        x_ecdf = np.concatenate(([-np.inf], sorted_x, [np.inf]))
        y_ecdf = np.concatenate(([0.0], np.arange(1, len(current_sample)+1)/n, [1.0]))
        ax.step(x_ecdf, y_ecdf, where='post', label='Эмпирическая ФР', color='#2980b9', linewidth=2)
        
        counts, bin_edges = np.histogram(current_sample, bins='auto')
        cdf_hist = np.cumsum(counts) / n
        hist_x = np.concatenate(([plot_min], bin_edges, [plot_max]))
        hist_y = np.concatenate(([0.0, 0.0], cdf_hist, [1.0]))
        ax.step(hist_x, hist_y, where='post', label='Гистограмма', color='#7f8c8d', linestyle=':', linewidth=2)

        if dist_name == "Пуассона":
            grid = np.arange(plot_min, plot_max + 1)
            ax.step(grid, dist_obj.cdf(grid), where='post', label='Теоретическая ФР', color='#c0392b', linewidth=2)
        else:
            grid = np.linspace(plot_min, plot_max, 1000)
            ax.plot(grid, dist_obj.cdf(grid), label='Теоретическая ФР', color='#c0392b', linewidth=2)

        # Оформление осей
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'Объем выборки: n = {n}', fontsize=12, fontstyle='italic')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('F(x)', fontsize=10)
        ax.grid(True, alpha=0.4, linestyle='-')
        ax.legend(loc='lower right', fontsize=9)
    
    fig.suptitle(f'Распределение {dist_name}: Функции распределения', fontsize=15, fontweight='bold')
    plt.subplots_adjust(top=0.82)
    plt.tight_layout()
    plt.show()

def plot_kernel_density(dist_name, samples_dict, sizes, dist_obj, limits):
    x_min, x_max = limits
    fig, axes = plt.subplots(1, len(sizes), figsize=(15, 5))
    
    for idx, n in enumerate(sizes):
        ax = axes[idx]
        current_sample = samples_dict[n]

        if dist_name == "Коши":
            def robust_bandwidth(obj):
                q75, q25 = np.percentile(obj.dataset[0], [75, 25])
                iqr = q75 - q25
                iqr = 1.0 if iqr == 0 else iqr
                bw = (iqr / 1.34) * (obj.n ** (-1/5)) 
                std = np.std(obj.dataset[0], ddof=1)
                std = 1.0 if std == 0 else std
                return bw / std 
            kde = stats.gaussian_kde(current_sample, bw_method=robust_bandwidth)
        else:
            kde = stats.gaussian_kde(current_sample)

        if dist_name == "Пуассона":
            grid_discrete = np.arange(x_min, x_max + 1)
            grid_continuous = np.linspace(x_min, x_max, 1000)
            
            kde_vals = kde(grid_continuous)
            ax.plot(grid_continuous, kde_vals, label='KDE (Оценка плотности)', color='#27ae60', linewidth=2)
            ax.fill_between(grid_continuous, kde_vals, alpha=0.15, color='#27ae60')
            ax.plot(grid_discrete, dist_obj.pmf(grid_discrete), 'o--', label='Теоретическая вероятность', color='#8e44ad')
        else:
            grid = np.linspace(x_min, x_max, 1000)
            kde_vals = kde(grid)
            
            ax.plot(grid, kde_vals, label='KDE (Оценка плотности)', color='#27ae60', linewidth=2)
            ax.fill_between(grid, kde_vals, alpha=0.15, color='#27ae60') # Красивая полупрозрачная заливка
            ax.plot(grid, dist_obj.pdf(grid), label='Теоретическая плотность', color='#8e44ad', linestyle='--', linewidth=2)
        
        ax.set_xlim(x_min, x_max)
        ax.set_title(f'Объем выборки: n = {n}', fontsize=12, fontstyle='italic')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('f(x)', fontsize=10)
        ax.grid(True, alpha=0.4, linestyle='-')
        ax.legend(fontsize=9)
    
    fig.suptitle(f'Распределение {dist_name}: Оценка плотности (KDE)', fontsize=15, fontweight='bold')
    plt.subplots_adjust(top=0.82)
    plt.tight_layout()
    plt.show()

def plot_bandwidth_comparison(dist_name, samples_dict, sizes, dist_obj, limits, bw_list=[0.2, 0.5, 'scott', 1.5]):
    x_min, x_max = limits
    fig, axes = plt.subplots(1, len(sizes), figsize=(15, 5))
    
    bw_colors = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6']
    
    for idx, n in enumerate(sizes):
        ax = axes[idx]
        current_sample = samples_dict[n]
        
        if dist_name == "Пуассона":
            grid_discrete = np.arange(x_min, x_max + 1)
            fine_grid = np.linspace(x_min, x_max, 1000)
            ax.plot(grid_discrete, dist_obj.pmf(grid_discrete), 's:', label='Теория', color='black', alpha=0.6)
            x_plot = fine_grid
        else:
            grid = np.linspace(x_min, x_max, 1000)
            ax.plot(grid, dist_obj.pdf(grid), label='Теория', color='black', linewidth=1.5, linestyle='-.')
            x_plot = grid

        for j, bw in enumerate(bw_list):
            kde = stats.gaussian_kde(current_sample, bw_method=bw)
            label_str = f'h = {bw}' if isinstance(bw, str) else f'h-множ. = {bw}'
            
            ax.plot(x_plot, kde(x_plot), label=label_str, color=bw_colors[j % len(bw_colors)], linewidth=1.5, alpha=0.85)
        
        ax.set_xlim(x_min, x_max)
        ax.set_title(f'n = {n}', fontsize=12, fontstyle='italic')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('Плотность', fontsize=10)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=9)
    
    fig.suptitle(f'Распределение {dist_name}: Влияние ширины окна KDE', fontsize=15, fontweight='bold')
    plt.subplots_adjust(top=0.82)
    plt.tight_layout()
    plt.show()

def main():
    # Инициализация генератора
    rng = np.random.default_rng(4)
    
    distributions = ["Нормальное", "Коши", "Лапласа", "Пуассона", "Равномерное"]
    n_sizes = [20, 60, 100]
    
    models = {
        "Нормальное": stats.norm(0, 1),
        "Коши": stats.cauchy(),
        "Лапласа": stats.laplace(0, 1 / np.sqrt(2)),
        "Пуассона": stats.poisson(10),
        "Равномерное": stats.uniform(-np.sqrt(3), 2 * np.sqrt(3))
    }
    
    axis_limits = {"Пуассона": (6, 14)}
    for dist in distributions:
        if dist not in axis_limits:
            axis_limits[dist] = (-4, 4)
    
    generated_data = {d: {s: None for s in n_sizes} for d in distributions}
    for s in n_sizes:
        generated_data["Нормальное"][s] = rng.normal(0, 1, s)
        generated_data["Коши"][s] = rng.standard_cauchy(s)
        generated_data["Лапласа"][s] = rng.laplace(0, 1 / np.sqrt(2), s)
        generated_data["Пуассона"][s] = rng.poisson(10, s)
        generated_data["Равномерное"][s] = rng.uniform(-np.sqrt(3), np.sqrt(3), s)
    
    for dist in distributions:
        plot_empirical_cdf(dist, generated_data[dist], n_sizes, models[dist], axis_limits[dist])
        plot_kernel_density(dist, generated_data[dist], n_sizes, models[dist], axis_limits[dist])
    
    dists_for_bw_check = ["Пуассона", "Лапласа"]
    for dist in dists_for_bw_check:
        plot_bandwidth_comparison(dist, generated_data[dist], n_sizes, models[dist], axis_limits[dist])

if __name__ == "__main__":
    main()