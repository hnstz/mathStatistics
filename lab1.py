import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calculate_bins(data):
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    if iqr > 0:
        h_fd = 2 * iqr / (len(data) ** (1/3))
        bins = int((max(data) - min(data)) / h_fd)
    else:
        bins = 10
    bins = min(bins, 100)
    unique_values = len(np.unique(data))
    return max(2, min(bins, unique_values))

def get_theoretical(name, x_range):
    if name == "Нормальное": return stats.norm.pdf(x_range, 0, 1)
    elif name == "Коши": return stats.cauchy.pdf(x_range, 0, 1)
    elif name == "Лапласа": return stats.laplace.pdf(x_range, 0, 1/np.sqrt(2))
    elif name == "Пуассона":
        x_disc = np.arange(-2, 12)
        return x_disc, stats.poisson.pmf(x_disc, 5)
    elif name == "Равномерное": return stats.uniform.pdf(x_range, -np.sqrt(3), 2*np.sqrt(3))

def paint_distribution(name, data_list, sample_sizes):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    hist_color = "#24819b"
    true_color = "#6e251d"
    
    for idx, (data, size) in enumerate(zip(data_list, sample_sizes)):
        ax = axes[idx]
        
        if name == "Коши":
            lower, upper = np.percentile(data, [1, 99])
            plot_data = data[(data >= lower) & (data <= upper)]
            label_text = 'Случайное (обрезано 1-99%)'
        else:
            plot_data = data
            label_text = 'Случайное'

        bins_count = calculate_bins(plot_data)

        if name == "Пуассона":
            unique, counts = np.unique(plot_data, return_counts=True)
            ax.bar(unique, counts/len(plot_data), width=0.8, color=hist_color, alpha=0.7, edgecolor='white', label=label_text)
            x_t, y_t = get_theoretical(name, None)
            ax.scatter(x_t, y_t, color=true_color, s=100, zorder=5, label='Теоретическое')
            ax.vlines(x_t, 0, y_t, colors=true_color, alpha=0.5, linewidth=2)
        else:
            ax.hist(plot_data, bins=bins_count, density=True, color=hist_color, alpha=0.7, edgecolor='white', label=label_text)
            x_range = np.linspace(min(plot_data), max(plot_data), 1000)
            ax.plot(x_range, get_theoretical(name, x_range), color=true_color, linewidth=3, label='Теоретическое')
        
        ax.set_title(f'Размер выборки: {size}', fontsize=14)
        ax.set_xlabel('Значение', fontsize=11)
        ax.set_ylabel('Плотность вероятности', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{name} распределение', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def calculate_characteristics(data):
    n = len(data)
    trim_count = int(n * 0.1)
    trimmed = np.sort(data)[trim_count:-trim_count] if trim_count > 0 else data
    return {
        'mean': np.mean(data), 'median': np.median(data),
        'zR': (np.min(data) + np.max(data)) / 2,
        'zQ': (np.percentile(data, 25) + np.percentile(data, 75)) / 2,
        'ztr': np.mean(trimmed)
    }

def print_results_table(dist_name, results, size):
    print(f"\n {dist_name} РАСПРЕДЕЛЕНИЕ (n = {size})".center(69))
    print(f"| {'Оценка':<25} | {'Среднее':>12} | {'Дисперсия':>12} | {'<x> +- sqrt(D)':>14} |")
    print(f"|{'-'*27}|{'-'*14}|{'-'*14}|{'-'*16}|")
    for k in ['mean', 'median', 'zR', 'zQ', 'ztr']:
        m, v = results[k]['mean'], results[k]['variance']
        print(f"| {k:<25} | {m:>12.6f} | {v:>12.6f} | {m:.1f} +- {np.sqrt(v):.1f} |")

def main():
    rng = np.random.default_rng(5)
    rng_cauchy = np.random.default_rng(4)
    sample_sizes = [10, 100, 1000]
    n_simulations = 1000
    
    distributions = {
        'Нормальное': lambda n: rng.normal(0, 1, n),
        'Коши': lambda n: rng_cauchy.standard_cauchy(n),
        'Лапласа': lambda n: rng.laplace(0, 1/np.sqrt(2), n),
        'Пуассона': lambda n: rng.poisson(5, n),
        'Равномерное': lambda n: rng.uniform(-np.sqrt(3), np.sqrt(3), n)
    }

    for name, func in distributions.items():
        for size in sample_sizes:
            data_res = {k: [] for k in ['mean', 'median', 'zR', 'zQ', 'ztr']}
            for _ in range(n_simulations):
                s = func(size)
                vals = calculate_characteristics(s)
                for k in data_res: data_res[k].append(vals[k])
            summary = {k: {'mean': np.mean(v), 'variance': np.var(v, ddof=1)} for k, v in data_res.items()}
            print_results_table(name, summary, size)

    for name, func in distributions.items():
        data_to_plot = [func(s) for s in sample_sizes]
        paint_distribution(name, data_to_plot, sample_sizes)

if __name__ == "__main__":
    main()