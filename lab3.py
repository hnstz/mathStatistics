import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def paint_distribution(name, data, sample_sizes):
    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(14, 5))
    
    box_color = '#e74c3c'  
    median_style = dict(color='#2c3e50', linewidth=2.5)
    flier_style = dict(marker='d', markerfacecolor='#95a5a6', markersize=6, alpha=0.7, linestyle='none') 
    whisker_style = dict(color='#34495e', linewidth=1.5, linestyle=':')

    for i, size in enumerate(sample_sizes):
        ax = axes[i]

        ax.boxplot(data[i], patch_artist=True, 
                   boxprops=dict(facecolor=box_color, color='#c0392b', alpha=0.8),
                   medianprops=median_style,
                   flierprops=flier_style,
                   whiskerprops=whisker_style,
                   capprops=dict(color='#34495e', linewidth=1.5))
        
        ax.set_title(f'Объем выборки: n = {size}', fontsize=13, fontstyle='italic')
        ax.set_ylabel('Значение случайной величины', fontsize=11)
        
        ax.set_xticks([]) 
        
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='gray')
        ax.set_axisbelow(True) 
    
    fig.suptitle(f'Анализ распределения: {name}', fontsize=16, family='serif')
    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    plt.show()
    
def count_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return len(outliers)

def main():
    random = np.random.default_rng(5)

    data = { "Нормальное": [], "Коши": [], "Лапласа": [], "Пуассона": [], "Равномерное": [] }
    sample_sizes = [20, 100]
    n_simulations = 1000
    outliers_count_for_simulations = {name: {size: [] for size in sample_sizes} for name in data.keys()}
    
    for j in range(n_simulations):
        for i in sample_sizes:
            data["Нормальное"].append(random.normal(0, 1, i))
            data["Коши"].append(random.standard_cauchy(i))
            data["Лапласа"].append(random.laplace(0, 1/np.sqrt(2), i))
            data["Пуассона"].append(random.poisson(10, i))
            data["Равномерное"].append(random.uniform(-np.sqrt(3), np.sqrt(3), i))
            for name in data:
                outliers_count_for_simulations[name][i].append(count_outliers(data[name][-1]))
                
        if j == 1:
            for name in data:
                paint_distribution(name, data[name], sample_sizes)
                
    print("=== Результаты моделирования ===")
    for name in data:
        print(f"\nРаспределение: {name}")
        for size in sample_sizes:
            mean_outliers = np.mean(np.array(outliers_count_for_simulations[name][size])/size)
            print(f"  Средняя доля выбросов (n={size}): {mean_outliers:.3f}")

if __name__ == "__main__":
    main()