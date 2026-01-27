import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from .utils import load_results, ensure_dir


def plot_figure1(exp1_results, exp2_results, output_path):
    sparsities = sorted(exp1_results.keys(), reverse=True)
    sparsities_pct = [s * 100 for s in sparsities]

    random_early_stop = []
    random_test_acc = []
    random_early_stop_err = []
    random_test_acc_err = []

    for s in sparsities:
        trials = exp1_results[s]
        early_stops = [t['early_stop_iter'] for t in trials]
        test_accs = [t['early_stop_test_acc'] for t in trials]

        random_early_stop.append(np.mean(early_stops))
        random_test_acc.append(np.mean(test_accs))
        random_early_stop_err.append(np.std(early_stops) if len(trials) > 1 else 0)
        random_test_acc_err.append(np.std(test_accs) if len(trials) > 1 else 0)

    winning_early_stop = []
    winning_test_acc = []
    winning_early_stop_err = []
    winning_test_acc_err = []

    for s in sparsities:
        trials = exp2_results[s]
        early_stops = [t['early_stop_iter'] for t in trials]
        test_accs = [t['early_stop_test_acc'] for t in trials]

        winning_early_stop.append(np.mean(early_stops))
        winning_test_acc.append(np.mean(test_accs))
        winning_early_stop_err.append(np.std(early_stops) if len(trials) > 1 else 0)
        winning_test_acc_err.append(np.std(test_accs) if len(trials) > 1 else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.errorbar(sparsities_pct, random_early_stop, yerr=random_early_stop_err,
                 fmt='o--', linewidth=1, markersize=4, capsize=3,
                 label='Random Sparse', color='#E63946', alpha=0.8)
    ax1.errorbar(sparsities_pct, winning_early_stop, yerr=winning_early_stop_err,
                 fmt='s-', linewidth=1, markersize=4, capsize=3,
                 label='Winning Tickets', color='#2A9D8F', alpha=0.8)
    ax1.set_xlabel('Percent of Weights Remaining (%)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Early-Stop Iteration (Val.)', fontsize=13, fontweight='bold')
    ax1.set_title('Learning Speed vs Sparsity', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11, loc='best')
    ax1.invert_xaxis()

    ax2.errorbar(sparsities_pct, random_test_acc, yerr=random_test_acc_err,
                 fmt='o--', linewidth=1, markersize=4, capsize=3,
                 label='Random Sparse', color='#E63946', alpha=0.8)
    ax2.errorbar(sparsities_pct, winning_test_acc, yerr=winning_test_acc_err,
                 fmt='s-', linewidth=1, markersize=4, capsize=3,
                 label='Winning Tickets', color='#2A9D8F', alpha=0.8)
    ax2.set_xlabel('Percent of Weights Remaining (%)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Validation Accuracy at Early-Stop (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Validation Accuracy vs Sparsity', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11, loc='best')
    ax2.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_figure3(exp3_results, exp5_results, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sparsities = sorted([s for s in exp3_results.keys() if s < 1.0], reverse=True)

    def get_accs(lc):
        return lc.get('val_accs', lc.get('test_accs', []))

    for sparsity in sparsities:
        trials = exp3_results[sparsity]
        curves = []
        for trial in trials:
            lc = trial['learning_curve']
            curves.append(get_accs(lc))

        avg_curve = np.mean(curves, axis=0)
        iterations = trials[0]['learning_curve']['iterations']

        mask = np.array(iterations) <= 15000
        ax1.plot(np.array(iterations)[mask], avg_curve[mask],
                linewidth=1, label=f'{sparsity*100:.1f}%', alpha=0.8)

    if 1.0 in exp3_results:
        trials = exp3_results[1.0]
        curves = [get_accs(t['learning_curve']) for t in trials]
        avg_curve = np.mean(curves, axis=0)
        iterations = trials[0]['learning_curve']['iterations']
        mask = np.array(iterations) <= 15000
        ax1.plot(np.array(iterations)[mask], avg_curve[mask],
                linewidth=1, label='100% (original)', color='black', alpha=0.9)

    ax1.set_xlabel('Training Iterations', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 15000)

    target_sparsities = [0.262, 0.134]
    colors = {'0.262': '#1f77b4', '0.134': '#ff7f0e'}

    for sparsity in target_sparsities:
        color = colors[str(sparsity)]

        if sparsity in exp3_results:
            trials = exp3_results[sparsity]
            curves = [get_accs(t['learning_curve']) for t in trials]
            avg_curve = np.mean(curves, axis=0)
            iterations = trials[0]['learning_curve']['iterations']
            mask = np.array(iterations) <= 15000
            ax2.plot(np.array(iterations)[mask], avg_curve[mask], linewidth=1,
                    label=f'{sparsity*100:.1f}% winning', color=color, alpha=0.9)

        if sparsity in exp5_results:
            trials = exp5_results[sparsity]
            curves = [get_accs(t['learning_curve']) for t in trials]
            avg_curve = np.mean(curves, axis=0)
            iterations = trials[0]['learning_curve']['iterations']
            mask = np.array(iterations) <= 15000
            ax2.plot(np.array(iterations)[mask], avg_curve[mask], linewidth=1, linestyle='--',
                    label=f'{sparsity*100:.1f}% reinit', color=color, alpha=0.7)

    ax2.set_xlabel('Training Iterations', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 15000)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_figure4(exp2_results, exp3_results, exp4_results, exp5_results, output_dir):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    oneshot_sparsities = sorted(exp2_results.keys(), reverse=True)
    iterative_sparsities = sorted(exp3_results.keys(), reverse=True)

    def get_mean_std(trials, key):
        values = [t[key] for t in trials]
        return np.mean(values), np.std(values) if len(values) > 1 else 0

    oneshot_win_early, oneshot_win_early_err = [], []
    oneshot_win_acc, oneshot_win_acc_err = [], []
    oneshot_win_train_acc, oneshot_win_train_acc_err = [], []
    for s in oneshot_sparsities:
        trials = exp2_results[s]
        m, e = get_mean_std(trials, 'early_stop_iter')
        oneshot_win_early.append(m); oneshot_win_early_err.append(e)
        m, e = get_mean_std(trials, 'early_stop_test_acc')
        oneshot_win_acc.append(m); oneshot_win_acc_err.append(e)
        m, e = get_mean_std(trials, 'train_acc_at_early_stop')
        oneshot_win_train_acc.append(m); oneshot_win_train_acc_err.append(e)

    iter_win_early, iter_win_early_err = [], []
    iter_win_acc, iter_win_acc_err = [], []
    iter_win_train_acc, iter_win_train_acc_err = [], []
    for s in iterative_sparsities:
        trials = exp3_results[s]
        m, e = get_mean_std(trials, 'early_stop_iter')
        iter_win_early.append(m); iter_win_early_err.append(e)
        m, e = get_mean_std(trials, 'early_stop_test_acc')
        iter_win_acc.append(m); iter_win_acc_err.append(e)
        m, e = get_mean_std(trials, 'train_acc_at_early_stop')
        iter_win_train_acc.append(m); iter_win_train_acc_err.append(e)

    oneshot_reinit_early, oneshot_reinit_early_err = [], []
    oneshot_reinit_acc, oneshot_reinit_acc_err = [], []
    oneshot_reinit_train_acc, oneshot_reinit_train_acc_err = [], []
    oneshot_reinit_sparsities = []
    for s in oneshot_sparsities:
        if s in exp4_results:
            oneshot_reinit_sparsities.append(s)
            trials = exp4_results[s]
            m, e = get_mean_std(trials, 'early_stop_iter')
            oneshot_reinit_early.append(m); oneshot_reinit_early_err.append(e)
            m, e = get_mean_std(trials, 'early_stop_test_acc')
            oneshot_reinit_acc.append(m); oneshot_reinit_acc_err.append(e)
            m, e = get_mean_std(trials, 'train_acc_at_early_stop')
            oneshot_reinit_train_acc.append(m); oneshot_reinit_train_acc_err.append(e)

    iter_reinit_early, iter_reinit_early_err = [], []
    iter_reinit_acc, iter_reinit_acc_err = [], []
    iter_reinit_train_acc, iter_reinit_train_acc_err = [], []
    iter_reinit_sparsities = sorted(exp5_results.keys(), reverse=True)
    for s in iter_reinit_sparsities:
        trials = exp5_results[s]
        m, e = get_mean_std(trials, 'early_stop_iter')
        iter_reinit_early.append(m); iter_reinit_early_err.append(e)
        m, e = get_mean_std(trials, 'early_stop_test_acc')
        iter_reinit_acc.append(m); iter_reinit_acc_err.append(e)
        m, e = get_mean_std(trials, 'train_acc_at_early_stop')
        iter_reinit_train_acc.append(m); iter_reinit_train_acc_err.append(e)

    ax1.errorbar([s*100 for s in iterative_sparsities], iter_win_early, yerr=iter_win_early_err,
            fmt='o-', linewidth=1, markersize=4, capsize=3, label='Iterative Winning', color='#1f77b4')
    ax1.errorbar([s*100 for s in iter_reinit_sparsities], iter_reinit_early, yerr=iter_reinit_early_err,
            fmt='s--', linewidth=1, markersize=4, capsize=3, label='Iterative Reinit', color='#ff7f0e')
    ax1.errorbar([s*100 for s in oneshot_sparsities], oneshot_win_early, yerr=oneshot_win_early_err,
            fmt='^-', linewidth=1, markersize=4, capsize=3, label='One-shot Winning', color='#2ca02c')
    ax1.errorbar([s*100 for s in oneshot_reinit_sparsities], oneshot_reinit_early, yerr=oneshot_reinit_early_err,
            fmt='v--', linewidth=1, markersize=4, capsize=3, label='One-shot Reinit', color='#d62728')

    ax1.set_xlabel('Percent of Weights Remaining (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Early-Stop Iteration', fontsize=12, fontweight='bold')
    ax1.set_title('Learning Speed', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    ax2.errorbar([s*100 for s in iterative_sparsities], iter_win_acc, yerr=iter_win_acc_err,
            fmt='o-', linewidth=1, markersize=4, capsize=3, label='Iterative Winning', color='#1f77b4')
    ax2.errorbar([s*100 for s in iter_reinit_sparsities], iter_reinit_acc, yerr=iter_reinit_acc_err,
            fmt='s--', linewidth=1, markersize=4, capsize=3, label='Iterative Reinit', color='#ff7f0e')
    ax2.errorbar([s*100 for s in oneshot_sparsities], oneshot_win_acc, yerr=oneshot_win_acc_err,
            fmt='^-', linewidth=1, markersize=4, capsize=3, label='One-shot Winning', color='#2ca02c')
    ax2.errorbar([s*100 for s in oneshot_reinit_sparsities], oneshot_reinit_acc, yerr=oneshot_reinit_acc_err,
            fmt='v--', linewidth=1, markersize=4, capsize=3, label='One-shot Reinit', color='#d62728')

    ax2.set_xlabel('Percent of Weights Remaining (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Test Accuracy at Early-Stop', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    ax3.errorbar([s*100 for s in iterative_sparsities], iter_win_train_acc, yerr=iter_win_train_acc_err,
            fmt='o-', linewidth=1, markersize=4, capsize=3, label='Iterative Winning', color='#1f77b4')
    ax3.errorbar([s*100 for s in iter_reinit_sparsities], iter_reinit_train_acc, yerr=iter_reinit_train_acc_err,
            fmt='s--', linewidth=1, markersize=4, capsize=3, label='Iterative Reinit', color='#ff7f0e')
    ax3.errorbar([s*100 for s in oneshot_sparsities], oneshot_win_train_acc, yerr=oneshot_win_train_acc_err,
            fmt='^-', linewidth=1, markersize=4, capsize=3, label='One-shot Winning', color='#2ca02c')
    ax3.errorbar([s*100 for s in oneshot_reinit_sparsities], oneshot_reinit_train_acc, yerr=oneshot_reinit_train_acc_err,
            fmt='v--', linewidth=1, markersize=4, capsize=3, label='One-shot Reinit', color='#d62728')

    ax3.set_xlabel('Percent of Weights Remaining (%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Training Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Train Accuracy at Early-Stop', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure4a.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    iter_win_final, iter_win_final_err = [], []
    for s in iterative_sparsities:
        trials = exp3_results[s]
        m, e = get_mean_std(trials, 'val_acc_at_final')
        iter_win_final.append(m)
        iter_win_final_err.append(e)

    iter_reinit_final, iter_reinit_final_err = [], []
    for s in iter_reinit_sparsities:
        trials = exp5_results[s]
        m, e = get_mean_std(trials, 'val_acc_at_final')
        iter_reinit_final.append(m)
        iter_reinit_final_err.append(e)

    ax.errorbar([s*100 for s in iterative_sparsities], iter_win_final, yerr=iter_win_final_err,
            fmt='o-', linewidth=1, markersize=4, capsize=3, label='Winning Ticket (Iterative)', color='#1f77b4')
    ax.errorbar([s*100 for s in iter_reinit_sparsities], iter_reinit_final, yerr=iter_reinit_final_err,
            fmt='s--', linewidth=1, markersize=4, capsize=3, label='Random Reinit (Iterative)', color='#ff7f0e')

    ax.set_xlabel('Percent of Weights Remaining (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy at Iteration 50k (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy at End of Training (Iterative)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure4b.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.errorbar([s*100 for s in oneshot_sparsities], oneshot_win_early, yerr=oneshot_win_early_err,
            fmt='^-', linewidth=1, markersize=4, capsize=3, label='Winning Ticket (Oneshot)', color='#2ca02c')
    ax1.errorbar([s*100 for s in oneshot_reinit_sparsities], oneshot_reinit_early, yerr=oneshot_reinit_early_err,
            fmt='v--', linewidth=1, markersize=4, capsize=3, label='Random Reinit (Oneshot)', color='#d62728')

    ax1.set_xlabel('Percent of Weights Remaining (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Early-Stop Iteration (Val.)', fontsize=12, fontweight='bold')
    ax1.set_title('Learning Speed (One-shot)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    ax2.errorbar([s*100 for s in oneshot_sparsities], oneshot_win_acc, yerr=oneshot_win_acc_err,
            fmt='^-', linewidth=1, markersize=4, capsize=3, label='Winning Ticket (Oneshot)', color='#2ca02c')
    ax2.errorbar([s*100 for s in oneshot_reinit_sparsities], oneshot_reinit_acc, yerr=oneshot_reinit_acc_err,
            fmt='v--', linewidth=1, markersize=4, capsize=3, label='Random Reinit (Oneshot)', color='#d62728')

    ax2.set_xlabel('Percent of Weights Remaining (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy at Early-Stop (Test) (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Test Accuracy (One-shot)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure4c.png", dpi=300, bbox_inches='tight')
    plt.close()


def load_experiment_results(metrics_dir, exp_name):
    filepath = f"{metrics_dir}/{exp_name}.pt"
    if os.path.exists(filepath):
        return load_results(filepath)
    return None


def generate_all_figures(config):
    metrics_dir = config['results_dir']
    figures_dir = config['figures_dir']
    ensure_dir(figures_dir)
    exp_results = {}
    required_exps = set()
    if 'fig1' in config:
        required_exps.update(config['fig1'].get('requires', []))
    if 'fig3' in config:
        required_exps.update(config['fig3'].get('requires', []))
    if 'fig4' in config:
        required_exps.update(config['fig4'].get('requires', []))

    exp_files = {
        'exp1': 'exp1_random_sparse',
        'exp2': 'exp2_oneshot_winning',
        'exp3': 'exp3_iterative_winning',
        'exp4': 'exp4_oneshot_reinit',
        'exp5': 'exp5_iterative_reinit'
    }

    for exp_key, exp_file in exp_files.items():
        if exp_key in required_exps:
            result = load_experiment_results(metrics_dir, exp_file)
            if result is not None:
                exp_results[exp_key] = result

    generated = []
    if 'fig1' in config:
        reqs = config['fig1'].get('requires', [])
        if all(exp in exp_results for exp in reqs):
            plot_figure1(exp_results['exp1'], exp_results['exp2'], f"{figures_dir}/figure1.png")
            generated.append('figure1.png')
    if 'fig3' in config:
        reqs = config['fig3'].get('requires', [])
        if all(exp in exp_results for exp in reqs):
            plot_figure3(exp_results['exp3'], exp_results['exp5'], f"{figures_dir}/figure3.png")
            generated.append('figure3.png')
    if 'fig4' in config:
        reqs = config['fig4'].get('requires', [])
        if all(exp in exp_results for exp in reqs):
            plot_figure4(exp_results['exp2'], exp_results['exp3'], exp_results['exp4'], exp_results['exp5'], figures_dir)
            generated.append('figure4a.png')
            generated.append('figure4b.png')
            generated.append('figure4c.png')
    return generated


def main():
    parser = argparse.ArgumentParser(description='Generate lottery ticket figures')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to base results directory (contains metrics/ and figures/)')
    args = parser.parse_args()

    base_dir = args.results.rstrip('/')
    metrics_dir = f"{base_dir}/metrics"
    figures_dir = f"{base_dir}/figures"
    ensure_dir(figures_dir)
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    from configs.config import config

    exp_results = {}

    required_exps = set()
    if 'fig1' in config:
        required_exps.update(config['fig1'].get('requires', []))
    if 'fig3' in config:
        required_exps.update(config['fig3'].get('requires', []))
    if 'fig4' in config:
        required_exps.update(config['fig4'].get('requires', []))

    exp_files = {
        'exp1': 'exp1_random_sparse',
        'exp2': 'exp2_oneshot_winning',
        'exp3': 'exp3_iterative_winning',
        'exp4': 'exp4_oneshot_reinit',
        'exp5': 'exp5_iterative_reinit'
    }

    for exp_key, exp_file in exp_files.items():
        if exp_key in required_exps:
            result = load_experiment_results(metrics_dir, exp_file)
            if result is not None:
                exp_results[exp_key] = result

    generated = []
    if 'fig1' in config:
        reqs = config['fig1'].get('requires', [])
        if all(exp in exp_results for exp in reqs):
            plot_figure1(exp_results['exp1'], exp_results['exp2'], f"{figures_dir}/figure1.png")
            generated.append('figure1.png')
    if 'fig3' in config:
        reqs = config['fig3'].get('requires', [])
        if all(exp in exp_results for exp in reqs):
            plot_figure3(exp_results['exp3'], exp_results['exp5'], f"{figures_dir}/figure3.png")
            generated.append('figure3.png')
    if 'fig4' in config:
        reqs = config['fig4'].get('requires', [])
        if all(exp in exp_results for exp in reqs):
            plot_figure4(exp_results['exp2'], exp_results['exp3'], exp_results['exp4'], exp_results['exp5'], figures_dir)
            generated.append('figure4a.png')
            generated.append('figure4b.png')
            generated.append('figure4c.png')


if __name__ == '__main__':
    main()
