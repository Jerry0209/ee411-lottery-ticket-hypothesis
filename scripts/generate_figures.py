"""
Generate all figures from experiment results.
Usage: python scripts/generate_figures.py --results ./results/minimal/
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import load_results, ensure_dir


def plot_figure1(exp1_results, exp2_results, output_path):
    """
    Figure 1: Overview comparison (Random vs One-Shot Winning Tickets)
    """
    print("\nGenerating Figure 1...")

    # Extract sparsity levels
    sparsities = sorted(exp1_results.keys(), reverse=True)
    sparsities_pct = [s * 100 for s in sparsities]

    # Process random sparse results
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
        # Use standard deviation for error bars (more statistically meaningful)
        random_early_stop_err.append(np.std(early_stops) if len(trials) > 1 else 0)
        random_test_acc_err.append(np.std(test_accs) if len(trials) > 1 else 0)

    # Process winning ticket results
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
        # Use standard deviation for error bars
        winning_early_stop_err.append(np.std(early_stops) if len(trials) > 1 else 0)
        winning_test_acc_err.append(np.std(test_accs) if len(trials) > 1 else 0)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Early-stop iteration (with std dev error bars)
    ax1.errorbar(sparsities_pct, random_early_stop, yerr=random_early_stop_err,
                 fmt='o--', linewidth=2, markersize=7, capsize=5,
                 label='Random Sparse', color='#E63946', alpha=0.8)
    ax1.errorbar(sparsities_pct, winning_early_stop, yerr=winning_early_stop_err,
                 fmt='s-', linewidth=2, markersize=7, capsize=5,
                 label='Winning Tickets', color='#2A9D8F', alpha=0.8)
    ax1.set_xlabel('Percent of Weights Remaining (%)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Early-Stop Iteration (Val.)', fontsize=13, fontweight='bold')
    ax1.set_title('Learning Speed vs Sparsity', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11, loc='best')
    ax1.invert_xaxis()

    # Plot 2: Test accuracy (with std dev error bars)
    ax2.errorbar(sparsities_pct, random_test_acc, yerr=random_test_acc_err,
                 fmt='o--', linewidth=2, markersize=7, capsize=5,
                 label='Random Sparse', color='#E63946', alpha=0.8)
    ax2.errorbar(sparsities_pct, winning_test_acc, yerr=winning_test_acc_err,
                 fmt='s-', linewidth=2, markersize=7, capsize=5,
                 label='Winning Tickets', color='#2A9D8F', alpha=0.8)
    ax2.set_xlabel('Percent of Weights Remaining (%)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Validation Accuracy at Early-Stop (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Validation Accuracy vs Sparsity', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11, loc='best')
    ax2.invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 1 to {output_path}")
    plt.close()


def plot_figure3(exp3_results, exp5_results, output_path):
    """
    Figure 3: Learning curves (3 subplots)
    """
    print("\nGenerating Figure 3...")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Get sparsity levels
    sparsities = sorted([s for s in exp3_results.keys() if s < 1.0], reverse=True)

    # Helper to get accuracy key (handles both old 'test_accs' and new 'val_accs')
    def get_accs(lc):
        return lc.get('val_accs', lc.get('test_accs', []))

    # Subplot 1: Acceleration phase (all sparsities, 0-15K iterations)
    for sparsity in sparsities:
        trials = exp3_results[sparsity]
        curves = []
        for trial in trials:
            lc = trial['learning_curve']
            curves.append(get_accs(lc))

        # Average across trials
        avg_curve = np.mean(curves, axis=0)
        iterations = trials[0]['learning_curve']['iterations']

        # Only plot up to 15K iterations
        mask = np.array(iterations) <= 15000
        ax1.plot(np.array(iterations)[mask], avg_curve[mask],
                linewidth=2, label=f'{sparsity*100:.1f}%', alpha=0.8)

    # Add 100% baseline
    if 1.0 in exp3_results:
        trials = exp3_results[1.0]
        curves = [get_accs(t['learning_curve']) for t in trials]
        avg_curve = np.mean(curves, axis=0)
        iterations = trials[0]['learning_curve']['iterations']
        mask = np.array(iterations) <= 15000
        ax1.plot(np.array(iterations)[mask], avg_curve[mask],
                linewidth=2.5, label='100% (original)', color='black', alpha=0.9)
    
    ax1.set_xlabel('Training Iterations', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Acceleration Phase', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 15000)

    # Subplot 2: Full training (all sparsities)
    for sparsity in sparsities:
        trials = exp3_results[sparsity]
        curves = [get_accs(t['learning_curve']) for t in trials]
        avg_curve = np.mean(curves, axis=0)
        iterations = trials[0]['learning_curve']['iterations']

        ax2.plot(iterations, avg_curve, linewidth=2,
                label=f'{sparsity*100:.1f}%', alpha=0.8)

    if 1.0 in exp3_results:
        trials = exp3_results[1.0]
        curves = [get_accs(t['learning_curve']) for t in trials]
        avg_curve = np.mean(curves, axis=0)
        iterations = trials[0]['learning_curve']['iterations']
        ax2.plot(iterations, avg_curve, linewidth=2.5,
                label='100% (original)', color='black', alpha=0.9)

    ax2.set_xlabel('Training Iterations', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Full Training', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Random reinitialization comparison
    reinit_sparsities = sorted(exp5_results.keys(), reverse=True)
    for sparsity in reinit_sparsities:
        # Winning ticket
        if sparsity in exp3_results:
            trials = exp3_results[sparsity]
            curves = [get_accs(t['learning_curve']) for t in trials]
            avg_curve = np.mean(curves, axis=0)
            iterations = trials[0]['learning_curve']['iterations']
            ax3.plot(iterations, avg_curve, linewidth=2.5,
                    label=f'{sparsity*100:.1f}% winning', alpha=0.9)

        # Random reinit
        trials = exp5_results[sparsity]
        curves = [get_accs(t['learning_curve']) for t in trials]
        avg_curve = np.mean(curves, axis=0)
        iterations = trials[0]['learning_curve']['iterations']
        ax3.plot(iterations, avg_curve, linewidth=2, linestyle='--',
                label=f'{sparsity*100:.1f}% reinit', alpha=0.7)

    # Add 100%
    if 1.0 in exp3_results:
        trials = exp3_results[1.0]
        curves = [get_accs(t['learning_curve']) for t in trials]
        avg_curve = np.mean(curves, axis=0)
        iterations = trials[0]['learning_curve']['iterations']
        ax3.plot(iterations, avg_curve, linewidth=2.5,
                label='100% (original)', color='black', alpha=0.9)

    ax3.set_xlabel('Training Iterations', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Winning vs Random Reinit', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9, loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 3 to {output_path}")
    plt.close()


def plot_figure4(exp2_results, exp3_results, exp4_results, exp5_results, output_dir):
    """
    Generate Figure 4a, 4b, 4c
    """
    print("\nGenerating Figure 4...")

    # Figure 4a: Complete comparison (3 subplots)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Process data for all methods
    oneshot_sparsities = sorted(exp2_results.keys(), reverse=True)
    iterative_sparsities = sorted(exp3_results.keys(), reverse=True)

    # Helper to compute mean and std
    def get_mean_std(trials, key):
        values = [t[key] for t in trials]
        return np.mean(values), np.std(values) if len(values) > 1 else 0

    # One-shot winning (green)
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

    # Iterative winning (blue)
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

    # One-shot reinit (red)
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

    # Iterative reinit (orange)
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

    # Plot 1: Early-stop iteration (with error bars)
    ax1.errorbar([s*100 for s in iterative_sparsities], iter_win_early, yerr=iter_win_early_err,
            fmt='o-', linewidth=2, markersize=6, capsize=4, label='Iterative Winning', color='#1f77b4')
    ax1.errorbar([s*100 for s in iter_reinit_sparsities], iter_reinit_early, yerr=iter_reinit_early_err,
            fmt='s--', linewidth=2, markersize=6, capsize=4, label='Iterative Reinit', color='#ff7f0e')
    ax1.errorbar([s*100 for s in oneshot_sparsities], oneshot_win_early, yerr=oneshot_win_early_err,
            fmt='^-', linewidth=2, markersize=6, capsize=4, label='One-shot Winning', color='#2ca02c')
    ax1.errorbar([s*100 for s in oneshot_reinit_sparsities], oneshot_reinit_early, yerr=oneshot_reinit_early_err,
            fmt='v--', linewidth=2, markersize=6, capsize=4, label='One-shot Reinit', color='#d62728')

    ax1.set_xlabel('Percent of Weights Remaining (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Early-Stop Iteration', fontsize=12, fontweight='bold')
    ax1.set_title('Learning Speed', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    # Plot 2: Test accuracy (with error bars)
    ax2.errorbar([s*100 for s in iterative_sparsities], iter_win_acc, yerr=iter_win_acc_err,
            fmt='o-', linewidth=2, markersize=6, capsize=4, label='Iterative Winning', color='#1f77b4')
    ax2.errorbar([s*100 for s in iter_reinit_sparsities], iter_reinit_acc, yerr=iter_reinit_acc_err,
            fmt='s--', linewidth=2, markersize=6, capsize=4, label='Iterative Reinit', color='#ff7f0e')
    ax2.errorbar([s*100 for s in oneshot_sparsities], oneshot_win_acc, yerr=oneshot_win_acc_err,
            fmt='^-', linewidth=2, markersize=6, capsize=4, label='One-shot Winning', color='#2ca02c')
    ax2.errorbar([s*100 for s in oneshot_reinit_sparsities], oneshot_reinit_acc, yerr=oneshot_reinit_acc_err,
            fmt='v--', linewidth=2, markersize=6, capsize=4, label='One-shot Reinit', color='#d62728')

    ax2.set_xlabel('Percent of Weights Remaining (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Test Accuracy at Early-Stop', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    # Plot 3: Training accuracy (with error bars)
    ax3.errorbar([s*100 for s in iterative_sparsities], iter_win_train_acc, yerr=iter_win_train_acc_err,
            fmt='o-', linewidth=2, markersize=6, capsize=4, label='Iterative Winning', color='#1f77b4')
    ax3.errorbar([s*100 for s in iter_reinit_sparsities], iter_reinit_train_acc, yerr=iter_reinit_train_acc_err,
            fmt='s--', linewidth=2, markersize=6, capsize=4, label='Iterative Reinit', color='#ff7f0e')
    ax3.errorbar([s*100 for s in oneshot_sparsities], oneshot_win_train_acc, yerr=oneshot_win_train_acc_err,
            fmt='^-', linewidth=2, markersize=6, capsize=4, label='One-shot Winning', color='#2ca02c')
    ax3.errorbar([s*100 for s in oneshot_reinit_sparsities], oneshot_reinit_train_acc, yerr=oneshot_reinit_train_acc_err,
            fmt='v--', linewidth=2, markersize=6, capsize=4, label='One-shot Reinit', color='#d62728')

    ax3.set_xlabel('Percent of Weights Remaining (%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Training Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Train Accuracy at Early-Stop', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure4a.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 4a to {output_dir}/figure4a.png")
    plt.close()

    print("✓ Figure 4 generation complete")


def main():
    parser = argparse.ArgumentParser(description='Generate lottery ticket figures')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results directory')
    args = parser.parse_args()
    
    results_dir = args.results
    figures_dir = results_dir.replace('results', 'figures')
    ensure_dir(figures_dir)
    
    print("="*70)
    print("GENERATING LOTTERY TICKET FIGURES")
    print("="*70)
    print(f"Results directory: {results_dir}")
    print(f"Figures directory: {figures_dir}")
    print("="*70)
    
    # Load all results
    print("\nLoading results...")
    exp1 = load_results(f"{results_dir}/exp1_random_sparse.pt")
    exp2 = load_results(f"{results_dir}/exp2_oneshot_winning.pt")
    exp3 = load_results(f"{results_dir}/exp3_iterative_winning.pt")
    exp4 = load_results(f"{results_dir}/exp4_oneshot_reinit.pt")
    exp5 = load_results(f"{results_dir}/exp5_iterative_reinit.pt")
    
    # Generate figures
    plot_figure1(exp1, exp2, f"{figures_dir}/figure1.png")
    plot_figure3(exp3, exp5, f"{figures_dir}/figure3.png")
    plot_figure4(exp2, exp3, exp4, exp5, figures_dir)
    
    print("\n" + "="*70)
    print("✓ All figures generated successfully!")
    print("="*70)
    print(f"\nFigures saved to: {figures_dir}/")
    print("  - figure1.png: Overview comparison")
    print("  - figure3.png: Learning curves")
    print("  - figure4a.png: Complete analysis")


if __name__ == '__main__':
    main()