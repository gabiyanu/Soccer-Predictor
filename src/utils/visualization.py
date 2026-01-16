"""
Visualization Utilities

Plotting functions for match predictions, team analysis, and simulation results.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def check_matplotlib():
    """Check if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def plot_score_matrix(
    score_matrix: np.ndarray,
    home_team: str = "Home",
    away_team: str = "Away",
    max_display: int = 6,
    cmap: str = 'YlOrRd',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Plot score probability matrix as heatmap.
    
    Args:
        score_matrix: 2D probability matrix [home_goals, away_goals]
        home_team: Home team name for label
        away_team: Away team name for label
        max_display: Maximum goals to display
        cmap: Matplotlib colormap
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure if matplotlib available
    """
    check_matplotlib()
    
    # Truncate matrix for display
    display_matrix = score_matrix[:max_display+1, :max_display+1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(display_matrix.T, cmap=cmap, aspect='auto', origin='lower')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Probability')
    
    # Add text annotations
    for i in range(max_display + 1):
        for j in range(max_display + 1):
            prob = display_matrix[i, j]
            if prob > 0.001:
                text = f'{prob:.1%}'
                color = 'white' if prob > display_matrix.max() * 0.6 else 'black'
                ax.text(i, j, text, ha='center', va='center', fontsize=9, color=color)
    
    # Labels and ticks
    ax.set_xlabel(f'{home_team} Goals', fontsize=12)
    ax.set_ylabel(f'{away_team} Goals', fontsize=12)
    ax.set_xticks(range(max_display + 1))
    ax.set_yticks(range(max_display + 1))
    ax.set_title(f'{home_team} vs {away_team}\nScore Probability Matrix', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_outcome_probabilities(
    home_win: float,
    draw: float,
    away_win: float,
    home_team: str = "Home",
    away_team: str = "Away",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Plot match outcome probabilities as bar chart.
    
    Args:
        home_win: Home win probability
        draw: Draw probability
        away_win: Away win probability
        home_team: Home team name
        away_team: Away team name
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    outcomes = [f'{home_team}\nWin', 'Draw', f'{away_team}\nWin']
    probs = [home_win, draw, away_win]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    bars = ax.bar(outcomes, probs, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add percentage labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{prob:.1%}',
            ha='center', va='bottom', fontsize=14, fontweight='bold'
        )
    
    ax.set_ylim(0, max(probs) * 1.15)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'{home_team} vs {away_team}\nMatch Outcome Probabilities', fontsize=14)
    
    # Add gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_elo_history(
    history: Dict[str, List[Tuple[Any, float]]],
    teams: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Plot Elo rating history over time.
    
    Args:
        history: Dict mapping team names to list of (date, rating) tuples
        teams: Optional list of teams to include
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if teams is None:
        teams = list(history.keys())
    
    for team in teams:
        if team in history and history[team]:
            dates, ratings = zip(*history[team])
            ax.plot(range(len(ratings)), ratings, label=team, linewidth=2)
    
    ax.set_xlabel('Matches', fontsize=12)
    ax.set_ylabel('Elo Rating', fontsize=12)
    ax.set_title('Elo Rating History', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_simulation_distribution(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_team: str = "Home",
    away_team: str = "Away",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Plot distribution of simulated goals.
    
    Args:
        home_goals: Array of simulated home goals
        away_goals: Array of simulated away goals
        home_team: Home team name
        away_team: Away team name
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    check_matplotlib()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    max_goals = max(home_goals.max(), away_goals.max())
    bins = np.arange(-0.5, max_goals + 1.5, 1)
    
    # Home goals
    axes[0].hist(home_goals, bins=bins, color='#3498db', edgecolor='black', alpha=0.8)
    axes[0].axvline(home_goals.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {home_goals.mean():.2f}')
    axes[0].set_xlabel('Goals', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'{home_team} Goals Distribution', fontsize=12)
    axes[0].legend()
    
    # Away goals
    axes[1].hist(away_goals, bins=bins, color='#e74c3c', edgecolor='black', alpha=0.8)
    axes[1].axvline(away_goals.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {away_goals.mean():.2f}')
    axes[1].set_xlabel('Goals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'{away_team} Goals Distribution', fontsize=12)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_calibration_curve(
    predicted_probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Plot probability calibration curve.
    
    Args:
        predicted_probs: Predicted probabilities
        outcomes: Binary outcomes
        n_bins: Number of calibration bins
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    check_matplotlib()
    
    from .stats import calibration_curve
    
    result = calibration_curve(predicted_probs, outcomes, n_bins)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Calibration plot
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    
    # Plot calibration with error bars
    mask = result.sample_sizes > 0
    ax1.scatter(
        result.expected_frequencies[mask],
        result.observed_frequencies[mask],
        s=result.sample_sizes[mask] * 2,
        alpha=0.7,
        c='blue'
    )
    ax1.plot(
        result.expected_frequencies[mask],
        result.observed_frequencies[mask],
        'b-', alpha=0.5
    )
    
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('Observed Frequency', fontsize=12)
    ax1.set_title(f'Calibration Curve\nBrier Score: {result.brier_score:.4f}', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    
    # Histogram of predictions
    ax2 = axes[1]
    ax2.hist(predicted_probs, bins=20, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Predictions', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_team_comparison(
    team_stats: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Plot radar chart comparing two teams.
    
    Args:
        team_stats: Dict with team names as keys and stat dicts as values
        metrics: List of metrics to compare
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    check_matplotlib()
    
    if len(team_stats) != 2:
        raise ValueError("Exactly 2 teams required for comparison")
    
    teams = list(team_stats.keys())
    
    if metrics is None:
        metrics = ['attack_strength', 'defense_strength', 'elo_rating', 
                   'possession_avg', 'xg_attack_strength', 'xg_defense_strength']
    
    # Filter to available metrics
    metrics = [m for m in metrics if m in team_stats[teams[0]] and m in team_stats[teams[1]]]
    
    if not metrics:
        raise ValueError("No common metrics found")
    
    # Normalize values for radar chart
    values_1 = []
    values_2 = []
    
    for metric in metrics:
        v1 = team_stats[teams[0]][metric]
        v2 = team_stats[teams[1]][metric]
        
        # Normalize to 0-1 scale
        max_val = max(v1, v2) * 1.2
        min_val = min(v1, v2) * 0.8
        
        if max_val != min_val:
            values_1.append((v1 - min_val) / (max_val - min_val))
            values_2.append((v2 - min_val) / (max_val - min_val))
        else:
            values_1.append(0.5)
            values_2.append(0.5)
    
    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    values_1 += values_1[:1]
    values_2 += values_2[:1]
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    ax.plot(angles, values_1, 'o-', linewidth=2, label=teams[0], color='#3498db')
    ax.fill(angles, values_1, alpha=0.25, color='#3498db')
    
    ax.plot(angles, values_2, 'o-', linewidth=2, label=teams[1], color='#e74c3c')
    ax.fill(angles, values_2, alpha=0.25, color='#e74c3c')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_title(f'{teams[0]} vs {teams[1]}\nTeam Comparison', fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_soccer_pitch(
    figsize: Tuple[int, int] = (12, 8),
    pitch_color: str = '#2d572c',
    line_color: str = 'white'
) -> Tuple[Any, Any]:
    """
    Draw a soccer pitch.
    
    Args:
        figsize: Figure size
        pitch_color: Pitch background color
        line_color: Line color
        
    Returns:
        Tuple of (figure, axes)
    """
    check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(pitch_color)
    
    # Pitch dimensions (normalized to 105x68)
    pitch_length = 105
    pitch_width = 68
    
    # Pitch outline
    ax.plot([0, pitch_length], [0, 0], line_color, linewidth=2)
    ax.plot([0, pitch_length], [pitch_width, pitch_width], line_color, linewidth=2)
    ax.plot([0, 0], [0, pitch_width], line_color, linewidth=2)
    ax.plot([pitch_length, pitch_length], [0, pitch_width], line_color, linewidth=2)
    
    # Center line
    ax.plot([pitch_length/2, pitch_length/2], [0, pitch_width], line_color, linewidth=2)
    
    # Center circle
    center_circle = plt.Circle((pitch_length/2, pitch_width/2), 9.15, 
                                fill=False, color=line_color, linewidth=2)
    ax.add_patch(center_circle)
    ax.plot(pitch_length/2, pitch_width/2, 'o', color=line_color, markersize=3)
    
    # Penalty areas
    # Left
    ax.plot([0, 16.5], [pitch_width/2 - 20.15, pitch_width/2 - 20.15], line_color, linewidth=2)
    ax.plot([0, 16.5], [pitch_width/2 + 20.15, pitch_width/2 + 20.15], line_color, linewidth=2)
    ax.plot([16.5, 16.5], [pitch_width/2 - 20.15, pitch_width/2 + 20.15], line_color, linewidth=2)
    
    # Right
    ax.plot([pitch_length, pitch_length - 16.5], [pitch_width/2 - 20.15, pitch_width/2 - 20.15], line_color, linewidth=2)
    ax.plot([pitch_length, pitch_length - 16.5], [pitch_width/2 + 20.15, pitch_width/2 + 20.15], line_color, linewidth=2)
    ax.plot([pitch_length - 16.5, pitch_length - 16.5], [pitch_width/2 - 20.15, pitch_width/2 + 20.15], line_color, linewidth=2)
    
    # Goal areas
    # Left
    ax.plot([0, 5.5], [pitch_width/2 - 9.15, pitch_width/2 - 9.15], line_color, linewidth=2)
    ax.plot([0, 5.5], [pitch_width/2 + 9.15, pitch_width/2 + 9.15], line_color, linewidth=2)
    ax.plot([5.5, 5.5], [pitch_width/2 - 9.15, pitch_width/2 + 9.15], line_color, linewidth=2)
    
    # Right
    ax.plot([pitch_length, pitch_length - 5.5], [pitch_width/2 - 9.15, pitch_width/2 - 9.15], line_color, linewidth=2)
    ax.plot([pitch_length, pitch_length - 5.5], [pitch_width/2 + 9.15, pitch_width/2 + 9.15], line_color, linewidth=2)
    ax.plot([pitch_length - 5.5, pitch_length - 5.5], [pitch_width/2 - 9.15, pitch_width/2 + 9.15], line_color, linewidth=2)
    
    # Penalty spots
    ax.plot(11, pitch_width/2, 'o', color=line_color, markersize=3)
    ax.plot(pitch_length - 11, pitch_width/2, 'o', color=line_color, markersize=3)
    
    ax.set_xlim(-2, pitch_length + 2)
    ax.set_ylim(-2, pitch_width + 2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig, ax


def create_prediction_report(
    prediction_result: Dict[str, Any],
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Create a comprehensive visual prediction report.
    
    Args:
        prediction_result: Dict with prediction data
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    check_matplotlib()
    
    fig = plt.figure(figsize=(14, 10))
    
    # Title
    home_team = prediction_result.get('home_team', 'Home')
    away_team = prediction_result.get('away_team', 'Away')
    fig.suptitle(f'{home_team} vs {away_team}\nMatch Prediction Report', 
                 fontsize=16, fontweight='bold')
    
    # Outcome probabilities (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    outcomes = ['Home Win', 'Draw', 'Away Win']
    probs = [
        prediction_result.get('home_win_prob', 0.33),
        prediction_result.get('draw_prob', 0.33),
        prediction_result.get('away_win_prob', 0.33)
    ]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = ax1.bar(outcomes, probs, color=colors)
    for bar, prob in zip(bars, probs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{prob:.1%}', ha='center', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, max(probs) * 1.2)
    ax1.set_title('Outcome Probabilities', fontsize=12)
    ax1.set_ylabel('Probability')
    
    # Expected goals (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    teams = [home_team, away_team]
    xg = [
        prediction_result.get('expected_home_goals', prediction_result.get('home_xg', 1.5)),
        prediction_result.get('expected_away_goals', prediction_result.get('away_xg', 1.2))
    ]
    bars = ax2.barh(teams, xg, color=['#3498db', '#e74c3c'])
    for bar, val in zip(bars, xg):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, max(xg) * 1.3)
    ax2.set_title('Expected Goals', fontsize=12)
    ax2.set_xlabel('xG')
    
    # Score distribution (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    score_dist = prediction_result.get('score_distribution', {})
    if score_dist:
        top_scores = sorted(score_dist.items(), key=lambda x: -x[1])[:8]
        scores, probs = zip(*top_scores)
        scores_str = [str(s) if isinstance(s, str) else f"{s[0]}-{s[1]}" for s in scores]
        ax3.bar(scores_str, probs, color='#9b59b6')
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Probability')
        ax3.set_title('Top Score Predictions', fontsize=12)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Markets (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    markets = {
        'BTTS': prediction_result.get('btts_prob', 0.5),
        'Over 1.5': prediction_result.get('over_1_5_prob', 0.7),
        'Over 2.5': prediction_result.get('over_2_5_prob', 0.5),
        'Over 3.5': prediction_result.get('over_3_5_prob', 0.3),
    }
    ax4.barh(list(markets.keys()), list(markets.values()), color='#1abc9c')
    ax4.set_xlim(0, 1)
    ax4.set_xlabel('Probability')
    ax4.set_title('Market Probabilities', fontsize=12)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
