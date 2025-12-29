"""
Red Bull Racing 2025 F1 Season Analysis
Data Storytelling and Visualization Script

This script analyzes Red Bull Racing's performance throughout the 2025 F1 season,
including all 24 Grand Prix races and sprint weekends.

Author: Data Analysis Team
Date: December 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the 2025 F1 database
print("Loading 2025 F1 Season Data...")
df = pd.read_csv('Ok_F1_2025_db.csv')

# Clean and prepare Red Bull Racing data
# According to official sources:
# - Rounds 1-2: Max Verstappen + Liam Lawson
# - Rounds 3-24: Max Verstappen + Yuki Tsunoda (promoted from Racing Bulls)

# Filter for Red Bull Racing team, excluding reserve/test drivers
rb_df = df[df['Team'] == 'Red Bull Racing'].copy()

# Clean driver names (remove non-breaking spaces)
rb_df['DRIVER'] = rb_df['DRIVER'].str.replace('\xa0', ' ', regex=False)

# Focus on race drivers only (exclude reserve driver appearances)
race_drivers = ['Max Verstappen', 'Liam Lawson', 'Yuki Tsunoda']
rb_df = rb_df[rb_df['DRIVER'].isin(race_drivers)].copy()

# Convert Race_Result_Pos to numeric, handling 'NC' (Not Classified) and other non-numeric values
rb_df['Race_Pos_Numeric'] = pd.to_numeric(rb_df['Race_Result_Pos'], errors='coerce')

# Create points per race summary
rb_df['Points'] = pd.to_numeric(rb_df['Total Pts'], errors='coerce')

print(f"Data loaded successfully!")
print(f"Total Red Bull Racing race entries: {len(rb_df)}")
print(f"Drivers: {rb_df['DRIVER'].unique()}")

# ============================================================================
# VISUALIZATION 1: Season Points Progression
# ============================================================================
def create_points_progression():
    """Creates a line chart showing points accumulation throughout the season"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Calculate cumulative points for each driver
    for driver in rb_df['DRIVER'].unique():
        driver_data = rb_df[rb_df['DRIVER'] == driver].sort_values('Round')
        cumulative_points = driver_data['Points'].cumsum()
        rounds = driver_data['Round']
        
        # Plot line
        ax.plot(rounds, cumulative_points, marker='o', linewidth=2.5, 
                markersize=8, label=driver, alpha=0.8)
    
    # Styling
    ax.set_xlabel('Race Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Points', fontsize=12, fontweight='bold')
    ax.set_title('Red Bull Racing: Season Points Progression (2025)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 25)
    
    # Add annotations for driver change
    ax.axvline(x=2.5, color='red', linestyle=':', linewidth=2, alpha=0.6)
    ax.text(2.5, ax.get_ylim()[1] * 0.95, 'Driver Change\n(Lawson → Tsunoda)', 
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('outputs/rb_points_progression.png', dpi=300, bbox_inches='tight')
    print("✓ Points progression chart created")
    return fig

# ============================================================================
# VISUALIZATION 2: Race Results Distribution
# ============================================================================
def create_results_distribution():
    """Creates a heatmap showing finishing positions throughout the season"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prepare data for heatmap
    pivot_data = rb_df.pivot_table(
        index='DRIVER', 
        columns='Round', 
        values='Race_Pos_Numeric', 
        aggfunc='first'
    )
    
    # Heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Finishing Position'}, 
                linewidths=0.5, ax=ax1, vmin=1, vmax=20)
    ax1.set_title('Race Finishing Positions by Round', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Race Round', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Driver', fontsize=11, fontweight='bold')
    
    # Podium finishes count
    podium_data = rb_df[rb_df['Race_Pos_Numeric'] <= 3].groupby('DRIVER').size()
    points_finishes = rb_df[rb_df['Race_Pos_Numeric'] <= 10].groupby('DRIVER').size()
    
    categories = list(podium_data.index)
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, podium_data.values, width, label='Podiums (1st-3rd)', 
                    color='#FFD700', edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, points_finishes.values, width, label='Points Finishes (1st-10th)', 
                    color='#4169E1', edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Driver', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax2.set_title('Podiums and Points Finishes', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=15, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/rb_results_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Results distribution chart created")
    return fig

# ============================================================================
# VISUALIZATION 3: Qualifying vs Race Performance
# ============================================================================
def create_quali_vs_race():
    """Analyzes the gap between qualifying and race positions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prepare data
    quali_race_data = rb_df.copy()
    quali_race_data['Quali_Pos_Numeric'] = pd.to_numeric(quali_race_data['Qualifying_Pos'], errors='coerce')
    quali_race_data = quali_race_data.dropna(subset=['Quali_Pos_Numeric', 'Race_Pos_Numeric'])
    quali_race_data['Position_Change'] = quali_race_data['Quali_Pos_Numeric'] - quali_race_data['Race_Pos_Numeric']
    
    # Position change analysis
    for driver in quali_race_data['DRIVER'].unique():
        driver_data = quali_race_data[quali_race_data['DRIVER'] == driver]
        ax1.scatter(driver_data['Round'], driver_data['Position_Change'], 
                   s=100, alpha=0.7, label=driver, edgecolors='black', linewidth=1)
    
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax1.fill_between([0, 25], 0, 20, alpha=0.1, color='green', label='Positions Gained')
    ax1.fill_between([0, 25], 0, -20, alpha=0.1, color='red', label='Positions Lost')
    
    ax1.set_xlabel('Race Round', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Position Change (Quali → Race)', fontsize=11, fontweight='bold')
    ax1.set_title('Qualifying vs Race Performance', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 25)
    
    # Average qualifying and race positions
    avg_data = quali_race_data.groupby('DRIVER').agg({
        'Quali_Pos_Numeric': 'mean',
        'Race_Pos_Numeric': 'mean'
    }).reset_index()
    
    x = np.arange(len(avg_data))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, avg_data['Quali_Pos_Numeric'], width, 
                    label='Avg Qualifying Position', color='#FF6B6B', edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, avg_data['Race_Pos_Numeric'], width, 
                    label='Avg Race Position', color='#4ECDC4', edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Driver', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Position', fontsize=11, fontweight='bold')
    ax2.set_title('Average Qualifying vs Race Position', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(avg_data['DRIVER'], rotation=15, ha='right')
    ax2.legend(fontsize=10)
    ax2.invert_yaxis()  # Lower position number is better
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('outputs/rb_quali_vs_race.png', dpi=300, bbox_inches='tight')
    print("✓ Qualifying vs race performance chart created")
    return fig

# ============================================================================
# VISUALIZATION 4: Sprint Weekend Performance
# ============================================================================
def create_sprint_analysis():
    """Analyzes performance in sprint weekends"""
    sprint_weekends = rb_df[rb_df['Sprint_Weekend'] == 'Yes'].copy()
    
    if len(sprint_weekends) == 0:
        print("⚠ No sprint weekend data available")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sprint positions
    sprint_weekends['Sprint_Pos_Numeric'] = pd.to_numeric(sprint_weekends['Sprint_Pos'], errors='coerce')
    sprint_weekends = sprint_weekends.dropna(subset=['Sprint_Pos_Numeric'])
    
    # Bar chart of sprint results
    sprint_by_driver = sprint_weekends.groupby(['DRIVER', 'GP'])['Sprint_Pos_Numeric'].first().reset_index()
    
    drivers = sprint_by_driver['DRIVER'].unique()
    gps = sprint_by_driver['GP'].unique()
    
    x = np.arange(len(gps))
    width = 0.35
    
    for i, driver in enumerate(drivers):
        driver_data = sprint_by_driver[sprint_by_driver['DRIVER'] == driver]
        positions = []
        x_positions = []
        
        for j, gp in enumerate(gps):
            gp_data = driver_data[driver_data['GP'] == gp]['Sprint_Pos_Numeric'].values
            if len(gp_data) > 0:
                positions.append(gp_data[0])
                x_positions.append(j + i*width)
        
        if positions:  # Only plot if there's data
            ax1.bar(x_positions, positions, width, label=driver, edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Sprint Weekend', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Sprint Race Position', fontsize=11, fontweight='bold')
    ax1.set_title('Sprint Race Results', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels([gp.replace('R', 'R').split('_')[0] for gp in gps], rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Sprint points contribution
    sprint_weekends['Sprint_Pts_Numeric'] = pd.to_numeric(sprint_weekends['Sprint_Pts'], errors='coerce')
    sprint_points = sprint_weekends.groupby('DRIVER')['Sprint_Pts_Numeric'].sum()
    race_points = sprint_weekends.groupby('DRIVER')['Race_Result_Pts'].sum()
    
    categories = list(sprint_points.index)
    x = np.arange(len(categories))
    
    bars1 = ax2.bar(x, sprint_points.values, label='Sprint Points', 
                    color='#FF6B6B', edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x, race_points.values, bottom=sprint_points.values, 
                    label='Race Points', color='#4ECDC4', edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Driver', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Points', fontsize=11, fontweight='bold')
    ax2.set_title('Points Breakdown in Sprint Weekends', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=15, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add total labels
    for i, (sprint, race) in enumerate(zip(sprint_points.values, race_points.values)):
        total = sprint + race
        ax2.text(i, total, f'{int(total)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/rb_sprint_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Sprint weekend analysis chart created")
    return fig

# ============================================================================
# VISUALIZATION 5: Team Performance Summary Dashboard
# ============================================================================
def create_summary_dashboard():
    """Creates a comprehensive dashboard with key statistics"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Total points
    ax1 = fig.add_subplot(gs[0, 0])
    total_points = rb_df.groupby('DRIVER')['Points'].sum().sort_values(ascending=False)
    bars = ax1.barh(range(len(total_points)), total_points.values, color=['#0600EF', '#DC0000', '#FFD700'])
    ax1.set_yticks(range(len(total_points)))
    ax1.set_yticklabels(total_points.index)
    ax1.set_xlabel('Total Points', fontweight='bold')
    ax1.set_title('Season Points Total', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='x')
    for i, (bar, val) in enumerate(zip(bars, total_points.values)):
        ax1.text(val, i, f' {int(val)}', va='center', fontweight='bold')
    
    # Win/Podium statistics
    ax2 = fig.add_subplot(gs[0, 1])
    wins = rb_df[rb_df['Race_Pos_Numeric'] == 1].groupby('DRIVER').size()
    podiums = rb_df[rb_df['Race_Pos_Numeric'] <= 3].groupby('DRIVER').size()
    
    stats_df = pd.DataFrame({'Wins': wins, 'Podiums': podiums}).fillna(0)
    stats_df.plot(kind='bar', ax=ax2, color=['#FFD700', '#C0C0C0'], 
                  edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Count', fontweight='bold')
    ax2.set_title('Wins & Podiums', fontweight='bold', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.set_xticklabels(stats_df.index, rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # DNF Analysis
    ax3 = fig.add_subplot(gs[0, 2])
    dnf_data = rb_df[rb_df['Race_Result_Pos'] == 'NC'].groupby('DRIVER').size()
    completed = rb_df[rb_df['Race_Result_Pos'] != 'NC'].groupby('DRIVER').size()
    
    if len(dnf_data) > 0:
        completion_df = pd.DataFrame({
            'Completed': completed,
            'DNF': dnf_data
        }).fillna(0)
        completion_df.plot(kind='bar', stacked=True, ax=ax3, 
                           color=['#4ECDC4', '#FF6B6B'], edgecolor='black', linewidth=1.5)
    else:
        completed.plot(kind='bar', ax=ax3, color='#4ECDC4', edgecolor='black', linewidth=1.5)
    
    ax3.set_ylabel('Races', fontweight='bold')
    ax3.set_title('Race Completion Rate', fontweight='bold', fontsize=12)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=15, ha='right')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Points per race
    ax4 = fig.add_subplot(gs[1, :])
    for driver in rb_df['DRIVER'].unique():
        driver_data = rb_df[rb_df['DRIVER'] == driver].sort_values('Round')
        ax4.plot(driver_data['Round'], driver_data['Points'], 
                marker='o', linewidth=2, markersize=6, label=driver, alpha=0.8)
    
    ax4.set_xlabel('Race Round', fontweight='bold')
    ax4.set_ylabel('Points Scored', fontweight='bold')
    ax4.set_title('Points Scored Per Race', fontweight='bold', fontsize=12)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim(0, 25)
    ax4.axvline(x=2.5, color='red', linestyle=':', linewidth=2, alpha=0.5)
    
    # Average finishing position by track type
    ax5 = fig.add_subplot(gs[2, :])
    
    # Categorize tracks (simplified)
    street_circuits = ['R2_china_2025', 'R6_miami_2025', 'R8_monaco_2025', 
                       'R17_azerbaijan_2025', 'R18_singapore_2025', 'R22_las-vegas_2025']
    
    rb_df['Track_Type'] = rb_df['GP'].apply(
        lambda x: 'Street' if x in street_circuits else 'Permanent'
    )
    
    track_type_avg = rb_df.groupby(['DRIVER', 'Track_Type'])['Race_Pos_Numeric'].mean().reset_index()
    
    drivers = track_type_avg['DRIVER'].unique()
    x = np.arange(len(drivers))
    width = 0.35
    
    street_data = track_type_avg[track_type_avg['Track_Type'] == 'Street'].set_index('DRIVER')['Race_Pos_Numeric']
    permanent_data = track_type_avg[track_type_avg['Track_Type'] == 'Permanent'].set_index('DRIVER')['Race_Pos_Numeric']
    
    bars1 = ax5.bar(x - width/2, [street_data.get(d, 0) for d in drivers], 
                    width, label='Street Circuits', color='#FF6B6B', edgecolor='black', linewidth=1.5)
    bars2 = ax5.bar(x + width/2, [permanent_data.get(d, 0) for d in drivers], 
                    width, label='Permanent Circuits', color='#4ECDC4', edgecolor='black', linewidth=1.5)
    
    ax5.set_xlabel('Driver', fontweight='bold')
    ax5.set_ylabel('Average Position', fontweight='bold')
    ax5.set_title('Performance by Track Type', fontweight='bold', fontsize=12)
    ax5.set_xticks(x)
    ax5.set_xticklabels(drivers, rotation=15, ha='right')
    ax5.legend(loc='upper right')
    ax5.invert_yaxis()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add overall title
    fig.suptitle('Red Bull Racing 2025 Season Performance Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig('outputs/rb_summary_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Summary dashboard created")
    return fig

# ============================================================================
# Generate statistics summary
# ============================================================================
def generate_statistics_summary():
    """Generates key statistics for the article"""
    print("\n" + "="*70)
    print("RED BULL RACING 2025 SEASON STATISTICS")
    print("="*70)
    
    for driver in rb_df['DRIVER'].unique():
        driver_data = rb_df[rb_df['DRIVER'] == driver]
        
        print(f"\n{driver}:")
        print(f"  Races: {len(driver_data)}")
        print(f"  Total Points: {driver_data['Points'].sum():.0f}")
        print(f"  Wins: {len(driver_data[driver_data['Race_Pos_Numeric'] == 1])}")
        print(f"  Podiums: {len(driver_data[driver_data['Race_Pos_Numeric'] <= 3])}")
        print(f"  Points Finishes: {len(driver_data[driver_data['Race_Pos_Numeric'] <= 10])}")
        print(f"  Average Finish: {driver_data['Race_Pos_Numeric'].mean():.2f}")
        print(f"  Best Finish: {driver_data['Race_Pos_Numeric'].min():.0f}")
        print(f"  DNFs: {len(driver_data[driver_data['Race_Result_Pos'] == 'NC'])}")
    
    print(f"\nTeam Total Points: {rb_df['Points'].sum():.0f}")
    print(f"Team Wins: {len(rb_df[rb_df['Race_Pos_Numeric'] == 1])}")
    print(f"Team Podiums: {len(rb_df[rb_df['Race_Pos_Numeric'] <= 3])}")
    print("="*70 + "\n")

# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("RED BULL RACING 2025 F1 SEASON - DATA ANALYSIS")
    print("="*70 + "\n")
    
    # Generate all visualizations
    create_points_progression()
    create_results_distribution()
    create_quali_vs_race()
    create_sprint_analysis()
    create_summary_dashboard()
    
    # Generate statistics
    generate_statistics_summary()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("All visualizations saved to outputs/")
    print("="*70 + "\n")
    
    plt.show()
