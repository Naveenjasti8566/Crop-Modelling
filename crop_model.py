#!/usr/bin/env python3
"""
Simple Maize Crop Growth Simulation Model

"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import os

def load_parameters(param_file='parameters.csv'):
    """Load simulation parameters from CSV file."""
    try:
        if not os.path.exists(param_file):
            print(f"Warning: {param_file} not found. Creating default parameter file...")
            create_default_parameters(param_file)
        
        params_df = pd.read_csv(param_file)
        print(f"Loaded parameters from: {param_file}")
        
        # Organize parameters by type
        crop_params = {}
        soil_params = {}
        weather_params = {}
        sim_params = {}
        
        for _, row in params_df.iterrows():
            param_type = row['parameter_type']
            param_name = row['parameter_name']
            value = row['value']
            
            # Convert to appropriate data type
            if param_name in ['start_date']:
                # Keep as string for date
                pass
            elif param_name in ['dry_day_probability', 'harvest_index']:
                # Convert to float for ratios
                value = float(value)
            elif param_name in ['days', 'random_seed'] or 'gdd_' in param_name:
                # Convert to int for counts and GDD
                value = int(value)
            else:
                # Convert to float for most parameters
                try:
                    value = float(value)
                except:
                    pass  # Keep as string if conversion fails
            
            # Store in appropriate dictionary
            if param_type == 'crop':
                crop_params[param_name] = value
            elif param_type == 'soil':
                soil_params[param_name] = value
            elif param_type == 'weather':
                weather_params[param_name] = value
            elif param_type == 'simulation':
                sim_params[param_name] = value
        
        print(f"Crop parameters: {len(crop_params)} loaded")
        print(f"Soil parameters: {len(soil_params)} loaded")
        print(f"Weather parameters: {len(weather_params)} loaded")
        print(f"Simulation parameters: {len(sim_params)} loaded")
        
        return crop_params, soil_params, weather_params, sim_params
        
    except Exception as e:
        print(f"Error loading parameters: {e}")
        print("Using default parameters...")
        return get_default_parameters()

def create_default_parameters(filename='parameters.csv'):
    """Create default parameter file."""
    default_data = [
        ['crop', 'base_temp', 10, 'Base temperature for GDD calculation (°C)'],
        ['crop', 'gdd_emergence', 100, 'GDD required for emergence'],
        ['crop', 'gdd_flowering', 800, 'GDD required for flowering'],
        ['crop', 'gdd_maturity', 1400, 'GDD required for maturity'],
        ['crop', 'max_biomass_rate', 300, 'Maximum daily biomass accumulation (kg/ha/day)'],
        ['crop', 'harvest_index', 0.45, 'Grain to total biomass ratio'],
        ['soil', 'field_capacity', 250, 'Maximum soil water storage (mm)'],
        ['soil', 'wilting_point', 100, 'Minimum available soil water (mm)'],
        ['soil', 'initial_water', 200, 'Initial soil water content (mm)'],
        ['weather', 'base_temperature', 25, 'Base temperature for weather generation (°C)'],
        ['weather', 'temp_variation', 5, 'Temperature seasonal variation amplitude (°C)'],
        ['weather', 'rainfall_rate', 15, 'Average rainfall intensity (mm/event)'],
        ['weather', 'dry_day_probability', 0.7, 'Probability of no rainfall (0-1)'],
        ['weather', 'solar_base', 20, 'Base solar radiation (MJ/m²/day)'],
        ['weather', 'solar_variation', 5, 'Solar radiation seasonal variation (MJ/m²/day)'],
        ['simulation', 'days', 120, 'Simulation duration (days)'],
        ['simulation', 'start_date', '2024-04-01', 'Simulation start date'],
        ['simulation', 'random_seed', 42, 'Random seed for reproducible results']
    ]
    
    df = pd.DataFrame(default_data, columns=['parameter_type', 'parameter_name', 'value', 'description'])
    df.to_csv(filename, index=False)
    print(f"Created default parameter file: {filename}")

def get_default_parameters():
    """Return default parameter dictionaries."""
    crop_params = {
        'base_temp': 10, 'gdd_emergence': 100, 'gdd_flowering': 800,
        'gdd_maturity': 1400, 'max_biomass_rate': 300, 'harvest_index': 0.45
    }
    soil_params = {
        'field_capacity': 250, 'wilting_point': 100, 'initial_water': 200
    }
    weather_params = {
        'base_temperature': 25, 'temp_variation': 5, 'rainfall_rate': 15,
        'dry_day_probability': 0.7, 'solar_base': 20, 'solar_variation': 5
    }
    sim_params = {
        'days': 120, 'start_date': '2024-04-01', 'random_seed': 42
    }
    return crop_params, soil_params, weather_params, sim_params

# Load parameters from CSV file
CROP_PARAMS, SOIL_PARAMS, WEATHER_PARAMS, SIM_PARAMS = load_parameters()

def generate_weather_data(days=None):
    """Generate simple weather data for simulation using parameters from CSV."""
    if days is None:
        days = SIM_PARAMS['days']
    
    np.random.seed(SIM_PARAMS['random_seed'])  # Use seed from parameters
    
    # Temperature  parameters
    base_temp = WEATHER_PARAMS['base_temperature']
    temp_variation = WEATHER_PARAMS['temp_variation']
    temps = base_temp + temp_variation * np.sin(np.linspace(0, np.pi, days)) + np.random.normal(0, 3, days)
    
    # Rainfall parameters
    rainfall = np.zeros(days)
    dry_prob = WEATHER_PARAMS['dry_day_probability']
    rain_prob = 1 - dry_prob
    num_rain_days = int(days * rain_prob)
    rain_days = np.random.choice(days, size=num_rain_days, replace=False)
    rainfall_rate = WEATHER_PARAMS['rainfall_rate']
    rainfall[rain_days] = np.random.exponential(rainfall_rate, len(rain_days))
    
    # Solar radiation using weather parameters
    solar_base = WEATHER_PARAMS['solar_base']
    solar_variation = WEATHER_PARAMS['solar_variation']
    solar = solar_base + solar_variation * np.sin(np.linspace(0, np.pi, days)) + np.random.normal(0, 2, days)
    solar = np.maximum(solar, 5)  # Minimum 5 MJ/m²/day
    
    return pd.DataFrame({
        'day': range(1, days+1),
        'temp': temps,
        'rainfall': rainfall,
        'solar': solar
    })

def calculate_gdd(temp, base_temp=10):
    """Calculate Growing Degree Days."""
    return max(0, temp - base_temp)

def get_growth_stage(cumulative_gdd):
    """Determine growth stage based on accumulated GDD."""
    if cumulative_gdd < CROP_PARAMS['gdd_emergence']:
        return 0, "Before Emergence"
    elif cumulative_gdd < CROP_PARAMS['gdd_flowering']:
        return 1, "Vegetative"
    elif cumulative_gdd < CROP_PARAMS['gdd_maturity']:
        return 2, "Flowering"
    else:
        return 3, "Maturity"

def calculate_biomass_increment(gdd_progress, solar, water_stress):
    """Calculate daily biomass increment."""

    if gdd_progress < 0.2:  # Early vegetative
        growth_factor = gdd_progress * 2
    elif gdd_progress < 0.6:  # Late vegetative/early flowering
        growth_factor = 0.8 + 0.4 * (gdd_progress - 0.2) / 0.4
    else:  # Late season decline
        growth_factor = 1.2 - 0.5 * (gdd_progress - 0.6) / 0.4
    
    # Radiation and stress effects
    radiation_effect = min(1.0, solar / 20)
    
    return CROP_PARAMS['max_biomass_rate'] * growth_factor * radiation_effect * water_stress

def update_soil_water(current_water, rainfall, et):
    """Update soil water balance."""
    new_water = current_water + rainfall - et
    
    # Drainage if above field capacity
    if new_water > SOIL_PARAMS['field_capacity']:
        new_water = SOIL_PARAMS['field_capacity']
    
    # Cannot go below wilting point
    if new_water < SOIL_PARAMS['wilting_point']:
        new_water = SOIL_PARAMS['wilting_point']
    
    return new_water

def calculate_water_stress(soil_water):
    """Calculate water stress factor (0-1, where 1 = no stress)."""
    available_water = soil_water - SOIL_PARAMS['wilting_point']
    max_available = SOIL_PARAMS['field_capacity'] - SOIL_PARAMS['wilting_point']
    
    if available_water <= 0:
        return 0.1  # Severe stress
    elif available_water >= max_available:
        return 1.0  # No stress
    else:
        return 0.3 + 0.7 * (available_water / max_available)  # Linear response

def simulate_crop_growth(weather_data):
    """Run the crop growth simulation."""
    
    # Initialize variables
    cumulative_gdd = 0
    total_biomass = 0
    soil_water = SOIL_PARAMS['initial_water']
    
    # Storage for daily results
    results = []
    
    for idx, day_data in weather_data.iterrows():
        day = day_data['day']
        temp = day_data['temp']
        rainfall = day_data['rainfall']
        solar = day_data['solar']
        
        # Calculate GDD
        daily_gdd = calculate_gdd(temp, CROP_PARAMS['base_temp'])
        cumulative_gdd += daily_gdd
        
        # Determine growth stage
        stage_num, stage_name = get_growth_stage(cumulative_gdd)
        
        # Calculate progress through growing season
        gdd_progress = min(1.0, cumulative_gdd / CROP_PARAMS['gdd_maturity'])
        
        # Simple ET calculation (crop coefficient * reference ET)
        if stage_num == 0:
            kc = 0.3
        elif stage_num == 1:
            kc = 0.7
        elif stage_num == 2:
            kc = 1.2
        else:
            kc = 0.8
        
        et = kc * (temp - 5) * 0.5  # Simplified ET formula
        et = max(0, et)
        
        # Update soil water
        soil_water = update_soil_water(soil_water, rainfall, et)
        
        # Cal water stress
        water_stress = calculate_water_stress(soil_water)
        
        # Cal biomass increment (only after emergence)
        if stage_num > 0:
            biomass_increment = calculate_biomass_increment(gdd_progress, solar, water_stress)
            total_biomass += biomass_increment
        else:
            biomass_increment = 0
        
        # Store results
        results.append({
            'day': day,
            'temp': temp,
            'rainfall': rainfall,
            'solar': solar,
            'daily_gdd': daily_gdd,
            'cumulative_gdd': cumulative_gdd,
            'growth_stage': stage_num,
            'stage_name': stage_name,
            'total_biomass': total_biomass,
            'soil_water': soil_water,
            'et': et,
            'water_stress': water_stress,
            'biomass_increment': biomass_increment
        })
    
    return pd.DataFrame(results)

def calculate_final_metrics(results):
    """Calculate final harvest metrics."""
    final_biomass = results['total_biomass'].iloc[-1]  # kg/ha
    grain_yield = final_biomass * CROP_PARAMS['harvest_index']  # kg/ha
    
    total_rainfall = results['rainfall'].sum()
    total_et = results['et'].sum()
    
    # Find key development stages
    emergence_day = results[results['growth_stage'] >= 1]['day'].iloc[0] if len(results[results['growth_stage'] >= 1]) > 0 else 0
    flowering_day = results[results['growth_stage'] >= 2]['day'].iloc[0] if len(results[results['growth_stage'] >= 2]) > 0 else 0
    maturity_day = results[results['growth_stage'] >= 3]['day'].iloc[0] if len(results[results['growth_stage'] >= 3]) > 0 else 120
    
    metrics = {
        'grain_yield_tons_ha': grain_yield / 1000,
        'total_biomass_kg_ha': final_biomass,
        'harvest_index': CROP_PARAMS['harvest_index'],
        'water_use_efficiency': grain_yield / total_et if total_et > 0 else 0,
        'total_rainfall_mm': total_rainfall,
        'total_et_mm': total_et,
        'days_to_emergence': emergence_day,
        'days_to_flowering': flowering_day,
        'days_to_maturity': maturity_day
    }
    
    return metrics

def create_plots(weather_data, results, metrics):
    """Create visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Maize Crop Growth Simulation Results', fontsize=16, fontweight='bold')
    
    # 1. GDD accumulation and growth stages
    ax1 = axes[0, 0]
    ax1.plot(results['day'], results['cumulative_gdd'], 'g-', linewidth=2, label='Cumulative GDD')
    
    # Mark growth stage transitions
    stage_lines = [CROP_PARAMS['gdd_emergence'], CROP_PARAMS['gdd_flowering'], CROP_PARAMS['gdd_maturity']]
    stage_labels = ['Emergence', 'Flowering', 'Maturity']
    colors = ['orange', 'red', 'brown']
    
    for line, label, color in zip(stage_lines, stage_labels, colors):
        ax1.axhline(y=line, color=color, linestyle='--', alpha=0.7, label=label)
    
    ax1.set_xlabel('Days After Sowing')
    ax1.set_ylabel('Cumulative GDD (°C·day)')
    ax1.set_title('GDD Accumulation & Growth Stages')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Biomass accumulation
    ax2 = axes[0, 1]
    ax2.plot(results['day'], results['total_biomass']/1000, 'b-', linewidth=2)
    ax2.fill_between(results['day'], results['total_biomass']/1000, alpha=0.3)
    ax2.set_xlabel('Days After Sowing')
    ax2.set_ylabel('Total Biomass (tons/ha)')
    ax2.set_title('Biomass Accumulation')
    ax2.grid(True, alpha=0.3)
    
    # 3. Water balance
    ax3 = axes[1, 0]
    ax3.plot(results['day'], results['soil_water'], 'c-', linewidth=2, label='Soil Water')
    ax3.axhline(y=SOIL_PARAMS['field_capacity'], color='blue', linestyle='--', alpha=0.7, label='Field Capacity')
    ax3.axhline(y=SOIL_PARAMS['wilting_point'], color='red', linestyle='--', alpha=0.7, label='Wilting Point')
    ax3.set_xlabel('Days After Sowing')
    ax3.set_ylabel('Soil Water (mm)')
    ax3.set_title('Soil Water Balance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Weather and stress
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    # Temperature and rainfall
    ax4.plot(results['day'], results['temp'], 'r-', alpha=0.7, label='Temperature')
    ax4.bar(results['day'], results['rainfall'], alpha=0.6, color='lightblue', label='Rainfall')
    
    # Water stress on secondary axis
    ax4_twin.plot(results['day'], results['water_stress'], 'g-', linewidth=2, label='Water Stress')
    
    ax4.set_xlabel('Days After Sowing')
    ax4.set_ylabel('Temperature (°C) / Rainfall (mm)', color='red')
    ax4_twin.set_ylabel('Water Stress Factor', color='green')
    ax4.set_title('Weather & Water Stress')
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    plt.savefig('results/crop_simulation_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved as: results/crop_simulation_results.png")
    
    return fig

def print_results(metrics):
    """Print final results summary."""
    print("\n" + "="*50)
    print("CROP SIMULATION RESULTS")
    print("="*50)
    print(f"Final Grain Yield:     {metrics['grain_yield_tons_ha']:.2f} tons/ha")
    print(f"Total Biomass:         {metrics['total_biomass_kg_ha']:.0f} kg/ha")
    print(f"Harvest Index:         {metrics['harvest_index']:.2f}")
    print(f"Water Use Efficiency:  {metrics['water_use_efficiency']:.1f} kg grain/mm ET")
    print(f"Total Rainfall:        {metrics['total_rainfall_mm']:.0f} mm")
    print(f"Total Evapotranspiration: {metrics['total_et_mm']:.0f} mm")
    print()
    print("DEVELOPMENT TIMING:")
    print(f"Days to Emergence:     {metrics['days_to_emergence']}")
    print(f"Days to Flowering:     {metrics['days_to_flowering']}")
    print(f"Days to Maturity:      {metrics['days_to_maturity']}")
    print("="*50)

def print_loaded_parameters():
    """Print the currently loaded parameters for verification."""
    print("\n" + "="*60)
    print("LOADED SIMULATION PARAMETERS")
    print("="*60)
    
    print("\nCROP PARAMETERS:")
    for key, value in CROP_PARAMS.items():
        print(f"  {key}: {value}")
    
    print("\nSOIL PARAMETERS:")
    for key, value in SOIL_PARAMS.items():
        print(f"  {key}: {value}")
    
    print("\nWEATHER PARAMETERS:")
    for key, value in WEATHER_PARAMS.items():
        print(f"  {key}: {value}")
    
    print("\nSIMULATION PARAMETERS:")
    for key, value in SIM_PARAMS.items():
        print(f"  {key}: {value}")
    print("="*60)

def main():
    """Main simulation function."""
    print("Starting Maize Crop Growth Simulation...")
    print("Study: Impact of weather variability on crop development")
    
    # Display loaded parameters
    print_loaded_parameters()
    
    # Generate weather data using parameters from CSV
    weather_data = generate_weather_data()
    print(f"\nGenerated weather data for {len(weather_data)} days")
    
    # Run simulation
    results = simulate_crop_growth(weather_data)
    print("Simulation completed successfully")
    
    # Calculate metrics
    metrics = calculate_final_metrics(results)
    
    # Display results
    print_results(metrics)
    
    # Create plots
    fig = create_plots(weather_data, results, metrics)
    
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    results.to_csv('results/simulation_results.csv', index=False)
    weather_data.to_csv('results/weather_data.csv', index=False)
    
    # Save parameter summary for reference
    param_summary = {
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'final_yield_tons_ha': metrics['grain_yield_tons_ha'],
        **CROP_PARAMS,
        **SOIL_PARAMS,
        **WEATHER_PARAMS,
        **SIM_PARAMS
    }
    
    param_df = pd.DataFrame([param_summary])
    param_df.to_csv('results/run_summary.csv', index=False)
    
    print("\nData saved to:")
    print("  - results/simulation_results.csv (daily results)")
    print("  - results/weather_data.csv (weather data)")
    print("  - results/run_summary.csv (parameters and final results)")
    
    return weather_data, results, metrics

if __name__ == "__main__":
    weather_data, results, metrics = main()