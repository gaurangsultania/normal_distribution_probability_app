import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.patches as patches

# Set up the page
st.title("Normal Distribution Calculator")
st.write("Explore the relationship between z-scores and probabilities on normal distributions")

# Add inputs for mean and standard deviation
st.header("Distribution Parameters")
col1, col2 = st.columns(2)

with col1:
    mean = st.number_input("Mean (μ):", value=0.0, step=0.1)

with col2:
    std_dev = st.number_input("Standard Deviation (σ):", min_value=0.1, value=1.0, step=0.1)

is_standard = (mean == 0.0 and std_dev == 1.0)
if is_standard:
    st.info("Using Standard Normal Distribution (μ = 0, σ = 1)")
else:
    st.info(f"Using Custom Normal Distribution (μ = {mean}, σ = {std_dev})")
    
    # Add a section to calculate z-score from a value in the custom distribution
    st.header("Calculate Z-score from Value")
    x_value = st.number_input(f"Enter a value from your N({mean}, {std_dev}²) distribution:", 
                             value=mean, step=0.1)
    z_from_x = (x_value - mean) / std_dev
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Value (X): {x_value:.4f}")
        st.write(f"Z-score: {z_from_x:.4f}")
    
    with col2:
        prob_below = stats.norm.cdf(z_from_x)
        st.write(f"P(X < {x_value:.4f}): {prob_below:.4f}")
        st.write(f"Percentile: {prob_below*100:.2f}%")
    
    # Add a small plot to visualize this specific value
    fig, ax = plt.subplots(figsize=(8, 3))
    x_range = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
    y = stats.norm.pdf(x_range, loc=mean, scale=std_dev)
    ax.plot(x_range, y, 'b-')
    
    # Shade area
    x_fill = np.linspace(min(x_range), x_value, 1000)
    y_fill = stats.norm.pdf(x_fill, loc=mean, scale=std_dev)
    ax.fill_between(x_fill, y_fill, color='skyblue', alpha=0.5)
    
    # Add vertical line
    ax.axvline(x=x_value, color='red', linestyle='--')
    
    # Labels
    ax.set_title(f"P(X < {x_value:.2f}) = {prob_below:.4f}")
    ax.set_xlabel("X value")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    st.pyplot(fig)

# Create tabs for the two different calculation methods
tab1, tab2 = st.tabs(["Z-score to Probability", "Probability to Z-score"])

with tab1:
    st.header("Find Probability from Z-score")
    st.write("Use the slider to select a z-score and see the corresponding probability P(Z < z)")
    
    # Create a slider for the z-score
    z_score = st.slider("Z-score", min_value=-3.99, max_value=3.99, value=0.0, step=0.01, key="z_slider")
    
    # Calculate probability
    probability = stats.norm.cdf(z_score)
    
    # Calculate the actual value in the original distribution
    actual_value = z_score * std_dev + mean
    
    # Display statistical information
    st.write(f"Z-score: {z_score:.2f}")
    if not is_standard:
        st.write(f"Corresponding value in your distribution: {actual_value:.4f}")
        st.write(f"Probability P(X < {actual_value:.4f}): {probability:.4f} or {probability*100:.2f}%")
    else:
        st.write(f"Probability P(Z < {z_score:.2f}): {probability:.4f} or {probability*100:.2f}%")
    st.write(f"Percentile: {probability*100:.2f}th")
    
    # Create a function to plot the normal distribution
    def plot_normal_distribution(z_score, mean=0, std=1):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Generate x values for plotting
        if is_standard:
            x = np.linspace(-4, 4, 1000)
            x_label = 'Z-score'
            title = 'Standard Normal Distribution'
            cut_point = z_score
        else:
            # For custom distribution, use appropriate range
            x = np.linspace(mean - 4*std, mean + 4*std, 1000)
            x_label = 'X value'
            title = f'Normal Distribution (μ={mean}, σ={std})'
            cut_point = z_score * std + mean
        
        # Calculate the normal distribution PDF
        y = stats.norm.pdf(x, loc=mean, scale=std)
        
        # Plot the PDF
        ax.plot(x, y, 'b-', linewidth=2)
        
        # Fill the area under the curve to the left of cut_point
        if cut_point > min(x):
            x_fill = np.linspace(min(x), cut_point, 1000)
            y_fill = stats.norm.pdf(x_fill, loc=mean, scale=std)
            ax.fill_between(x_fill, y_fill, color='skyblue', alpha=0.5)
        
        # Add a vertical line at the cut_point
        ax.axvline(x=cut_point, color='red', linestyle='--')
        
        # Add text with the probability
        prob = stats.norm.cdf(z_score)
        if is_standard:
            ax.text(0.05, 0.95, f'P(Z < {z_score:.2f}) = {prob:.4f}', 
                    transform=ax.transAxes, fontsize=12, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            ax.text(0.05, 0.95, f'P(X < {cut_point:.2f}) = {prob:.4f}', 
                    transform=ax.transAxes, fontsize=12, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Set labels and grid
        ax.set_xlabel(x_label)
        ax.set_ylabel('Probability Density')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set appropriate y-limit
        ax.set_ylim(0, max(y) * 1.1)
        
        return fig
    
    # Display the normal distribution plot
    st.pyplot(plot_normal_distribution(z_score, mean, std_dev))

with tab2:
    st.header("Find Z-score from Probability")
    st.write("Enter a probability value (0-1) to find the corresponding z-score and value")
    
    # Input for probability
    prob_input = st.number_input("Enter probability (0-1):", 
                                min_value=0.0001, max_value=0.9999, 
                                value=0.95, step=0.01, format="%.4f",
                                key="prob_input")
    
    # Calculate z-score from probability
    calculated_z = stats.norm.ppf(prob_input)
    
    # Calculate the actual value in the original distribution
    calculated_value = calculated_z * std_dev + mean
    
    # Display the result
    st.write(f"For P(X < x) = {prob_input:.4f} or {prob_input*100:.2f}%:")
    st.write(f"The z-score is: {calculated_z:.4f}")
    if not is_standard:
        st.write(f"The value in your distribution is: {calculated_value:.4f}")
    
    # Common probability values
    st.write("### Common Critical Values")
    common_probs = {
        "90% (0.90)": {"z": stats.norm.ppf(0.90), "x": stats.norm.ppf(0.90) * std_dev + mean},
        "95% (0.95)": {"z": stats.norm.ppf(0.95), "x": stats.norm.ppf(0.95) * std_dev + mean},
        "97.5% (0.975)": {"z": stats.norm.ppf(0.975), "x": stats.norm.ppf(0.975) * std_dev + mean},
        "99% (0.99)": {"z": stats.norm.ppf(0.99), "x": stats.norm.ppf(0.99) * std_dev + mean},
        "99.5% (0.995)": {"z": stats.norm.ppf(0.995), "x": stats.norm.ppf(0.995) * std_dev + mean}
    }
    
    for label, values in common_probs.items():
        if is_standard:
            st.write(f"{label}: z = {values['z']:.4f}")
        else:
            st.write(f"{label}: z = {values['z']:.4f}, x = {values['x']:.4f}")
    
    # Create a function to plot the normal distribution with probability
    def plot_normal_distribution_from_prob(prob, mean=0, std=1):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate the corresponding z-score
        z_value = stats.norm.ppf(prob)
        
        # Generate x values for plotting
        if is_standard:
            x = np.linspace(-4, 4, 1000)
            x_label = 'Z-score'
            title = 'Standard Normal Distribution'
            cut_point = z_value
        else:
            # For custom distribution, use appropriate range
            x = np.linspace(mean - 4*std, mean + 4*std, 1000)
            x_label = 'X value'
            title = f'Normal Distribution (μ={mean}, σ={std})'
            cut_point = z_value * std + mean
        
        # Calculate the normal distribution PDF
        y = stats.norm.pdf(x, loc=mean, scale=std)
        
        # Plot the PDF
        ax.plot(x, y, 'b-', linewidth=2)
        
        # Fill the area under the curve to the left of cut_point
        if cut_point > min(x):
            x_fill = np.linspace(min(x), cut_point, 1000)
            y_fill = stats.norm.pdf(x_fill, loc=mean, scale=std)
            ax.fill_between(x_fill, y_fill, color='skyblue', alpha=0.5)
        
        # Add a vertical line at the cut_point
        ax.axvline(x=cut_point, color='red', linestyle='--')
        
        # Add text with the probability and value
        if is_standard:
            ax.text(0.05, 0.95, f'P(Z < {z_value:.4f}) = {prob:.4f}', 
                    transform=ax.transAxes, fontsize=12, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            ax.text(0.05, 0.95, f'P(X < {cut_point:.4f}) = {prob:.4f}', 
                    transform=ax.transAxes, fontsize=12, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Set labels and grid
        ax.set_xlabel(x_label)
        ax.set_ylabel('Probability Density')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set appropriate y-limit
        ax.set_ylim(0, max(y) * 1.1)
        
        return fig
    
    # Display the normal distribution plot
    st.pyplot(plot_normal_distribution_from_prob(prob_input, mean, std_dev))

# Add explanation section about z-scores and transformation
st.write("### About Normal Distributions and Z-scores")
st.write("""
- A z-score represents how many standard deviations a data point is from the mean.
- To convert from a normal distribution X ~ N(μ, σ²) to the standard normal Z ~ N(0, 1):
  - Z = (X - μ) / σ
- To convert from a z-score back to the original distribution:
  - X = Z × σ + μ
- Key percentiles on the standard normal distribution:
  - Z = 0 corresponds to the 50th percentile (the mean)
  - Z = ±1 corresponds to approximately the 16th and 84th percentiles
  - Z = ±1.96 corresponds to approximately the 2.5th and 97.5th percentiles (95% confidence interval)
  - Z = ±2.58 corresponds to approximately the 0.5th and 99.5th percentiles (99% confidence interval)
""")

if not is_standard:
    st.write(f"""
    ### For Your Current Distribution N({mean}, {std_dev}²)
    - Mean (μ) = {mean}
    - Standard Deviation (σ) = {std_dev}
    - To convert to a z-score: Z = (X - {mean}) / {std_dev}
    - To convert from a z-score: X = Z × {std_dev} + {mean}
    """)