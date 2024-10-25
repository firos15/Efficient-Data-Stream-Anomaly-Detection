import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Function to simulate a continuous data stream with trend, seasonal component, and noise
def data_stream(n_pts=1000, seasonal_period=50):
    np.random.seed(42)  # Seed for reproducibility
    # Linear trend growing over time
    trend = np.linspace(0, 10, n_pts)
    
    # Seasonal pattern with specified period
    seasonal = 5 * np.sin(np.linspace(0, 2 * np.pi * n_pts / seasonal_period, n_pts))
    
    # Gaussian noise to simulate randomness in real data
    noise = np.random.normal(0, 1, n_pts)
    data = trend + seasonal + noise
    
    # Introduce anomalies to the data in random (5% in total )
    anomalies = np.random.choice(n_pts, size=int(0.05 * n_pts), replace=False)
    
    #Large spikes for the anomalies to simulate realworld outliers
    data[anomalies] += np.random.normal(20, 5, len(anomalies))
    return data

# Function for detecting anomalies using Isolation forest
# Contamination parameter is set in 5% based on the number of anomalies given
def detect_anomalies(data):
    if len(data) < 2:  # warning if small datasets
        raise ValueError("Data too small for anomaly detection, min= 2 points.")
    
    try:
        # Isolation forest for detecting anomalies
        model = IsolationForest(contamination=0.05)
        data = data.reshape(-1, 1)  # Reshape data for model input
        model.fit(data)  # Trains our model with data 
        predictions = model.predict(data)  # Predict anomalies (-1 indicates anomaly)
    except Exception as e:
        # Error handling 
        print(f"An error occurred during process: {e}")
        return np.array([])  # Return empty array if error occured

    return predictions

# Realtime data stream simulation and anomaly detection with plotting 
def anomaly_detection(n_pts=1000, delay=0.1):
    # Generate data stream
    data = data_stream(n_pts)
    
    plt.ion()  # Interactive mode for live plotting
    fig, ax = plt.subplots()
    
    # Plot customizations
    ax.set_facecolor('#eafff5')  #background color
    line, = ax.plot(data, color='green', label='Data Stream')  # Line plot for data
    scatter = ax.scatter([], [], color='red', label='Anomalies')  # Placeholder for anomalies
    ax.legend()
    
    plt.title(f'Data Stream Anomaly Detection: {n_pts} Points')
    plt.xlabel('Data Point Index')
    plt.ylabel('Value')
    plt.grid(True)

    #Loop for data points to simulate live processing
    for i in range(n_pts):
        current_data = data[:i + 1]
        
        # Start detecting anomalies after a few points for avoiding early noise coming in
        if i > 10:
            # Detect anomalies for the current data subset
            predictions = detect_anomalies(current_data)
            anomalies = np.where(predictions == -1)[0]
            
            # Update scatter plot with detected anomalies
            scatter.set_offsets(np.c_[anomalies, current_data[anomalies]])
            

        # Update the line plot as more data points are coming in
        line.set_ydata(current_data)
        line.set_xdata(np.arange(i + 1))
        ax.relim()
        ax.autoscale_view()  # Scaling the plot and adjust size
        
        # Updates the plot realtime
        plt.draw()
        plt.pause(delay)  # To control the speed of live simulation
    
    plt.ioff()  
    plt.show()

#Run the realtime detection process
anomaly_detection()

