from kalman_filter import KalmanFilter
import numpy as np

def test_kalman_filter():
    print("Testing Kalman Filter...")
    
    # Initialize KF
    kf = KalmanFilter(process_noise=0.1, measurement_noise=10.0, initial_state=0.0)
    
    # Simulate a constant velocity object (pos = 0, 10, 20, 30...)
    measurements = [0, 10, 20, 30, 40, 50]
    
    print(f"{'Step':<5} | {'Meas':<10} | {'Pred':<10} | {'Est':<10} | {'Vel':<10}")
    print("-" * 55)
    
    for i, z in enumerate(measurements):
        pred = kf.predict()
        est = kf.update(z)
        vel = kf.get_velocity()
        print(f"{i:<5} | {z:<10.1f} | {pred:<10.1f} | {est:<10.1f} | {vel:<10.1f}")
        
    # Simulate missing measurements (Predict only)
    print("\nSimulating missing data (Predict only)...")
    for i in range(5):
        pred = kf.predict()
        print(f"{i+6:<5} | {'None':<10} | {pred:<10.1f} | {'N/A':<10} | {kf.get_velocity():<10.1f}")

    # Check if prediction projected correctly
    if abs(kf.get_position() - 100.0) < 5.0: # Should be around 100 after 5 steps of +10
        print("\nSUCCESS: Kalman Filter tracked and predicted correctly.")
    else:
        print(f"\nFAILURE: Kalman Filter drifted. Expected ~100, got {kf.get_position()}")

if __name__ == "__main__":
    test_kalman_filter()
