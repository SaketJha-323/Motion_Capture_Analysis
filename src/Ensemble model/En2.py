import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class ComprehensiveAnomalyDetector:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.gpr_models = {}
        self.isolation_forest = None
        self.one_class_svm = None
        self.lstm_autoencoder = None
        
    def calculate_velocity(self, position, times):
        """Calculate velocity from position data"""
        return np.diff(position) / np.diff(times)
    
    def prepare_drone_data(self, file_path):
        """Load and prepare drone data"""
        data = pd.read_csv(file_path)
        data.columns = ['Frame', 'Time', 'X', 'Y', 'Z']
        
        # Convert to meters and calculate velocities
        times = data['Time'].values
        x = data['X'].values / 100
        y = data['Y'].values / 100
        z = data['Z'].values / 100
        
        # Calculate velocities
        vx = self.calculate_velocity(x, times)
        vy = self.calculate_velocity(y, times)
        vz = self.calculate_velocity(z, times)
        
        # Important: Adjust times array to match velocity array length
        times = times[1:]  # Remove first element since velocities have n-1 elements
        
        # Ensure all arrays are same size
        min_length = min(len(vx), len(vy), len(vz))
        return {
            'vx': vx[:min_length],
            'vy': vy[:min_length],
            'vz': vz[:min_length],
            'times': times[:min_length]
        }
    
    def train_gpr_models(self, velocities):
        """Train GPR models for each velocity component"""
        X_train = np.arange(len(velocities['vx'])).reshape(-1, 1)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        
        for component in ['vx', 'vy', 'vz']:
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                alpha=1e-2
            )
            self.gpr_models[component] = gpr.fit(X_train, velocities[component])
    
    def get_gpr_anomalies(self, velocities):
        """Detect anomalies using GPR"""
        X_train = np.arange(len(velocities['vx'])).reshape(-1, 1)
        anomalies = {}
        predictions = {}
        uncertainties = {}
        
        for component in ['vx', 'vy', 'vz']:
            pred, sigma = self.gpr_models[component].predict(X_train, return_std=True)
            error = np.abs(pred - velocities[component])
            threshold = np.percentile(error, 95)
            anomalies[component] = error > threshold
            predictions[component] = pred
            uncertainties[component] = sigma
            
        return anomalies, predictions, uncertainties
    
    def train_ensemble_models(self, data):
        """Train ensemble of anomaly detection models"""
        scaled_data = self.scaler.fit_transform(data)
        
        # Isolation Forest
        self.isolation_forest = IsolationForest(contamination=0.05, random_state=42)
        self.isolation_forest.fit(scaled_data)
        
        # One-Class SVM
        self.one_class_svm = OneClassSVM(kernel='rbf', nu=0.05)
        self.one_class_svm.fit(scaled_data)
        
        # LSTM Autoencoder
        sequence_length = 10
        n_features = scaled_data.shape[1]
        
        self.lstm_autoencoder = Sequential([
            LSTM(128, activation='relu', input_shape=(sequence_length-1, n_features), return_sequences=True),
            Dropout(0.2),
            LSTM(64, activation='relu', return_sequences=False),
            RepeatVector(sequence_length-1),
            LSTM(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(128, activation='relu', return_sequences=True),
            TimeDistributed(Dense(n_features))
        ])
        
        self.lstm_autoencoder.compile(optimizer='adam', loss='mse')
        
        # Prepare sequences for LSTM
        sequences = np.array([scaled_data[i:i+sequence_length] 
                            for i in range(len(scaled_data)-sequence_length+1)])
        X_train = sequences[:, :-1]
        self.lstm_autoencoder.fit(X_train, X_train, epochs=20, batch_size=32, verbose=0)
    
    def get_ensemble_anomalies(self, data):
        """Get anomalies from ensemble models"""
        scaled_data = self.scaler.transform(data)
        
        # Get scores from each model
        if_scores = self.isolation_forest.score_samples(scaled_data)
        svm_scores = self.one_class_svm.score_samples(scaled_data)
        
        # Get LSTM reconstruction error
        sequence_length = 10
        sequences = np.array([scaled_data[i:i+sequence_length] 
                            for i in range(len(scaled_data)-sequence_length+1)])
        X_test = sequences[:, :-1]
        reconstructed = self.lstm_autoencoder.predict(X_test)
        lstm_scores = np.mean(np.square(X_test - reconstructed), axis=(1, 2))
        
        # Ensure all score arrays are the same length by truncating the longer ones
        min_length = min(len(if_scores), len(svm_scores), len(lstm_scores))
        if_scores = if_scores[:min_length]
        svm_scores = svm_scores[:min_length]
        lstm_scores = lstm_scores[:min_length]
        
        # Normalize scores
        if_scores = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())
        svm_scores = (svm_scores - svm_scores.min()) / (svm_scores.max() - svm_scores.min())
        lstm_scores = (lstm_scores - lstm_scores.min()) / (lstm_scores.max() - lstm_scores.min())
        
        # Combine scores
        combined_scores = (if_scores + svm_scores + lstm_scores) / 3
        threshold = np.percentile(combined_scores, 95)
        
        return combined_scores > threshold, combined_scores
        
    def visualize_results(self, velocities, gpr_anomalies, gpr_predictions, 
                        gpr_uncertainties, ensemble_anomalies, ensemble_scores):
        """Create comprehensive visualizations"""
        # First, find the minimum length among all arrays
        min_length = min(
            len(ensemble_anomalies),
            len(velocities['vx']),
            len(gpr_anomalies['vx']),
            len(gpr_predictions['vx']),
            len(gpr_uncertainties['vx'])
        )
        
        # Truncate all arrays to the same length
        truncated_velocities = {
            component: velocities[component][:min_length]
            for component in ['vx', 'vy', 'vz']
        }
        
        truncated_gpr_anomalies = {
            component: gpr_anomalies[component][:min_length]
            for component in ['vx', 'vy', 'vz']
        }
        
        truncated_predictions = {
            component: gpr_predictions[component][:min_length]
            for component in ['vx', 'vy', 'vz']
        }
        
        truncated_uncertainties = {
            component: gpr_uncertainties[component][:min_length]
            for component in ['vx', 'vy', 'vz']
        }
        
        # Truncate ensemble anomalies
        ensemble_anomalies = ensemble_anomalies[:min_length]
        
        # Create 2x2 subplot
        fig = plt.figure(figsize=(20, 15))
        
        # 1. GPR Predictions and Anomalies for each component
        components = ['vx', 'vy', 'vz']
        titles = ['X Velocity', 'Y Velocity', 'Z Velocity']
        
        for i, (component, title) in enumerate(zip(components, titles)):
            ax = fig.add_subplot(2, 2, i+1)
            ax.plot(truncated_velocities[component], label='Actual', alpha=0.6)
            ax.plot(truncated_predictions[component], label='GPR Prediction', alpha=0.6)
            ax.fill_between(
                range(len(truncated_uncertainties[component])),
                truncated_predictions[component] - 2 * truncated_uncertainties[component],
                truncated_predictions[component] + 2 * truncated_uncertainties[component],
                alpha=0.2, label='Uncertainty'
            )
            ax.scatter(
                np.where(truncated_gpr_anomalies[component])[0],
                truncated_velocities[component][truncated_gpr_anomalies[component]], 
                color='red', label='GPR Anomalies'
            )
            ax.set_title(f'GPR Analysis - {title}')
            ax.legend()
        
        # 2. 3D Visualization with both GPR and Ensemble anomalies
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        
        # Plot normal points
        normal_mask = ~(ensemble_anomalies | truncated_gpr_anomalies['vx'] | 
                        truncated_gpr_anomalies['vy'] | truncated_gpr_anomalies['vz'])
        ax.scatter(
            truncated_velocities['vx'][normal_mask],
            truncated_velocities['vy'][normal_mask],
            truncated_velocities['vz'][normal_mask],
            c='blue', label='Normal', alpha=0.6
        )
        
        # Plot GPR anomalies
        gpr_mask = (truncated_gpr_anomalies['vx'] | truncated_gpr_anomalies['vy'] | 
                    truncated_gpr_anomalies['vz']) & ~ensemble_anomalies
        ax.scatter(
            truncated_velocities['vx'][gpr_mask],
            truncated_velocities['vy'][gpr_mask],
            truncated_velocities['vz'][gpr_mask],
            c='red', label='GPR Anomalies', alpha=0.8
        )
        
        # Plot Ensemble anomalies
        ensemble_mask = ensemble_anomalies & ~(truncated_gpr_anomalies['vx'] | 
                                            truncated_gpr_anomalies['vy'] | 
                                            truncated_gpr_anomalies['vz'])
        ax.scatter(
            truncated_velocities['vx'][ensemble_mask],
            truncated_velocities['vy'][ensemble_mask],
            truncated_velocities['vz'][ensemble_mask],
            c='yellow', label='Ensemble Anomalies', alpha=0.8
        )
        
        # Plot points detected by both methods
        both_mask = ensemble_anomalies & (truncated_gpr_anomalies['vx'] | 
                                        truncated_gpr_anomalies['vy'] | 
                                        truncated_gpr_anomalies['vz'])
        ax.scatter(
            truncated_velocities['vx'][both_mask],
            truncated_velocities['vy'][both_mask],
            truncated_velocities['vz'][both_mask],
            c='purple', label='Both Methods', alpha=0.8
        )
        
        ax.set_xlabel('X Velocity')
        ax.set_ylabel('Y Velocity')
        ax.set_zlabel('Z Velocity')
        ax.set_title('3D Visualization of Anomalies')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n=== Anomaly Detection Summary ===")
        print(f"Total points analyzed: {len(truncated_velocities['vx'])}")
        print(f"GPR anomalies: {np.sum(truncated_gpr_anomalies['vx'] | truncated_gpr_anomalies['vy'] | truncated_gpr_anomalies['vz'])}")
        print(f"Ensemble anomalies: {np.sum(ensemble_anomalies)}")
        print(f"Anomalies detected by both methods: {np.sum(both_mask)}")
        
        # Print detailed anomaly report
        self.print_anomaly_report(truncated_velocities, truncated_gpr_anomalies, ensemble_anomalies)


        
    def print_anomaly_report(self, velocities, gpr_anomalies, ensemble_anomalies):
        """Print detailed analysis of detected anomalies"""
        print("\n=== Detailed Anomaly Report ===")
        
        # Analyze each anomaly point
        all_anomalies = np.where(ensemble_anomalies | gpr_anomalies['vx'] | 
                                gpr_anomalies['vy'] | gpr_anomalies['vz'])[0]
        
        for idx in all_anomalies:
            print(f"\nAnomaly at index {idx}:")
            print(f"Velocities: X={velocities['vx'][idx]:.3f}, "
                  f"Y={velocities['vy'][idx]:.3f}, Z={velocities['vz'][idx]:.3f}")
            
            # Determine which method(s) detected it
            methods = []
            if ensemble_anomalies[idx]:
                methods.append("Ensemble")
            if gpr_anomalies['vx'][idx] or gpr_anomalies['vy'][idx] or gpr_anomalies['vz'][idx]:
                methods.append("GPR")
            
            print(f"Detected by: {', '.join(methods)}")
            
            # Analyze the type of anomaly
            components = []
            if gpr_anomalies['vx'][idx]:
                components.append("X velocity")
            if gpr_anomalies['vy'][idx]:
                components.append("Y velocity")
            if gpr_anomalies['vz'][idx]:
                components.append("Z velocity")
                
            if components:
                print(f"Anomalous components: {', '.join(components)}")
            
            print("-" * 40)

def main():
    # Initialize detector
    detector = ComprehensiveAnomalyDetector()
    
    # Load and prepare data
    file_path = 'data/raw/Drone_CoD.csv'
    velocities = detector.prepare_drone_data(file_path)
    
    # Train GPR models
    detector.train_gpr_models(velocities)
    gpr_anomalies, gpr_predictions, gpr_uncertainties = detector.get_gpr_anomalies(velocities)
    
    # Train ensemble models
    velocity_data = np.column_stack((velocities['vx'], velocities['vy'], velocities['vz']))
    detector.train_ensemble_models(velocity_data)
    ensemble_anomalies, ensemble_scores = detector.get_ensemble_anomalies(velocity_data)
    
    # Visualize results
    detector.visualize_results(
        velocities,
        gpr_anomalies,
        gpr_predictions,
        gpr_uncertainties,
        ensemble_anomalies,
        ensemble_scores
    )

if __name__ == "__main__":
    main()
