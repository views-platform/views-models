import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class TensorHandler:
    """
    Handles conversions between 5D NumPy zstack, wide-format DataFrame, and long-format DataFrame.
    Ensures strict validation and provides visualization for sanity checks.
    """

    def __init__(self, data=None, source_format=None):
        """
        Initializes the TensorHandler with either a zstack, wide DataFrame, or long DataFrame.

        Args:
            data (np.ndarray | pd.DataFrame | None): Input data in one of the three formats or None for empty initialization.
            source_format (str | None): Either 'zstack', 'df_wide', or 'df_long'. If None, the object remains uninitialized.
        """


        self.source_format = source_format
        self.metadata = {}  # To store column names for reconstruction

        if source_format == "zstack":
            self.zstack = data
            self.df_wide = None
            self.df_long = None
            self.metadata["zstack_features"] = None # we can add more metadata here

        elif source_format == "df_wide" and data is not None:
            self.df_wide = data
            self.zstack = None
            self.df_long = None
            self.metadata["df_columns"] = list(data.columns)

        elif source_format == "df_long" and data is not None:
            self.df_long = data
            self.zstack = None
            self.df_wide = None
            self.metadata["df_long_columns"] = list(data.columns)

        else:
            self.zstack = None
            self.df_wide = None
            self.df_long = None

        # composition
        self.plotter = TensorPlotter()  # Composition: Delegate plotting
        self.generator = DataGenerator()  # Composition: Delegate data generation


    def plot_zstack(self, *args, **kwargs):
        """Delegates to TensorPlotter"""
        TensorPlotter.plot_zstack(self.zstack, *args, **kwargs)

    def plot_df_wide(self, *args, **kwargs):
        """Delegates to TensorPlotter"""
        TensorPlotter.plot_df_wide(self.df_wide, *args, **kwargs)

    def plot_df_long(self, *args, **kwargs):
        """Delegates to TensorPlotter"""
        TensorPlotter.plot_df_long(self.df_long, *args, **kwargs)


    def generate_synthetic_zstack(self, *args, **kwargs):
        """
        Generates a synthetic 5D NumPy zstack and stores it internally.
        Delegates to DataGenerator.
        """

        self.zstack, self.metadata['zstack_features'] = self.generator.generate_synthetic_zstack(*args, **kwargs)
        self.source_format = "zstack"

        # Reset other representations to avoid inconsistencies
        self.df_wide = None
        self.df_long = None

        return self.zstack, self.metadata



    def generate_synthetic_df_wide(self, *args, **kwargs):
        """
        ....
        """

        return None

    def to_wide_df(self, drop = False, asses = True):
        """Converts a stored zstack into a wide-format DataFrame."""

        if self.zstack is not None:
            self.df_wide = self.zstack_to_df_wide(self.zstack)

            if asses:
                # Check if the reconstructed wide-format DataFrame matches the original
                np.testing.assert_allclose(self.zstack, self.df_wide_to_zstack(self.df_wide), rtol=1e-5)

        elif self.df_long is not None:
            self.df_wide = self.df_long_to_df_wide(self.df_long)

            if asses:
                # Check if the reconstructed wide-format DataFrame matches the original
                np.testing.assert_allclose(self.df_long, self.df_wide_to_df_long(self.df_long), rtol=1e-5)

        else:
            raise ValueError("No zstack or long DataFrame available for conversion. Generate or load data first.")

        self.source_format = "df_wide"

        if drop: # we delete other data formats
            self.zstack = None
            self.df_long = None

        return self.df_wide
    

    def zstack_to_df_wide(self, zstack):
        """
        Converts a 5D NumPy array [z, row, col, feature, sample] into a DataFrame 
        where each feature column contains a list of samples.

        Returns:
            df_wide (pd.DataFrame): DataFrame with columns ['z', 'row', 'col', 'feature01', 'feature02', ...]
        """
        z_dim, row_dim, col_dim, num_features, num_samples = zstack.shape

        z, row, col = np.meshgrid(
            np.arange(z_dim), 
            np.arange(row_dim), 
            np.arange(col_dim), 
            indexing='ij'
        )

        data = {
            'z': z.flatten(),
            'row': row.flatten(),
            'col': col.flatten()
        }

        for f in range(num_features):
            feature_name = f'feature{f+1:02d}'
            data[feature_name] = zstack[:, :, :, f, :].reshape(-1, num_samples).tolist()

        return pd.DataFrame(data)


    def to_long_df(self, drop = False, asses = True):
        """Converts a stored zstack into a long-format DataFrame."""

        if self.zstack is not None:
              self.df_long = self.zstack_to_df_long(self.zstack)

              if asses:
                    # Check if the reconstructed long-format DataFrame matches the original
                    np.testing.assert_allclose(self.zstack, self.df_long_to_zstack(self.df_long), rtol=1e-5)
        
        elif self.df_wide is not None:
                self.df_long = self.df_wide_to_df_long(self.df_wide)

                if asses:
                    # Check if the reconstructed long-format DataFrame matches the original
                    np.testing.assert_allclose(self.zstack, self.df_long_to_zstack(self.df_long), rtol=1e-5)
        
        else:
            raise ValueError("No zstack or wide DataFrame available for conversion. Generate or load data first.")
        
        self.source_format = "df_long"

        if drop: # we delete other data formats
            self.zstack = None
            self.df_wide = None
        
        return self.df_long


    def zstack_to_df_long(self, zstack):
        """
        Converts a 5D NumPy array [z, row, col, feature, sample] into a long-format DataFrame
        where each row represents a single sample draw.

        Returns:
            df_long (pd.DataFrame): DataFrame with columns ['z', 'row', 'col', 'sample_id', 'feature01', 'feature02', ...]
        """
        z_dim, row_dim, col_dim, num_features, num_samples = zstack.shape

        z, row, col, sample = np.meshgrid(
            np.arange(z_dim), 
            np.arange(row_dim), 
            np.arange(col_dim), 
            np.arange(num_samples),
            indexing='ij'
        )

        data = {
            'z': z.flatten(),
            'row': row.flatten(),
            'col': col.flatten(),
            'sample_id': sample.flatten()
        }

        for f in range(num_features):
            feature_name = f'feature{f+1:02d}'
            data[feature_name] = zstack[:, :, :, f, :].reshape(-1)

        return pd.DataFrame(data)


    def to_zstack(self, drop = False, asses = True):
        """ Converts a stored wide-format or long-format DataFrame back into a 5D NumPy zstack."""

        if self.df_wide is not None:
            self.zstack = self.df_wide_to_zstack(self.df_wide)

            if asses:
                # Check if the reconstructed zstack matches the original
                np.testing.assert_allclose(self.df_wide, self.df_wide_to_zstack(self.df_wide), rtol=1e-5)

        elif self.df_long is not None:
            self.zstack = self.df_long_to_zstack(self.df_long)

            if asses:
                # Check if the reconstructed zstack matches the original
                np.testing.assert_allclose(self.df_wide, self.df_long_to_zstack(self.df_long), rtol=1e-5)

        else:
            raise ValueError("No wide or long DataFrame available for conversion. Generate or load data first.")
        
        self.source_format = "zstack"

        if drop: # we delete other data formats
            self.df_wide = None
            self.df_long = None

        
        return self.zstack


    def df_wide_to_zstack(self, df):
        """
        Converts a wide-format DataFrame back into a 5D NumPy array [z, row, col, feature, sample].

        Args:
            df (pd.DataFrame): Wide-format DataFrame where feature columns contain lists of sample values.

        Returns:
            np.ndarray: Reconstructed zstack with shape [z, row, col, feature, sample].
        """

        z_dim = df['z'].max() + 1
        row_dim = df['row'].max() + 1
        col_dim = df['col'].max() + 1
        feature_columns = [col for col in df.columns if col.startswith("feature")]
        num_features = len(feature_columns)

        sample_length_check = df[feature_columns[0]].values[0]
        num_samples = len(sample_length_check) if isinstance(sample_length_check, list) else 1

        zstack = np.zeros((z_dim, row_dim, col_dim, num_features, num_samples))

        z_indices = df['z'].values
        row_indices = df['row'].values
        col_indices = df['col'].values

        for i, feature in enumerate(feature_columns):
            feature_data = df[feature].apply(lambda x: x if isinstance(x, list) else [x])  # Ensure lists
            sample_values = np.stack(feature_data.values)  # Convert to 2D array
            zstack[z_indices, row_indices, col_indices, i, :] = sample_values

        return zstack


    def df_long_to_zstack(self, df_long):

        """
        Converts a long-format DataFrame (from zstack_to_df_long) back into a 5D NumPy array [z, row, col, feature, sample].

        Args:
            df_long (pd.DataFrame): Long-format DataFrame where each row represents a single sample.

        Returns:
            np.ndarray: Reconstructed zstack with shape [z, row, col, feature, sample].
        """
        
        # Extract shape details dynamically
        z_dim = df_long['z'].max() + 1
        row_dim = df_long['row'].max() + 1
        col_dim = df_long['col'].max() + 1
        num_samples = df_long['sample_id'].max() + 1
        feature_columns = [col for col in df_long.columns if col.startswith("feature")]
        num_features = len(feature_columns)

        # Initialize empty zstack
        zstack = np.zeros((z_dim, row_dim, col_dim, num_features, num_samples))

        # Assign values dynamically for each feature
        for i, feature in enumerate(feature_columns):
            for row in df_long.itertuples(index=False):
                zstack[int(row.z), int(row.row), int(row.col), i, int(row.sample_id)] = getattr(row, feature)

        return zstack


class DataGenerator:
    def __init__(self):
        pass  # No need for internal state

    def generate_synthetic_zstack(
        self,
        z_dim=10,
        row_dim=50,
        col_dim=50,
        num_samples=5,
        noise_level=0.05
    ):
        """
        Generates a synthetic 5D NumPy zstack with structured patterns and random noise,
        ensuring proper separation of deterministic and stochastic features.

        Dimensions:
        - (z_dim, row_dim, col_dim, num_features, num_samples)

        Args:
            z_dim (int): Number of time steps.
            row_dim (int): Number of rows (spatial resolution).
            col_dim (int): Number of columns (spatial resolution).
            num_deterministic_features (int): Number of deterministic features.
            num_stochastic_features (int): Number of stochastic features.
            num_samples (int): Number of Monte Carlo samples for stochastic features.
            noise_level (float): Standard deviation for stochastic feature noise.

        Returns:
            np.ndarray: Synthetic zstack with shape (z_dim, row_dim, col_dim, num_features, num_samples)
            dict: Feature metadata, specifying which features are deterministic vs. stochastic.
        """

        num_deterministic_features=3 # no need to have this as an argument
        num_stochastic_features=3 # no need to have this as an argument
        base_features = 5  # month_id, row, col, pg_id, c_id
        total_features = base_features + num_deterministic_features + num_stochastic_features

        # Ensure valid feature allocation
        assert total_features <= base_features + num_deterministic_features + num_stochastic_features, \
            f"Total features ({total_features}) exceed allocated space ({num_deterministic_features + num_stochastic_features + base_features})"

        # Initialize zstack
        zstack = np.zeros((z_dim, row_dim, col_dim, total_features, num_samples))

        # Feature order: month_id, row, col, pg_id, c_id, then deterministic and stochastic features
        feature_metadata = {
            "deterministic_features": ["month_id", "row", "col", "pg_id", "c_id"] + [f"det_feature_{i+1}" for i in range(num_deterministic_features)],
            "stochastic_features": [f"stoch_feature_{i+1}" for i in range(num_stochastic_features)]
        }

        # Assign deterministic features
        zstack[:, :, :, 0, :] = np.tile(np.arange(1, z_dim + 1)[:, None, None], (1, row_dim, col_dim))[:, :, :, None]  # month_id
        zstack[:, :, :, 1, :] = np.tile(np.arange(row_dim)[None, :, None], (z_dim, 1, col_dim))[:, :, :, None]  # row
        zstack[:, :, :, 2, :] = np.tile(np.arange(col_dim)[None, None, :], (z_dim, row_dim, 1))[:, :, :, None]  # col
        zstack[:, :, :, 3, :] = np.tile(np.arange(row_dim * col_dim).reshape(1, row_dim, col_dim), (z_dim, 1, 1))[:, :, :, None]  # pg_id
        #zstack[:, :, :, 4, :] = np.random.randint(1, 10, (1, row_dim, col_dim))[:, :, :, None]  # c_id
        zstack[:, :, :, 4, :] = np.tile(((np.arange(row_dim)[:, None] // 10) % 2 + (np.arange(col_dim)[None, :] // 10) % 2) % 2 + 1, (z_dim, 1, 1))[:, :, :, None]  # c_id


        # Structured deterministic feature patterns
        deterministic_patterns = [
            (slice(15, 35), slice(15, 35)),  # Middle square
            (slice(5, 15), slice(5, 15)),  # Top-left square
            (slice(5, 15), slice(col_dim-15, col_dim-5)),  # Top-right square
            (slice(row_dim-15, row_dim-5), slice(5, 15)),  # Bottom-left square
            (slice(row_dim-15, row_dim-5), slice(col_dim-15, col_dim-5)),  # Bottom-right square
        ]

        for f, (row_slice, col_slice) in zip(range(base_features, base_features + num_deterministic_features), deterministic_patterns):
            zstack[:, row_slice, col_slice, f, :] = 1.0  # Assign structured deterministic pattern

        # Structured stochastic feature patterns
        stochastic_patterns = [
            (slice(10, 40), slice(22, 28)),  # Middle vertical rectangle
            (slice(10, 40), slice(5, 15)),  # Left vertical rectangle
            (slice(10, 40), slice(col_dim-15, col_dim-5)),  # Right vertical rectangle
            (slice(22, 28), slice(10, 40)),  # Middle horizontal rectangle
            (slice(5, 15), slice(10, 40)),  # Top horizontal rectangle
        ]

        for f, (row_slice, col_slice) in zip(range(base_features + num_deterministic_features, total_features), stochastic_patterns):
            zstack[:, row_slice, col_slice, f, :] = np.random.normal(0.5, noise_level, (z_dim, 1, 1, num_samples))

        print(f"✅ Synthetic zstack created with shape: {zstack.shape}")
        return zstack, feature_metadata


class TensorPlotter:

    """ TensorPlotter is a class designed to handle the visualization of tensor data and its derived DataFrames.
    It provides methods to plot z-stacks, wide-format DataFrames, and long-format DataFrames, facilitating 
    the inspection and analysis of multi-dimensional data.

    Attributes:
        tensor_handler (TensorHandler): An instance of TensorHandler that provides the tensor data to be visualized.
        zstack (np.ndarray): A 5D numpy array representing the z-stack data with dimensions (z, height, width, features, samples).
        df_wide (pd.DataFrame): A wide-format DataFrame where each feature column contains lists of sample values.
        df_long (pd.DataFrame): A long-format DataFrame where each row represents a single sample with feature values.

    Methods:
        plot_zstack(num_z_slices=None, features=None, samples=None):
            Plots the stored z-stack showing different features, time steps, and sample draws.

        plot_df_wide(num_z_slices=5, feature_columns=None, num_samples=5):
            Each feature column contains a list of samples, so individual samples are plotted.
            
        plot_df_long(num_z_slices=5, num_samples=5, features=None): """
    

    @staticmethod
    def plot_zstack(zstack, num_z_slices=None, features=None, samples=None):
        """
        Plots the stored zstack showing different features, time steps, and sample draws.
    
        Args:
            num_z_slices (int, optional): Number of time steps to visualize. Default is 5.
            features (list, optional): List of feature indices to plot. Defaults to all features.
            samples (list, optional): List of sample indices to plot. Defaults to all samples.
        """
        if zstack is None:
            raise ValueError("No zstack available for plotting. Generate or load a zstack first.")
    
        num_z_slices = zstack.shape[0] if num_z_slices is None else num_z_slices
        features = np.arange(0, zstack.shape[3], 1) if features is None else features
        samples = np.arange(0, zstack.shape[4], 1) if samples is None else samples
    
        num_rows = len(features) * len(samples)
        
        # Ensure at least a 2D array to avoid indexing issues
        fig, axes = plt.subplots(num_rows, num_z_slices, figsize=(15, 3 * num_rows))
        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)  # Convert 1D array to 2D
    
        row_idx = 0  # Track row index in subplot
        for z in range(num_z_slices):  
            row_idx = 0  # Reset row index for each z iteration
            for f in features:  
                for s in samples:  
                    ax = axes[row_idx, z]  

                    # or should we have something more general here?
                    vmin = np.min(zstack[:, :, :, f, :])
                    vmax = np.max(zstack[:, :, :, f, :])

                    ax.imshow(zstack[z, :, :, f, s], vmin=vmin, vmax=vmax, cmap='viridis')
    
                    if row_idx == 0:
                        ax.set_title(f"z = {z}")
    
                    if z == 0:
                        ax.set_ylabel(f"Feature {f+1}, Sample {s+1}")
    
                    ax.set_xticks([])
                    ax.set_yticks([])
    
                    row_idx += 1  # Move to next subplot row
    
        plt.tight_layout()


    @staticmethod
    def plot_df_wide(df_wide, num_z_slices=5, feature_columns=None, num_samples=5):
        """
        Plots a sanity check for the wide-format DataFrame (zstack_to_df output).
        Each feature column contains a list of samples, so we plot individual samples.

        Args:
            df (pd.DataFrame): Wide-format DataFrame where feature columns contain lists of sample values.
            num_z_slices (int, optional): Number of z slices to visualize. Default is 5.
            feature_columns (list, optional): List of feature columns to plot. Defaults to all available features.
            num_samples (int, optional): Number of samples per feature to plot. Default is 5.
        """

        if df_wide is None:
            raise ValueError("No wide-format DataFrame available for plotting. Convert a zstack to wide format first.")

        if feature_columns is None:
            non_feature_columns = ['row', 'col', 'z']
            feature_columns = [col for col in df_wide.columns if col not in non_feature_columns]

        num_features = len(feature_columns)
        num_rows = num_features * num_samples  # Each feature-sample pair gets a row

        # Get the width and height of the plot
        width = df_wide['col'].max() + 1
        height = df_wide['row'].max() + 1

        # Create figure with a grid layout: (features * samples) x (z slices)
        fig, axes = plt.subplots(num_rows, num_z_slices, figsize=(15, 3 * num_rows))

        # Ensure axes is always 2D
        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for z in range(num_z_slices):
            df_slice = df_wide[df_wide['z'] == z]

            row_idx = 0  # Reset row index for each z slice

            for f, feature in enumerate(feature_columns):
                feature_matrix = np.stack(df_slice[feature].values)  # Convert lists to array

                for s in range(num_samples):
                    if num_z_slices == 1:
                        ax = axes[row_idx]  # Select subplot row for single z slice
                    else:
                        ax = axes[row_idx, z]  # Select subplot row for multiple z slices
                    feature_sample = feature_matrix[:, s]  # Extract s-th sample

                    ax.scatter(df_slice['col'], height - df_slice['row'], s=5, c=feature_sample, cmap='viridis', vmin=0, vmax=1)

                    # Titles for first row only
                    if row_idx == 0:
                        ax.set_title(f"z = {z}")

                    # Row labels for first column only
                    if z == 0:
                        ax.set_ylabel(f"Feature {f+1}, Sample {s+1}")

                    ax.set_xlim(0, width)
                    ax.set_ylim(0, height)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    row_idx += 1  # Move to next subplot row

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_df_long(df_long, num_z_slices=5, num_samples=5, features=None):

        """
        Plots a sanity check for the long-format DataFrame (zstack_to_df_long output).
        Each row corresponds to a (feature, sample) pair, and each column corresponds to a z-slice.

        Args:
            df_long (pd.DataFrame): Long-format DataFrame where each row represents a single sample.
            num_z_slices (int, optional): Number of z slices to visualize. Default is 5.
            num_samples (int, optional): Number of unique sample_ids to randomly select. If None, all samples are used.
            features (list, optional): List of feature column names to plot. If None, all feature columns are used.
        """

        if df_long is None:
            raise ValueError("No long-format DataFrame available for plotting. Convert a zstack to long format first.")

        # Select a random subset of samples if num_samples is provided
        unique_samples = df_long['sample_id'].unique()
        
        if num_samples is None:
            selected_samples = unique_samples  # Use all available samples
        
        else:
            num_samples = min(num_samples, len(unique_samples))  # Ensure it doesn't exceed available samples
            selected_samples = np.random.choice(unique_samples, num_samples, replace=False)

        # Extract available feature columns dynamically
        available_features = [col for col in df_long.columns if col.startswith("feature")]

        # Use only specified features, or default to all available ones
        if features is None:
            features = available_features
        else:
            # Ensure selected features exist in the DataFrame
            features = [f for f in features if f in available_features]

        num_features = len(features)
        num_rows = num_features * num_samples  # Each row is a (feature, sample) pair
        num_cols = num_z_slices  # Each column is a z-slice

        # Create figure with a grid layout: (num_features * num_samples) rows, num_z_slices columns
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))

        # Ensure axes is always a 2D array, even if num_rows or num_cols is 1
        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        if num_cols == 1:
            axes = np.expand_dims(axes, axis=1)

        # FEATURE → SAMPLE → Z loop order
        for f_idx, feature in enumerate(features):  
            for sample_idx, sample in enumerate(selected_samples):  
                row_idx = f_idx * num_samples + sample_idx  # Compute row index for (feature, sample) pair

                df_sample = df_long[df_long['sample_id'] == sample]  # Filter for the current sample

                for z in range(num_z_slices):  
                    ax = axes[row_idx, z]  # Select subplot

                    df_slice = df_sample[df_sample['z'] == z]

                    # Scatter plot for visualization
                    scatter = ax.scatter(df_slice['col'], 50 - df_slice['row'], s=5, c=df_slice[feature], cmap='viridis', vmin=0, vmax=1)

                    # Titles for first row only
                    if row_idx == 0:
                        ax.set_title(f"z = {z}")

                    # Row labels for first column only
                    if z == 0:
                        ax.set_ylabel(f"{feature}, Sample {sample}")

                    # Formatting
                    ax.set_xlim(0, 50)
                    ax.set_ylim(0, 50)
                    ax.set_xticks([])
                    ax.set_yticks([])

        plt.tight_layout()
        plt.show()