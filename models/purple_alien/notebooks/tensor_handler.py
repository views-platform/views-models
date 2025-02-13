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
        TensorPlotter.plot_zstack(self.zstack, self.metadata['zstack_features'], *args, **kwargs)

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

    def to_wide_df(self, drop = False, assess = True):
        """Converts a stored zstack into a wide-format DataFrame."""

        if self.zstack is not None:
            self.df_wide = self.zstack_to_df_wide(self.zstack)

#            if assess:
#                # Check if the reconstructed wide-format DataFrame matches the original
#                np.testing.assert_allclose(self.zstack, self.df_wide_to_zstack(self.df_wide), rtol=1e-5)
#
#        elif self.df_long is not None:
#            self.df_wide = self.df_long_to_df_wide(self.df_long)
#
#            if assess:
#                # Check if the reconstructed wide-format DataFrame matches the original
#                np.testing.assert_allclose(self.df_long, self.df_wide_to_df_long(self.df_long), rtol=1e-5)
#
#        else:
#            raise ValueError("No zstack or long DataFrame available for conversion. Generate or load data first.")
#
        self.source_format = "df_wide"

        if drop: # we delete other data formats
            self.zstack = None
            self.df_long = None

        return self.df_wide


    def zstack_to_df_wide(self, zstack):
        """
        Converts a 5D NumPy array [z, row, col, feature, sample] into a DataFrame 
        where deterministic features are stored as scalars and stochastic features as lists.

        Returns:
            df_wide (pd.DataFrame): DataFrame with columns ['month_id', 'row', 'col', 'pg_id', 'c_id', ..., 'stoch_feature01', ...]
        """
        z_dim, row_dim, col_dim, num_features, num_samples = zstack.shape

        # Extract feature names from metadata
        feature_metadata = self.metadata['zstack_features']
        deterministic_features = feature_metadata['deterministic_features']
        stochastic_features = feature_metadata['stochastic_features']

        # Generate base spatial-temporal grid
        z, row, col = np.meshgrid(
            np.arange(z_dim), 
            np.arange(row_dim), 
            np.arange(col_dim), 
            indexing='ij'
        )

        data = {
            'month_id': z.flatten(),
            'row': row.flatten(),
            'col': col.flatten()
        }

        # Extract deterministic features (stored as scalars)
        for f_idx, feature_name in enumerate(deterministic_features):
            data[feature_name] = zstack[:, :, :, f_idx, 0].flatten()

        # Extract stochastic features (stored as lists)
        for f_idx, feature_name in enumerate(stochastic_features, start=len(deterministic_features)):
            data[feature_name] = zstack[:, :, :, f_idx, :].reshape(-1, num_samples).tolist()

        return pd.DataFrame(data)

#

    def to_long_df(self, drop = False, asses = True):
        """Converts a stored zstack into a long-format DataFrame."""

        if self.zstack is not None:
              self.df_long = self.zstack_to_df_long(self.zstack)

#              if asses:
#                    # Check if the reconstructed long-format DataFrame matches the original
#                    np.testing.assert_allclose(self.zstack, self.df_long_to_zstack(self.df_long), rtol=1e-5)
#        
#        elif self.df_wide is not None:
#                self.df_long = self.df_wide_to_df_long(self.df_wide)
#
#                if asses:
#                    # Check if the reconstructed long-format DataFrame matches the original
#                    np.testing.assert_allclose(self.zstack, self.df_long_to_zstack(self.df_long), rtol=1e-5)
#        
#        else:
#            raise ValueError("No zstack or wide DataFrame available for conversion. Generate or load data first.")
        
        self.source_format = "df_long"

        if drop: # we delete other data formats
            self.zstack = None
            self.df_wide = None
        
        return self.df_long


    def zstack_to_df_long(self, zstack):
        """
        Converts a 5D NumPy array [z, row, col, feature, sample] into a long-format DataFrame
        where:
          - Deterministic features are stored once per (z, row, col).
          - Stochastic features are expanded per sample_id.

        Args:
            zstack (np.ndarray): 5D tensor [z, row, col, feature, sample]
            metadata (dict): Metadata containing feature names.

        Returns:
            pd.DataFrame: Long-format DataFrame with structure:
                          ['month_id', 'row', 'col', 'pg_id', 'c_id', 'sample_id', <features>]
        """

        # Extract feature names from metadata
        feature_metadata = self.metadata['zstack_features']
        deterministic_features = feature_metadata['deterministic_features']
        stochastic_features = feature_metadata['stochastic_features']

        z_dim, row_dim, col_dim, num_features, num_samples = zstack.shape

        # Create (z, row, col) base meshgrid
        z, row, col = np.meshgrid(
            np.arange(z_dim), 
            np.arange(row_dim), 
            np.arange(col_dim), 
            indexing="ij"
        )

        base_data = {
            "month_id": (z + 1).flatten(),  # Ensure first month is 1, not 0
            "row": row.flatten(),
            "col": col.flatten(),
        }

        # Extract deterministic features (stored once per (z, row, col))
        for f_idx, feature in enumerate(deterministic_features):
            base_data[feature] = zstack[:, :, :, f_idx, 0].flatten()  # Only take 1 sample

        df_deterministic = pd.DataFrame(base_data)

        # Expand for stochastic features (adds sample_id)
        z_expanded, row_expanded, col_expanded, sample_expanded = np.meshgrid(
            np.arange(z_dim),
            np.arange(row_dim),
            np.arange(col_dim),
            np.arange(num_samples),
            indexing="ij"
        )

        stochastic_data = {
            "month_id": (z_expanded + 1).flatten(),
            "row": row_expanded.flatten(),
            "col": col_expanded.flatten(),
            "sample_id": sample_expanded.flatten(),
        }

        for f_idx, feature in enumerate(stochastic_features):
            feature_idx = len(deterministic_features) + f_idx  # Get correct feature index
            stochastic_data[feature] = zstack[:, :, :, feature_idx, :].flatten()

        df_stochastic = pd.DataFrame(stochastic_data)

        # Merge deterministic & stochastic into final `df_long`
        df_long = df_stochastic.merge(df_deterministic, on=["month_id", "row", "col"], how="left")

        return df_long




















    def df_wide_to_zstack(self, df):
        """
        Converts a wide-format DataFrame back into a 5D NumPy array [z, row, col, feature, sample].

        Args:
            df (pd.DataFrame): Wide-format DataFrame where feature columns contain lists of sample values.

        Returns:
            np.ndarray: Reconstructed zstack with shape [z, row, col, feature, sample].
        """

        z_dim = df['month_id'].max() + 1
        row_dim = df['row'].max() + 1
        col_dim = df['col'].max() + 1
        feature_columns = [col for col in df.columns if col.startswith("feature")]
        num_features = len(feature_columns)

        sample_length_check = df[feature_columns[0]].values[0]
        num_samples = len(sample_length_check) if isinstance(sample_length_check, list) else 1

        zstack = np.zeros((z_dim, row_dim, col_dim, num_features, num_samples))

        z_indices = df['month_id'].values
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
        noise_level=0.5
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
        num_base_features = 5  # month_id, row, col, pg_id, c_id
        num_total_features = num_base_features + num_deterministic_features + num_stochastic_features

        # Ensure valid feature allocation
        assert num_total_features <= num_base_features + num_deterministic_features + num_stochastic_features, \
            f"Total features ({num_total_features}) exceed allocated space ({num_deterministic_features + num_stochastic_features + num_base_features})"

        # Initialize zstack
        zstack = np.zeros((z_dim, row_dim, col_dim, num_total_features, num_samples))

        # Feature order: month_id, row, col, pg_id, c_id, then deterministic and stochastic features
        feature_metadata = {
            "deterministic_features": ["month_id", "row", "col", "pg_id", "c_id"] + [f"deterministic_feature_{i+1}" for i in range(num_deterministic_features)],
            "stochastic_features": [f"stochastic_feature_{i+1}" for i in range(num_stochastic_features)]
        }

        # Assign deterministic features
        zstack[:, :, :, 0, :] = np.tile(np.arange(1, z_dim + 1)[:, None, None], (1, row_dim, col_dim))[:, :, :, None]  # month_id
        zstack[:, :, :, 1, :] = np.tile(np.arange(row_dim)[None, :, None], (z_dim, 1, col_dim))[:, :, :, None]  # row
        zstack[:, :, :, 2, :] = np.tile(np.arange(col_dim)[None, None, :], (z_dim, row_dim, 1))[:, :, :, None]  # col
        zstack[:, :, :, 3, :] = np.tile(np.arange(row_dim * col_dim).reshape(1, row_dim, col_dim), (z_dim, 1, 1))[:, :, :, None]  # pg_id
        zstack[:, :, :, 4, :] = np.tile(((np.arange(row_dim)[:, None] // 10) % 2 + (np.arange(col_dim)[None, :] // 10) % 2) % 2 + 1, (z_dim, 1, 1))[:, :, :, None]  # c_id


        # Structured deterministic feature patterns
        deterministic_patterns = [
            (slice(15, 35), slice(15, 35)),  # Middle square
            (slice(5, 15), slice(5, 15)),  # Top-left square
            (slice(5, 15), slice(col_dim-15, col_dim-5)),  # Top-right square
            (slice(row_dim-15, row_dim-5), slice(5, 15)),  # Bottom-left square
            (slice(row_dim-15, row_dim-5), slice(col_dim-15, col_dim-5)),  # Bottom-right square
        ]

        for f, (row_slice, col_slice) in zip(range(num_base_features, num_base_features + num_deterministic_features), deterministic_patterns):
            zstack[:, row_slice, col_slice, f, :] = 1.0  # Assign structured deterministic pattern


        # Structured stochastic feature patterns
        stochastic_patterns = [
            (slice(10, 40), slice(22, 28)),  # Middle vertical rectangle
            (slice(10, 40), slice(5, 15)),  # Left vertical rectangle
            (slice(10, 40), slice(col_dim-15, col_dim-5)),  # Right vertical rectangle
            (slice(22, 28), slice(10, 40)),  # Middle horizontal rectangle
            (slice(5, 15), slice(10, 40)),  # Top horizontal rectangle
        ]

        for f, (row_slice, col_slice) in zip(range(num_base_features + num_deterministic_features, num_total_features), stochastic_patterns):
            zstack[:, row_slice, col_slice, f, :] = np.random.normal(0.5, noise_level, (z_dim, 1, 1, num_samples))

       # add the month_id values to all deterministic and stochastic features (but not base) to give them different values for each month
        for f in range(num_base_features, num_total_features):
            zstack[:, :, :, f, :] += zstack[:, :, :, 0, :] * 0.1 


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
    def plot_zstack(zstack, metadata, num_z_slices=None, features=None, samples=None):
        """
        Plots the stored zstack showing different features, time steps, and sample draws.

        Args:
            zstack (np.ndarray): The 5D tensor to plot.
            metadata (dict): Metadata containing feature names and types.
            num_z_slices (int, optional): Number of time steps to visualize. Default is 5.
            features (list, optional): List of feature indices to plot. Defaults to all features.
            samples (list, optional): List of sample indices to plot. Defaults to all samples.
        """

        if zstack is None:
            raise ValueError("No zstack available for plotting. Generate or load a zstack first.")

        # Extract metadata
        deterministic_features = metadata["deterministic_features"]
        stochastic_features = metadata["stochastic_features"]

        num_z_slices = min(zstack.shape[0], num_z_slices if num_z_slices is not None else 5)
        features = np.arange(zstack.shape[3]) if features is None else features
        samples = np.arange(zstack.shape[4]) if samples is None else samples

        # ✅ Calculate number of rows correctly
        num_deterministic = len(deterministic_features)
        num_stochastic = len(stochastic_features) * len(samples)
        num_rows = num_deterministic + num_stochastic  # ✅ Avoids `IndexError`

        # ✅ Create figure
        fig, axes = plt.subplots(num_rows, num_z_slices, figsize=(15, 3 * num_rows))
        axes = np.atleast_2d(axes)  # ✅ Ensures 2D structure

        row_idx = 0  # ✅ Tracks subplot row

        for z in range(num_z_slices):  
            row_idx = 0  # ✅ Reset row index for each z iteration

            # 🚀 **Plot Deterministic Features**
            for f_idx, feature_name in enumerate(deterministic_features):
                ax = axes[row_idx, z] if num_z_slices > 1 else axes[row_idx]
                vmin, vmax = np.min(zstack[:, :, :, f_idx, 0]), np.max(zstack[:, :, :, f_idx, 0])
                ax.imshow(zstack[z, :, :, f_idx, 0], vmin=vmin, vmax=vmax, cmap='viridis')

                if row_idx == 0:
                    ax.set_title(f"z = {z}")

                if z == 0:
                    ax.set_ylabel(f"{feature_name} (Det)")

                ax.set_xticks([])
                ax.set_yticks([])
                row_idx += 1  # ✅ Move to next subplot row

            # 🚀 **Plot Stochastic Features**
            for f_idx, feature_name in enumerate(stochastic_features):
                for s_idx, s in enumerate(samples):
                    ax = axes[row_idx, z] if num_z_slices > 1 else axes[row_idx]
                    vmin, vmax = np.min(zstack[:, :, :, num_deterministic + f_idx, :]), np.max(zstack[:, :, :, num_deterministic + f_idx, :])
                    ax.imshow(zstack[z, :, :, num_deterministic + f_idx, s], vmin=vmin, vmax=vmax, cmap='viridis')

                    if row_idx == 0:
                        ax.set_title(f"z = {z}")

                    if z == 0:
                        ax.set_ylabel(f"{feature_name} (Stoch, S={s})")

                    ax.set_xticks([])
                    ax.set_yticks([])
                    row_idx += 1  # ✅ Move to next subplot row

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_df_wide(df_wide, num_z_slices=5, feature_columns=None, num_samples=5):
        """
        Plots a sanity check for the wide-format DataFrame (`df_wide`).
        - Deterministic features are plotted as single values.
        - Stochastic features are plotted as multiple samples.

        Args:
            df_wide (pd.DataFrame): Wide-format DataFrame where feature columns contain lists of sample values.
            num_z_slices (int, optional): Number of z slices (months) to visualize. Default is 5.
            feature_columns (list, optional): List of feature columns to plot. Defaults to all available features.
            num_samples (int, optional): Number of samples per stochastic feature to plot. Default is 5.
        """

        if df_wide is None:
            raise ValueError("No wide-format DataFrame available for plotting. Convert a zstack to wide format first.")

        if "month_id" not in df_wide.columns:
            raise KeyError("Missing 'month_id' column in df_wide.")

        # 🚀 **Identify Deterministic vs Stochastic Features**
        basis_feature_columns = ["month_id", "row", "col", "pg_id", "c_id"]
        all_features = [col for col in df_wide.columns if col not in basis_feature_columns]

        deterministic_features = [col for col in all_features if not isinstance(df_wide[col].iloc[0], list)]
        stochastic_features = [col for col in all_features if isinstance(df_wide[col].iloc[0], list)]

        if feature_columns is None:
            feature_columns = deterministic_features + stochastic_features

        # 🚀 **Ensure num_z_slices doesn't exceed available months**
        unique_months = sorted(df_wide["month_id"].unique())  # Ensure sorted order
        num_z_slices = min(num_z_slices, len(unique_months))

        # 🚀 **Determine Grid Size**
        width = df_wide["col"].max() + 1
        height = df_wide["row"].max() + 1
        num_rows = len(basis_feature_columns) + len(deterministic_features) + len(stochastic_features) * num_samples  # ✅ Ensure correct allocation

        # 🚀 **Create Figure & Subplots**
        fig, axes = plt.subplots(max(1, num_rows), max(1, num_z_slices), figsize=(15, 3 * max(1, num_rows)))
        axes = np.atleast_2d(axes)  # Ensure 2D array indexing works

        for z_idx, month in enumerate(unique_months[:num_z_slices]):  
            row_idx = 0  # ✅ Reset row index for each z_slice

            df_slice = df_wide[df_wide["month_id"] == month]  # ✅ Filtering correctly

            for feature in basis_feature_columns:
                if row_idx >= num_rows:
                    raise IndexError(f"row_idx {row_idx} exceeds allocated subplots ({num_rows})")
                
                ax = axes[row_idx, z_idx] if num_z_slices > 1 else axes[row_idx]
                feature_values = df_slice[feature].values

                vmin = df_wide[feature].min()
                vmax = df_wide[feature].max()

                ax.scatter(df_slice["col"], height - df_slice["row"], s=5, c=feature_values, cmap="viridis", vmin=vmin, vmax=vmax)

                ax.set_ylabel(feature)

                if row_idx == 0:
                    ax.set_title(f"Month {month}")
                    
                ax.set_xlim(0, width)
                ax.set_ylim(0, height)
                ax.set_xticks([])
                ax.set_yticks([])

                row_idx += 1  # ✅ Correct Increment


            # 🚀 **Plot Deterministic Features**
            for feature in deterministic_features:
                if row_idx >= num_rows:
                    raise IndexError(f"row_idx {row_idx} exceeds allocated subplots ({num_rows})")  # 🚀 Added Safety Check

                ax = axes[row_idx, z_idx] if num_z_slices > 1 else axes[row_idx]
                feature_values = df_slice[feature].values

                vmin = df_wide[feature].min()
                vmax = df_wide[feature].max()
                ax.scatter(df_slice["col"], height - df_slice["row"], s=5, c=feature_values, cmap="viridis", vmin=vmin, vmax=vmax)

                ax.set_ylabel(f"{feature} (Det)")
                if row_idx == 0:
                    ax.set_title(f"Month {month}")

                ax.set_xlim(0, width)
                ax.set_ylim(0, height)
                ax.set_xticks([])
                ax.set_yticks([])

                row_idx += 1

            # 🚀 **Plot Stochastic Features**
            for feature in stochastic_features:
                feature_matrix = np.stack(df_slice[feature].values)  

                for s in range(min(num_samples, feature_matrix.shape[1])):  
                    if row_idx >= num_rows:
                        raise IndexError(f"row_idx {row_idx} exceeds allocated subplots ({num_rows})")  # 🚀 Safety Check

                    ax = axes[row_idx, z_idx] if num_z_slices > 1 else axes[row_idx]
                    feature_sample = feature_matrix[:, s]

                    vmin = np.array([lst for lst in df_wide["stochastic_feature_1"]]).min()
                    vmax = np.array([lst for lst in df_wide["stochastic_feature_1"]]).max()
                    ax.scatter(df_slice["col"], height - df_slice["row"], s=5, c=feature_sample, cmap="viridis", vmin=vmin, vmax=vmax)

                    ax.set_ylabel(f"{feature} (Stoch, S={s})")
                    if row_idx == 0:
                        ax.set_title(f"Month {month}")

                    ax.set_xlim(0, width)
                    ax.set_ylim(0, height)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    row_idx += 1  # 🚀 Corrected Increment

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_df_long(df_long, num_z_slices=5, num_samples=5, features=None):
        """
        Plots a sanity check for the long-format DataFrame (`df_long`).
        - Deterministic features are plotted as single values per `(month_id, row, col)`.
        - Stochastic features are plotted per `(month_id, row, col, sample_id)`.

        Args:
            df_long (pd.DataFrame): Long-format DataFrame.
            num_z_slices (int, optional): Number of `month_id` slices to visualize. Default is 5.
            num_samples (int, optional): Number of unique `sample_id`s to randomly select.
            features (list, optional): List of feature column names to plot.
        """

        if df_long is None:
            raise ValueError("No long-format DataFrame available for plotting.")

        if "month_id" not in df_long.columns:
            raise KeyError("Missing 'month_id' column in df_long.")

        # 🚀 **Identify Deterministic vs Stochastic Features**
        feature_classification = TensorPlotter.classify_features(df_long)

        basis_feature_columns = feature_classification['basis']
        deterministic_features = feature_classification['deterministic']
        stochastic_features = feature_classification['stochastic']
        

        # **Feature Selection Handling**
        if features is not None:
            basis_feature_columns = [f for f in basis_feature_columns if f in features]
            deterministic_features = [f for f in deterministic_features if f in features]
            stochastic_features = [f for f in stochastic_features if f in features]

        # 🚀 **Ensure num_z_slices doesn’t exceed available months**
        unique_months = sorted(df_long["month_id"].unique())  
        num_z_slices = min(num_z_slices, len(unique_months))

        # 🚀 **Determine Grid Size**
        width, height = df_long["col"].max() + 1, df_long["row"].max() + 1
        num_rows = len(basis_feature_columns) + len(deterministic_features) + len(stochastic_features) * num_samples

        # 🚀 **Create Figure & Subplots**
        fig, axes = plt.subplots(max(1, num_rows), max(1, num_z_slices), figsize=(15, 3 * max(1, num_rows)))
        axes = np.atleast_2d(axes)  # Ensure always 2D

    
        for z_idx, month in enumerate(unique_months[:num_z_slices]):  
            df_slice = df_long[df_long["month_id"] == month]

            row_idx = 0  # ✅ Reset row index once before looping


            # 🚀 **Plot Basis Features First**
            for feature in basis_feature_columns:
                if row_idx >= num_rows:
                    raise IndexError(f"row_idx {row_idx} exceeds allocated subplots ({num_rows})")  # ✅ Safety check
                
                vmin = df_long[feature].min()
                vmax = df_long[feature].max()

                ax = axes[row_idx, z_idx] if axes.ndim == 2 else axes[row_idx]
                ax.scatter(df_slice["col"], height - df_slice["row"], s=5, c=df_slice[feature], cmap="viridis", vmin = vmin, vmax = vmax)
                ax.set_ylabel(feature)

                if row_idx == 0:
                    ax.set_title(f"Month {month}")

                row_idx += 1  # ✅ Increment correctly

            # 🚀 **Plot Deterministic Features**
            for feature in deterministic_features:
                if row_idx >= num_rows:
                    raise IndexError(f"row_idx {row_idx} exceeds allocated subplots ({num_rows})")
                
                vmin = df_long[feature].min()
                vmax = df_long[feature].max()

                ax = axes[row_idx, z_idx] if axes.ndim == 2 else axes[row_idx]
                ax.scatter(df_slice["col"], height - df_slice["row"], s=5, c=df_slice[feature], cmap="viridis",
                           vmin=vmin, vmax=vmax)

                ax.set_ylabel(f"{feature} (Det)")
                if row_idx == 0:
                    ax.set_title(f"Month {month}")

                row_idx += 1  # ✅ Correct Increment

            # 🚀 **Plot Stochastic Features Next (Per Sample)**
            if "sample_id" in df_long.columns:
                unique_samples = df_long["sample_id"].unique()
                selected_samples = np.random.choice(unique_samples, min(num_samples, len(unique_samples)), replace=False)

                for feature in stochastic_features:
                    for s in selected_samples:
                        df_sample = df_slice[df_slice["sample_id"] == s]

                        if row_idx >= num_rows:
                            raise IndexError(f"row_idx {row_idx} exceeds allocated subplots ({num_rows})")
                        
                        vmin = df_long[feature].min()
                        vmax = df_long[feature].max()

                        ax = axes[row_idx, z_idx] if axes.ndim == 2 else axes[row_idx]
                        ax.scatter(df_sample["col"], height - df_sample["row"], s=5, c=df_sample[feature], cmap="viridis",
                                   vmin=vmin, vmax=vmax)

                        ax.set_ylabel(f"{feature} (S={s})")
                        if row_idx == 0:
                            ax.set_title(f"Month {month}")

                        row_idx += 1  # ✅ Corrected Increment

        plt.tight_layout()
        plt.show()


    @staticmethod
    def classify_features(df_long):
        """
        Identifies stochastic and deterministic features in a long-format DataFrame.

        A feature is:
        - Deterministic if its variance across 'sample_id' is 0 for all (z, row, col).
        - Stochastic if its variance across 'sample_id' is non-zero for any (z, row, col).

        Args:
            df_long (pd.DataFrame): A long-format DataFrame with columns ['z', 'row', 'col', 'sample_id', feature1, feature2, ...]

        Returns:
            dict: {'stochastic': [list of stochastic features], 'deterministic': [list of deterministic features]}
        """

        # Exclude non-feature columns
        basis_feature_columns = ['month_id', 'row', 'col', 'pg_id', 'c_id', 'sample_id']
        feature_columns = [col for col in df_long.columns if col not in basis_feature_columns]

        # Initialize lists for classification
        stochastic_features = []
        deterministic_features = []

        # Compute variance across samples for each feature
        for feature in feature_columns:
            variance = df_long.groupby(['month_id', 'row', 'col'])[feature].var().fillna(0)

            if (variance == 0).all():
                deterministic_features.append(feature)
            else:
                stochastic_features.append(feature)

        return {
            'basis': basis_feature_columns,
            'stochastic': stochastic_features,
            'deterministic': deterministic_features
        }



