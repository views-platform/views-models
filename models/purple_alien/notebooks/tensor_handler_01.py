import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging


class TensorHandler:
    """
    Handles conversions between 5D NumPy zstack or wide-format DataFrame.
    Ensures strict validation and provides visualization for sanity checks.
    """

    def __init__(self, data=None):
        """
        Initializes the ZStackHandler with either a 5D zstack or a wide-format DataFrame.

        Args:
            data (np.ndarray | pd.DataFrame | None): Input data in either zstack or DataFrame format.
        """

        self.logger = logging.getLogger(__name__)
        self.generator = DataGenerator()

        self.zstack = None
        self.df_wide = None
        self.grid_shape = None  # (row, col)
        self.metadata = {"zstack_features": None, "df_columns": None}

        if data is None:
            self.logger.info("!! No data provided !!.")
            return

        if isinstance(data, np.ndarray):
            if len(data.shape) != 5:
                raise ValueError(f"Expected a 5D zstack (time, row, col, features, samples), but got shape {data.shape}.")
            self.zstack = data
            self.grid_shape = (data.shape[1], data.shape[2])  # Extract (row, col)
            self.logger.info(f"✅ Initialized with zstack of shape {data.shape}.")

        elif isinstance(data, pd.DataFrame):

            if type(data.index) == pd.MultiIndex:
                data = data.reset_index()

            required_columns = {"priogrid_gid", "row", "col", "month_id", "c_id"}
            missing_columns = required_columns - set(data.columns)

            if missing_columns:
                raise ValueError(f"❌ Missing columns in DataFrame: {missing_columns}")

            self.df_wide = data
            self.metadata["df_columns"] = list(data.columns)

            if "row" in data.columns and "col" in data.columns:
                self.grid_shape = (data["row"].max() + 1, data["col"].max() + 1)

            logger.info(f"✅ Initialized with DataFrame of shape {data.shape}.")

        else:
            raise TypeError("❌ Expected data to be a NumPy array (zstack) or Pandas DataFrame (df_wide).")
        


    def set_grid_shape(self):
        """
        Sets the grid shape `(row_dim, col_dim)` based on the stored `zstack`.
        """

        if self.zstack is None:
            self.logger.error("❌ Cannot determine grid shape: `zstack` is not loaded.")
            raise ValueError("Cannot determine grid shape: `zstack` is not loaded.")

        _, row_dim, col_dim, _, _ = self.zstack.shape
        self.grid_shape = (row_dim, col_dim)  # ✅ Store it as an attribute

        self.logger.info(f"✅ Grid shape set to {self.grid_shape}.")
        return self.grid_shape


    def plot_zstack(self, *args, **kwargs):
        """Delegates to TensorPlotter"""
        TensorPlotter.plot_zstack(self.zstack, self.metadata['zstack_features'], *args, **kwargs)

    def plot_df_wide(self, *args, **kwargs):
        """Delegates to TensorPlotter"""
        TensorPlotter.plot_df_wide(self.df_wide, *args, **kwargs)


    def generate_synthetic_zstack(self, *args, **kwargs):
        """
        Generates a synthetic 5D NumPy zstack and stores it internally.
        Delegates to DataGenerator.
        """

        self.zstack, self.metadata['zstack_features'] = self.generator.generate_synthetic_zstack(*args, **kwargs)
        self.source_format = "zstack"

        # set the grid shape
        self.set_grid_shape()

        # Reset other representations to avoid inconsistencies
        self.df_wide = None

        return self.zstack, self.metadata
    

    def infer_metadata_from_wide_df(self):
        """
        Infers metadata from a wide-format DataFrame.
        - Deterministic features: Known identifiers (e.g., 'month_id', 'row', etc.).
        - Stochastic features: Columns where entries are lists.
        """
        if not hasattr(self, "df_wide") or self.df_wide is None:
            self.logger.error("❌ Cannot infer metadata: `df_wide` is not initialized.")
            raise ValueError("Cannot infer metadata: `df_wide` is not available.")

        self.logger.info("ℹ️ Inferring metadata from wide-format DataFrame...")

        # Define known deterministic feature columns
        known_deterministic_features = {"month_id", "row", "col", "pg_id", "c_id"}

        # Identify deterministic features that exist in df_wide
        deterministic_features = [col for col in self.df_wide.columns if col in known_deterministic_features]

        # add all the features that do not have lists as entries
        for col in self.df_wide.columns:
            if col not in known_deterministic_features:
                if not self.df_wide[col].apply(lambda x: isinstance(x, list)).any():
                    deterministic_features.append(col)

        # Identify stochastic features:
        stochastic_features = []
        for col in self.df_wide.columns:
            if col not in deterministic_features:
                # Check if most (or all) values in the column are lists
                if self.df_wide[col].apply(lambda x: isinstance(x, list)).any():
                    stochastic_features.append(col)

        # Store metadata
        metadata = {
            "zstack_features": {
                "deterministic_features": deterministic_features,
                "stochastic_features": stochastic_features
            }
        }

        self.logger.info(f"✅ Metadata inferred: {self.metadata}")

        return metadata["zstack_features"]



    def to_wide_df(self, drop = False, assess = True):
        """
        Converts a stored zstack into a wide-format DataFrame.
        """

        if self.zstack is not None:
            self.df_wide = self.zstack_to_df_wide(self.zstack)


        else:
            self.logger.error("❌ No zstack available for conversion. Generate or load data first.")
            raise ValueError("No zstack available for conversion. Generate or load data first.")        


        if assess:
            self.logger.info("Assessing reconstruction of wide-format DataFrame...")

            try:
                np.testing.assert_allclose(self.zstack, self.df_wide_to_zstack(self.df_wide), rtol=1e-5)
                self.logger.info("✅ Assessment passed - zstack reconstructed from wide-format DataFrame matches the original.")

            except AssertionError as e:
                self.logger.exception(f"❌ Assessment failed - reconstructed zstack does not match the original! {e}")
                raise  # Re-raise to enforce fail-fast


        self.source_format = "df_wide"

        if drop: # we delete other data formats
            self.zstack = None

        return self.df_wide



    def to_zstack(self, drop = False, assess = True):
        """
        Converts a stored wide-format DataFrame  tensor into a zstack.
        """

        if self.df_wide is not None:
            self.zstack = self.df_wide_to_zstack(self.df_wide)

        else:
            self.logger.error("❌ No wide-format DataFrame available for conversion. Generate or load data first.")
            raise ValueError("No wide-format DataFrame available for conversion. Generate or load data first.")


        if assess:
            self.logger.info("Assessing reconstruction of zstack...")
            try:
                if self.df_wide is not None:
                    np.testing.assert_allclose(self.zstack, self.df_wide_to_zstack(self.df_wide), rtol=1e-5)


                self.logger.info("✅ Assessment passed - zstack reconstructed from DataFrame matches the original.")

            except AssertionError as e:
                self.logger.exception(f"❌ Assessment failed - reconstructed zstack does not match the original! {e}")
                raise  # Re-raise to enforce fail-fast

        self.source_format = "zstack"

        if drop:
            self.df_wide = None

        return self.zstack


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



  
    def df_wide_to_zstack(self, df_wide):
        """
        Converts a wide-format DataFrame (`df_wide`) back into a 5D NumPy `zstack` array.

        Args:
            df_wide (pd.DataFrame): Wide-format DataFrame where each feature column contains lists of sample values.

        Returns:
            np.ndarray: 5D zstack with shape `(month_id, row, col, num_features, num_samples)`.
        """

        if df_wide is None:
            raise ValueError("No wide-format DataFrame available for conversion.")

        # **🚀 Extract Grid Dimensions**
        unique_months = sorted(df_wide["month_id"].unique())  
        unique_rows = sorted(df_wide["row"].unique())
        unique_cols = sorted(df_wide["col"].unique())

        z_dim = len(unique_months)
        row_dim = len(unique_rows)
        col_dim = len(unique_cols)

        # **🚀 Extract Metadata for Feature Handling**
        feature_metadata = self.metadata["zstack_features"]
        deterministic_features = feature_metadata["deterministic_features"]
        stochastic_features = feature_metadata["stochastic_features"]

        num_deterministic = len(deterministic_features)
        num_stochastic = len(stochastic_features)
        num_features = num_deterministic + num_stochastic

        # **🚀 Determine Number of Samples from Any Stochastic Feature**
        sample_col = stochastic_features[0] if num_stochastic > 0 else deterministic_features[0]
        num_samples = len(df_wide[sample_col].iloc[0]) if isinstance(df_wide[sample_col].iloc[0], list) else 1

        # **🚀 Initialize an Empty 5D `zstack`**
        zstack = np.zeros((z_dim, row_dim, col_dim, num_features, num_samples), dtype=np.float32)

        # **🚀 Populate `zstack`**
        for z_idx, month in enumerate(unique_months):
            df_slice = df_wide[df_wide["month_id"] == month]

            for row_idx, row in enumerate(unique_rows):
                for col_idx, col in enumerate(unique_cols):
                    df_cell = df_slice[(df_slice["row"] == row) & (df_slice["col"] == col)]

                    if df_cell.empty:
                        continue  # Skip empty cells
                    
                    # **🚀 Populate Deterministic Features (Broadcast Across Samples)**
                    for f_idx, feature in enumerate(deterministic_features):
                        value = df_cell[feature].values[0]  # Extract single value
                        zstack[z_idx, row_idx, col_idx, f_idx, :] = value  # Copy across all samples

                    # **🚀 Populate Stochastic Features (Per Sample)**
                    for f_idx, feature in enumerate(stochastic_features):
                        feature_values = df_cell[feature].values[0]  # Extract list of samples
                        zstack[z_idx, row_idx, col_idx, num_deterministic + f_idx, :] = feature_values

        return zstack





class DataGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

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

        self.logger.info(f"✅ Synthetic zstack created with shape: {zstack.shape}")
        return zstack, feature_metadata


class TensorPlotter:

    """ TensorPlotter is a class designed to handle the visualization of tensor data and its derived DataFrames.
    It provides methods to plot z-stacks and wide-format DataFrames facilitating 
    the inspection and analysis of multi-dimensional data.

    Attributes:
        tensor_handler (TensorHandler): An instance of TensorHandler that provides the tensor data to be visualized.
        zstack (np.ndarray): A 5D numpy array representing the z-stack data with dimensions (z, height, width, features, samples).
        df_wide (pd.DataFrame): A wide-format DataFrame where each feature column contains lists of sample values.

    Methods:
        plot_zstack(num_z_slices=None, features=None, samples=None):
            Plots the stored z-stack showing different features, time steps, and sample draws.

        plot_df_wide(num_z_slices=5, feature_columns=None, num_samples=5):
            Each feature column contains a list of samples, so individual samples are plotted.
            
    """
    

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