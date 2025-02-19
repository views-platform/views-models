import numpy as np
import pandas as pd
import logging


class ZStackHandler:
    """
    Handles conversions between a 5D NumPy zstack and a wide-format DataFrame.
    """

    def __init__(self, data=None):
        """
        Initializes the ZStackHandler with either a 5D zstack or a wide-format DataFrame.

        Args:
            data (np.ndarray | pd.DataFrame | None): Input data in either zstack or DataFrame format.
        """

        self.generator = DataGenerator()

        self.zstack = None
        self.df_wide = None
        self.grid_shape = None  # (row, col)
        self.metadata = {"zstack_features": None, "df_columns": None}

        if data is None:
            logger.info("!! No data provided !!.")
            return

        if isinstance(data, np.ndarray):
            if len(data.shape) != 5:
                raise ValueError(f"Expected a 5D zstack (time, row, col, features, samples), but got shape {data.shape}.")
            self.zstack = data
            self.grid_shape = (data.shape[1], data.shape[2])  # Extract (row, col)
            logger.info(f"✅ Initialized with zstack of shape {data.shape}.")

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
        


    def generate_synthetic_zstack(self, *args, **kwargs):
        """
        Generates a synthetic 5D NumPy zstack and stores it internally.
        Ensures that generated data conforms to expected structure.
        """

        # ✅ Generate synthetic data (ensures correct format)
        self.zstack, self.metadata['zstack_features'] = self.generator.generate_synthetic_zstack(*args, **kwargs)

        # ✅ Validate that the generated zstack is 5D
        if len(self.zstack.shape) != 5:
            raise ValueError(f"❌ Expected a 5D zstack (time, row, col, features, samples), but got shape {self.zstack.shape}.")

        # ✅ Store grid dimensions
        time_dim, row_dim, col_dim, num_features, num_samples = self.zstack.shape
        self.grid_shape = (row_dim, col_dim)  # Store spatial dimensions
        self.metadata['num_features'] = num_features
        self.metadata['num_samples'] = num_samples

        # ✅ Reset other representations to avoid inconsistencies
        self.df_wide = None
        self.metadata.pop('df_columns', None)  # Ensure outdated df metadata is removed

        logger.info(f"✅ Synthetic zstack generated with shape {self.zstack.shape}.")

        return self.zstack, self.metadata


    def calculate_absolute_indices(df_wide, row_col_names=("row", "col"), time_col="month_id"):
        """
        Computes absolute indices for 'row', 'col', and 'month_id' in the DataFrame.    
        This is needed to turn a pandas DataFrame into a 5D zstack.

        Args:
            df (pd.DataFrame): The input DataFrame containing at least:
                - `row_col_names[0]` (e.g., 'row'): Row index.
                - `row_col_names[1]` (e.g., 'col'): Column index.
                - `time_col` (e.g., 'month_id'): Temporal index.

            row_col_names (tuple): Column names for spatial indices (row, col).
            time_col (str): Column name for time index.

        Returns:
            pd.DataFrame: A new DataFrame with additional absolute index columns:
                - `abs_row`
                - `abs_col`
                - `abs_month`
        """

        required_columns = set(row_col_names) | {time_col}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(f"❌ Missing required columns: {missing_columns}")

        row_name, col_name = row_col_names

        return df_wide.assign(
            abs_row=(df_wide[row_name] - df_wide[row_name].min()).astype(int),
            abs_col=(df_wide[col_name] - df_wide[col_name].min()).astype(int),
            abs_month=(df_wide[time_col] - df_wide[time_col].min()).astype(int)
        )



#
#
#
#
#
#
#    def _infer_grid_shape(self):
#        """Infers the (row, col) shape from the stored zstack."""
#        if self.zstack is not None:
#            _, row_dim, col_dim, _ = self.zstack.shape
#            self.grid_shape = (row_dim, col_dim)
#
#    def _check_df_columns(self):
#        """Ensures required columns exist in df_wide."""
#        required_columns = {"pg_id", "col", "row", "month_id", "c_id"}
#        missing = required_columns - set(self.df_wide.columns)
#        if missing:
#            raise ValueError(f"Missing columns in DataFrame: {missing}")
#
#    def _check_zstack_shape(self):
#        """Ensures zstack has expected dimensions."""
#        if self.zstack is None:
#            raise ValueError("No zstack loaded.")
#        if len(self.zstack.shape) != 4:
#            raise ValueError(f"ZStack must be 4D (time, row, col, features), but got {self.zstack.shape}")
#
#    def _infer_metadata_from_df(self):
#        """Infers metadata: deterministic and stochastic feature separation."""
#        deterministic_features = {"month_id", "row", "col", "pg_id", "c_id"}
#        deterministic = [col for col in self.df_wide.columns if col in deterministic_features]
#        stochastic = [col for col in self.df_wide.columns if col not in deterministic_features and self.df_wide[col].apply(lambda x: isinstance(x, list)).any()]
#        return {"deterministic_features": deterministic, "stochastic_features": stochastic}
#
#    def df_to_zstack(self):
#        """
#        Converts a stored wide-format DataFrame into a 4D NumPy zstack.
#        """
#        if self.df_wide is None:
#            raise ValueError("No wide-format DataFrame available for conversion.")
#
#        self.zstack = self._convert_df_to_zstack(self.df_wide)
#        self.source_format = "zstack"
#        return self.zstack
#
#    def zstack_to_df(self):
#        """
#        Converts a stored 4D NumPy zstack back into a wide-format DataFrame.
#        """
#        if self.zstack is None:
#            raise ValueError("No zstack available for conversion.")
#
#        self.df_wide = self._convert_zstack_to_df(self.zstack)
#        self.source_format = "df_wide"
#        return self.df_wide
#
#    def validate_conversion(self):
#        """
#        Ensures that converting DF → ZStack → DF retains data consistency.
#        """
#        if self.df_wide is None or self.zstack is None:
#            raise ValueError("Both DataFrame and ZStack must be available for validation.")
#
#        reconstructed_zstack = self._convert_df_to_zstack(self.df_wide)
#        np.testing.assert_allclose(self.zstack, reconstructed_zstack, rtol=1e-5)
#        print("✅ Validation passed: DF → ZStack → DF maintains data integrity.")
#
#    def save_zstack_npy(self, path):
#        """Saves the zstack as a .npy file."""
#        if self.zstack is None:
#            raise ValueError("No zstack to save.")
#        np.save(path, self.zstack)
#
#    def load_zstack_npy(self, path):
#        """Loads a zstack from a .npy file."""
#        if not os.path.isfile(path):
#            raise FileNotFoundError(f"No file found at {path}")
#        self.zstack = np.load(path)
#        self._infer_grid_shape()
#        self.source_format = "zstack"
#
#    def _convert_df_to_zstack(self, df):
#        """
#        Internal: Converts a DataFrame to a 4D NumPy zstack.
#        """
#        self._check_df_columns()
#        unique_months = sorted(df["month_id"].unique())
#        row_dim, col_dim = self.grid_shape
#        num_features = len(self.metadata["zstack_features"]["deterministic_features"]) + len(self.metadata["zstack_features"]["stochastic_features"])
#
#        num_time_steps = len(unique_months)
#        zstack = np.zeros((num_time_steps, row_dim, col_dim, num_features), dtype=np.float32)
#
#        for t_idx, month in enumerate(unique_months):
#            df_slice = df[df["month_id"] == month]
#
#            for _, row in df_slice.iterrows():
#                r, c = int(row["row"]), int(row["col"])
#                for f_idx, feature in enumerate(self.metadata["zstack_features"]["deterministic_features"]):
#                    zstack[t_idx, r, c, f_idx] = row[feature]
#
#                for f_idx, feature in enumerate(self.metadata["zstack_features"]["stochastic_features"], start=len(self.metadata["zstack_features"]["deterministic_features"])):
#                    zstack[t_idx, r, c, f_idx] = np.mean(row[feature])  # Taking mean as a placeholder
#
#        return zstack
#
#    def _convert_zstack_to_df(self, zstack):
#        """
#        Internal: Converts a 4D NumPy zstack to a DataFrame.
#        """
#        num_time_steps, row_dim, col_dim, num_features = zstack.shape
#        data = {"month_id": [], "row": [], "col": []}
#        for feature in self.metadata["zstack_features"]["deterministic_features"] + self.metadata["zstack_features"]["stochastic_features"]:
#            data[feature] = []
#
#        for t in range(num_time_steps):
#            for r in range(row_dim):
#                for c in range(col_dim):
#                    data["month_id"].append(t)
#                    data["row"].append(r)
#                    data["col"].append(c)
#                    for f_idx, feature in enumerate(self.metadata["zstack_features"]["deterministic_features"]):
#                        data[feature].append(zstack[t, r, c, f_idx])
#                    for f_idx, feature in enumerate(self.metadata["zstack_features"]["stochastic_features"], start=len(self.metadata["zstack_features"]["deterministic_features"])):
#                        data[feature].append([zstack[t, r, c, f_idx]])  # Store stochastic as lists
#
#        return pd.DataFrame(data)
#
#
#
#
#
#
#
#
#





















class DataGenerator:
    def __init__(self):
        pass

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

        return zstack, feature_metadata

