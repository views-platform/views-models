import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging

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

        self.logger = logging.getLogger(__name__)

        self.source_format = source_format
        self.metadata = {}  # To store column names for reconstruction
        self.grid_shape = None # To store grid dimensions for reconstruction

        if source_format == "zstack":
            self.zstack = data
            self.df_wide = None
            self.df_long = None
            self.tensor_canonical = None
            self.metadata["zstack_features"] = None # we can add more metadata here
            self.grid_shape = (data.shape[1], data.shape[2]) if data is not None else None

        elif source_format == "df_wide" and data is not None:

            if type(data.index) == pd.MultiIndex:
                data = data.reset_index(drop=False)

            # remane to column priogrid_gid to pg_id
            data.rename(columns={"priogrid_gid": "pg_id"}, inplace=True)

            self.df_wide = data
            self.zstack = None
            self.df_long = None
            self.tensor_canonical = None
            self.metadata["df_columns"] = list(data.columns)
            self.metadata["zstack_features"] = self.infer_metadata_from_wide_df()

            if "row" in data.columns and "col" in data.columns:
                row_dim = data["row"].max() + 1
                col_dim = data["col"].max() + 1
                self.grid_shape = (row_dim, col_dim)

        elif source_format == "df_long" and data is not None:

            # not implemented yet - log and raise an error

            self.logger.error("❌ initialization from long format is not implemented yet.")
            raise NotImplementedError("initialization from long format is not implemented yet.")
        
            # what is lagging here is just making sure that we get the zstack_features in the metadata infered from somewhere

            self.df_long = data
            self.zstack = None
            self.df_wide = None
            self.tensor_canonical = None
            self.metadata["df_long_columns"] = list(data.columns)

            if "row" in data.columns and "col" in data.columns:
                row_dim = data["row"].max() + 1
                col_dim = data["col"].max() + 1
                self.grid_shape = (row_dim, col_dim)

        elif source_format == "tensor_canonical" and data is not None:

            # not implemented yet - log and raise an error
            self.logger.error("❌ initialization from canonical tensor is not implemented yet.")
            raise NotImplementedError("initialization from canonical tensor is not implemented yet.")
        
            # what is lagging here is just making sure that we get the zstack_features in the metadata infered from somewhere

            self.tensor_canonical = data
            self.zstack = None
            self.df_wide = None
            self.df_long = None
            self.metadata["tensor_canonical_features"] = None # we can add more metadata here

            # 🚨 **Cannot Infer `grid_shape` Directly From `tensor_canonical`!**
            self.grid_shape = None  # Requires mapping pg_id → (row, col) externally

        else:
            self.zstack = None
            self.df_wide = None
            self.df_long = None
            self.tensor_canonical = None

        # composition
        self.plotter = TensorPlotter()  # Composition: Delegate plotting
        self.generator = DataGenerator()  # Composition: Delegate data generation

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

    def plot_df_long(self, *args, **kwargs):
        """Delegates to TensorPlotter"""
        TensorPlotter.plot_df_long(self.df_long, *args, **kwargs)

    def plot_tensor_canonical(self, *args, **kwargs):
        """Delegates to TensorPlotter"""
        TensorPlotter.plot_tensor_canonical(self.tensor_canonical, self.metadata['zstack_features'], *args, **kwargs)


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
        self.df_long = None

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
        Converts a stored zstack, long-format DataFrame, or canonical tensor into a wide-format DataFrame.
        """

        if self.zstack is not None:
            self.df_wide = self.zstack_to_df_wide(self.zstack)

        elif self.df_long is not None:

            # we create the zstack from the long format
            self.zstack = self.df_long_to_zstack(self.df_long)

            # then the wide format from the zstack
            self.df_wide = self.zstack_to_df_wide(self.zstack)

        elif self.tensor_canonical is not None:

            # we create the zstack from the canonical tensor
            self.zstack = self.tensor_canonical_to_zstack(self.tensor_canonical)

            # then the wide format from the zstack
            self.df_wide = self.zstack_to_df_wide(self.zstack)

        else:
            self.logger.error("❌ No zstack, long DataFrame, or canonical tensor available for conversion. Generate or load data first.")
            raise ValueError("No zstack, long DataFrame, or canonical tensor available for conversion. Generate or load data first.")        


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
            self.df_long = None
            self.tensor_canonical = None

        return self.df_wide


    def to_long_df(self, drop = False, assess = True):
        """
        Converts a stored zstack, wide-format DataFrame, or canonical tensor into a long-format DataFrame. 
        """

        if self.zstack is not None:
              self.df_long = self.zstack_to_df_long(self.zstack)


        elif self.df_wide is not None:
            
            # we create the zstack from the wide format
            self.zstack = self.df_wide_to_zstack(self.df_wide)

            # then the long format from the zstack
            self.df_long = self.zstack_to_df_long(self.zstack)


        elif self.tensor_canonical is not None:

            # we create the zstack from the canonical tensor
            self.zstack = self.tensor_canonical_to_zstack(self.tensor_canonical)

            # then the long format from the zstack
            self.df_long = self.zstack_to_df_long(self.zstack)

        else:
            self.logger.error("❌ No zstack, wide DataFrame, or canonical tensor available for conversion. Generate or load data first.")
            raise ValueError("No zstack, wide DataFrame, or canonical tensor available for conversion. Generate or load data first.")


        if assess:
            self.logger.info("Assessing reconstruction of long-format DataFrame...")

            try:
                np.testing.assert_allclose(self.zstack, self.df_long_to_zstack(self.df_long), rtol=1e-5)
                self.logger.info("✅ Assessment passed - zstack reconstructed from long-format DataFrame matches the original.")

            except AssertionError as e:
                self.logger.exception(f"❌ Assessment failed - reconstructed zstack does not match the original!. {e}")
                raise  # Re-raise to enforce fail-fast


        
        self.source_format = "df_long"

        if drop: # we delete other data formats
            self.zstack = None
            self.df_wide = None
            self.tensor_canonical = None
        
        return self.df_long



    def to_tensor_canonical(self, drop = False, assess = True):
        """
        Converts a stored zstack, wide-format DataFrame, or long-format DataFrame into a canonical tensor.
        """

        if self.zstack is not None:
            self.tensor_canonical = self.zstack_to_tensor_canonical(self.zstack)

        elif self.df_wide is not None:
            
            # we create the zstack from the wide format
            self.zstack = self.df_wide_to_zstack(self.df_wide)

            # then the canonical tensor from the zstack
            self.tensor_canonical = self.zstack_to_tensor_canonical(self.zstack)

        elif self.df_long is not None:

            # we create the zstack from the long format
            self.zstack = self.df_long_to_zstack(self.df_long)

            # then the canonical tensor from the zstack
            self.tensor_canonical = self.zstack_to_tensor_canonical(self.zstack)

        else:
            self.logger.error("❌ No zstack, wide DataFrame, or long DataFrame available for conversion. Generate or load data first.")
            raise ValueError("No zstack, wide DataFrame, or long DataFrame available for conversion. Generate or load data first.")


        if assess:
            self.logger.info("Assessing reconstruction of canonical tensor...")

            try:
                np.testing.assert_allclose(self.zstack, self.tensor_canonical_to_zstack(self.tensor_canonical), rtol=1e-5)
                self.logger.info("✅ Assessment passed - zstack reconstructed from canonical tensor matches the original.")

            except AssertionError as e:
                self.logger.exception(f"❌ Assessment failed - reconstructed zstack does not match the original! {e}")
                raise


        self.source_format = "tensor_canonical"

        if drop: # we delete other data formats
            self.zstack = None
            self.df_wide = None
            self.df_long = None

        return self.tensor_canonical


    def to_zstack(self, drop = False, assess = True):
        """
        Converts a stored wide-format DataFrame or long-format DataFrame or canonical tensor into a zstack.
        """

        if self.df_wide is not None:
            self.zstack = self.df_wide_to_zstack(self.df_wide)

        elif self.df_long is not None:
            self.zstack = self.df_long_to_zstack(self.df_long)

        elif self.tensor_canonical is not None:
            self.zstack = self.tensor_canonical_to_zstack(self.tensor_canonical)

        else:
            self.logger.error("❌ No wide-format DataFrame, long-format DataFrame, or canonical tensor available for conversion. Generate or load data first.")
            raise ValueError("No wide-format DataFrame, long-format DataFrame, or canonical tensor available for conversion. Generate or load data first.")


        if assess:
            self.logger.info("Assessing reconstruction of zstack...")
            try:
                if self.df_wide is not None:
                    np.testing.assert_allclose(self.zstack, self.df_wide_to_zstack(self.df_wide), rtol=1e-5)

                elif self.df_long is not None:
                    np.testing.assert_allclose(self.zstack, self.df_long_to_zstack(self.df_long), rtol=1e-5)

                elif self.tensor_canonical is not None:
                    np.testing.assert_allclose(self.zstack, self.tensor_canonical_to_zstack(self.tensor_canonical), rtol=1e-5)

                self.logger.info("✅ Assessment passed - zstack reconstructed from DataFrame matches the original.")

            except AssertionError as e:
                self.logger.exception(f"❌ Assessment failed - reconstructed zstack does not match the original! {e}")
                raise  # Re-raise to enforce fail-fast

        self.source_format = "zstack"

        if drop:
            self.df_wide = None
            self.df_long = None
            self.tensor_canonical = None

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

#

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


    def zstack_to_tensor_canonical(self, zstack):
        """
        Converts a 5D NumPy `zstack` array into `tensor_canonical` format with shape:
        `(month_id, pg_id, num_samples, num_features)`
        """
        # 🚀 **Extract Feature Metadata**
        feature_metadata = self.metadata['zstack_features']
        deterministic_features = feature_metadata['deterministic_features']
        stochastic_features = feature_metadata['stochastic_features']

        z_dim, row_dim, col_dim, num_features, num_samples = zstack.shape

        # 🚀 **Get PRIO Grid ID from `zstack` (pg_id)**
        if "pg_id" not in deterministic_features:
            raise ValueError("pg_id is missing from deterministic features")

        pg_id_index = deterministic_features.index("pg_id")
        pg_id = zstack[:, :, :, pg_id_index, 0]  # (z, row, col)

        # 🚀 **Flatten Spatial Information**
        pg_id_flat = pg_id.reshape(z_dim, -1)  # (z, row*col)

        # 🚀 **Flatten Deterministic Features**
        deterministic_tensors = [
            zstack[:, :, :, idx, 0].reshape(z_dim, -1, 1) for idx, feature in enumerate(deterministic_features)
        ]

        # 🚀 **Flatten Stochastic Features**
        stochastic_tensors = [
            zstack[:, :, :, len(deterministic_features) + idx, :].reshape(z_dim, -1, num_samples, 1)
            for idx, feature in enumerate(stochastic_features)
        ]

        # 🚀 **Stack Features Correctly**
        deterministic_stack = np.concatenate(deterministic_tensors, axis=-1)  # (z, pg_id, num_deterministic_features)
        stochastic_stack = np.concatenate(stochastic_tensors, axis=-1)  # (z, pg_id, num_samples, num_stochastic_features)

        # 🚀 **Fix Dimensionality Mismatch**
        deterministic_stack = np.expand_dims(deterministic_stack, axis=2)  # (z, pg_id, 1, num_deterministic_features)
        deterministic_stack = np.tile(deterministic_stack, (1, 1, num_samples, 1))  # (z, pg_id, num_samples, num_deterministic_features)

        # 🚀 **Concatenate Along Feature Axis**
        tensor_canonical = np.concatenate([deterministic_stack, stochastic_stack], axis=-1)  # (z, pg_id, num_samples, num_features)

        return tensor_canonical

  
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


    def df_long_to_zstack(self, df_long):
        """
        Converts a long-format DataFrame (`df_long`) back into a 5D NumPy `zstack` array.

        Args:
            df_long (pd.DataFrame): Long-format DataFrame where each row represents a single `(month_id, row, col, sample_id)`.

        Returns:
            np.ndarray: 5D `zstack` array with shape `(month_id, row, col, num_features, num_samples)`.
        """

        if df_long is None:
            raise ValueError("No long-format DataFrame available for conversion.")

        # 🚀 **Extract Grid Dimensions**
        unique_months = sorted(df_long["month_id"].unique())
        unique_rows = sorted(df_long["row"].unique())
        unique_cols = sorted(df_long["col"].unique())

        z_dim = len(unique_months)
        row_dim = len(unique_rows)
        col_dim = len(unique_cols)

        # 🚀 **Extract Metadata for Feature Handling**
        feature_metadata = self.metadata["zstack_features"]
        deterministic_features = feature_metadata["deterministic_features"]
        stochastic_features = feature_metadata["stochastic_features"]

        num_deterministic = len(deterministic_features)
        num_stochastic = len(stochastic_features)
        num_features = num_deterministic + num_stochastic

        # **🚀 Determine Number of Samples from Stochastic Data**
        unique_samples = sorted(df_long["sample_id"].unique())
        num_samples = len(unique_samples)

        # **🚀 Initialize an Empty 5D `zstack`**
        zstack = np.zeros((z_dim, row_dim, col_dim, num_features, num_samples), dtype=np.float32)

        # **🚀 Populate `zstack`**
        for z_idx, month in enumerate(unique_months):
            df_slice = df_long[df_long["month_id"] == month]

            for row_idx, row in enumerate(unique_rows):
                for col_idx, col in enumerate(unique_cols):
                    df_cell = df_slice[(df_slice["row"] == row) & (df_slice["col"] == col)]

                    if df_cell.empty:
                        continue  # Skip empty cells

                    # **🚀 Populate Deterministic Features (Broadcast Across Samples)**
                    for f_idx, feature in enumerate(deterministic_features):
                        value = df_cell[feature].values[0]  # Deterministic = 1 value per (month_id, row, col)
                        zstack[z_idx, row_idx, col_idx, f_idx, :] = value  # Copy across all samples

                    # **🚀 Populate Stochastic Features (Per Sample)**
                    for f_idx, feature in enumerate(stochastic_features):
                        for s_idx, sample in enumerate(unique_samples):
                            df_sample = df_cell[df_cell["sample_id"] == sample]
                            if not df_sample.empty:
                                zstack[z_idx, row_idx, col_idx, num_deterministic + f_idx, s_idx] = df_sample[feature].values[0]

        return zstack


    def tensor_canonical_to_zstack(self, tensor_canonical):
        """
        Converts a canonical tensor `(month_id, pg_id, num_samples, num_features)`
        back into a `zstack` with shape `(month_id, row, col, num_features, num_samples)`.

        Args:
            tensor_canonical (np.ndarray): Canonical tensor `(z, pg_id, num_samples, num_features)`.

        Returns:
            np.ndarray: Reconstructed 5D `zstack` `(month_id, row, col, num_features, num_samples)`.
        """

        # **🚀 Extract Metadata**
        feature_metadata = self.metadata["zstack_features"]
        deterministic_features = feature_metadata["deterministic_features"]
        stochastic_features = feature_metadata["stochastic_features"]

        num_deterministic = len(deterministic_features)
        num_stochastic = len(stochastic_features)
        num_features = num_deterministic + num_stochastic

        # **🚀 Extract Tensor Shapes**
        z_dim, pg_dim, num_samples, _ = tensor_canonical.shape

        # **🚀 Ensure `grid_shape` is available**
        if self.grid_shape is None:
            raise ValueError("`grid_shape` is not set. Run `set_grid_shape()` first.")

        row_dim, col_dim = self.grid_shape  # Extract row and column dimensions

        # **🚀 Retrieve PRIO Grid Mapping (pg_id → row, col)**
        pg_id_map = self.pg_id_to_row_col()  # Dictionary {pg_id: (row, col)}

        # **🚀 Initialize Empty `zstack`**
        zstack = np.zeros((z_dim, row_dim, col_dim, num_features, num_samples), dtype=np.float32)

        # 🚀 **Iterate Over pg_ids to Place Values Correctly**
        for pg_idx in range(pg_dim):
            if pg_idx not in pg_id_map:
                continue  # Skip missing pg_id mappings

            row, col = pg_id_map[pg_idx]

            # 🚀 **Extract Deterministic Features (Same Across Samples)**
            deterministic_values = tensor_canonical[:, pg_idx, 0, :num_deterministic]  # (z, num_deterministic)

            # 🚀 **Expand Deterministic Features Across Samples**
            deterministic_values = np.repeat(deterministic_values[:, np.newaxis, :], num_samples, axis=1)  # (z, num_samples, num_deterministic)

            # 🚀 **Insert Deterministic Features Into `zstack`**
            zstack[:, row, col, :num_deterministic, :] = np.transpose(deterministic_values, (0, 2, 1))  # (z, row, col, num_deterministic, num_samples)

            # 🚀 **Extract Stochastic Features (Per Sample)**
            stochastic_values = tensor_canonical[:, pg_idx, :, num_deterministic:]  # (z, num_samples, num_stochastic)

            # 🚀 **Insert Stochastic Features Into `zstack`**
            zstack[:, row, col, num_deterministic:, :] = np.transpose(stochastic_values, (0, 2, 1))  # (z, row, col, num_stochastic, num_samples)

        return zstack


    def pg_id_to_row_col(self):
        """
        Constructs a mapping from `pg_id` to `(row, col)` coordinates.

        Returns:
            dict: A dictionary mapping `pg_id` -> (row, col)
        """

        # 🚀 **Extract `pg_id` From `zstack`**
        pg_id_feature = "pg_id"
        feature_metadata = self.metadata["zstack_features"]

        if pg_id_feature not in feature_metadata["deterministic_features"]:
            raise ValueError(f"Expected `{pg_id_feature}` in deterministic features, but it is missing!")

        pg_id_index = feature_metadata["deterministic_features"].index(pg_id_feature)

        # 🚀 **Get Grid Dimensions**
        z_dim, row_dim, col_dim, _, _ = self.zstack.shape

        # 🚀 **Extract `pg_id` from zstack**
        pg_id_array = self.zstack[0, :, :, pg_id_index, 0]  # Extract from first timestep

        # 🚀 **Build Mapping**
        pg_id_map = {}
        for row in range(row_dim):
            for col in range(col_dim):
                pg_id = pg_id_array[row, col]
                if pg_id not in pg_id_map:
                    pg_id_map[pg_id] = (row, col)  # Store the first occurrence

        return pg_id_map




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
    def plot_tensor_canonical(tensor_canonical, metadata, num_z_slices=5, num_samples=5, features=None):
        """
        Plots a sanity check for the canonical tensor representation (`tensor_canonical`).
        - Deterministic features are plotted as single values (repeated across all samples).
        - Stochastic features are plotted per `(sample_id)`, properly mapped to `pg_id`.

        Args:
            tensor_canonical (np.ndarray): Canonical tensor with shape `(month_id, pg_id, num_samples, num_features)`.
            metadata (dict): Metadata containing feature names and classifications.
            num_z_slices (int, optional): Number of `month_id` slices to visualize. Default is 5.
            num_samples (int, optional): Number of samples per stochastic feature to plot. Default is 5.
            features (list, optional): List of feature column names to plot. If None, all features are used.
        """

        if tensor_canonical is None:
            raise ValueError("No canonical tensor available for plotting.")

        # Extract metadata
        deterministic_features = metadata["deterministic_features"]
        stochastic_features = metadata["stochastic_features"]

        # **Feature Selection Handling**
        if features is not None:
            deterministic_features = [f for f in deterministic_features if f in features]
            stochastic_features = [f for f in stochastic_features if f in features]

        num_z_slices = min(num_z_slices, tensor_canonical.shape[0])  # Ensure we don't exceed available time steps
        num_samples = min(num_samples, tensor_canonical.shape[2])  # Ensure we don't exceed available samples

        num_deterministic = len(deterministic_features)
        num_stochastic = len(stochastic_features) * num_samples
        num_rows = num_deterministic + num_stochastic  # ✅ Avoids `IndexError`

        # **Create a fake spatial grid for visualization** 
        pg_id_positions = np.arange(tensor_canonical.shape[1])  # Create fake spatial positions for `pg_id`
        width = int(np.sqrt(len(pg_id_positions)))  # Approximate a square layout
        height = width  # Keep a square shape

        # 🚀 **Create Figure & Subplots**
        fig, axes = plt.subplots(max(1, num_rows), max(1, num_z_slices), figsize=(15, 3 * max(1, num_rows)))
        axes = np.atleast_2d(axes)  # ✅ Ensures 2D structure

        row_idx = 0  # ✅ Tracks subplot row

        for z in range(num_z_slices):  
            row_idx = 0  # ✅ Reset row index for each `z` iteration

            # 🚀 **Plot Deterministic Features**
            for f_idx, feature_name in enumerate(deterministic_features):
                ax = axes[row_idx, z] if num_z_slices > 1 else axes[row_idx]

                feature_values = tensor_canonical[z, :, 0, f_idx]  # Only first sample since deterministic
                
                # should be across all samples and all time steps - sample doesn't matter so 0 is fine
                vmin, vmax = np.min(tensor_canonical[:, :, 0, f_idx]), np.max(tensor_canonical[:, :, 0, f_idx])

                scatter = ax.scatter(pg_id_positions % width, height - (pg_id_positions // width), 
                                     s=50, c=feature_values, cmap='viridis', vmin=vmin, vmax=vmax)

                if row_idx == 0:
                    ax.set_title(f"Month {z+1}")

                if z == 0:
                    ax.set_ylabel(f"{feature_name} (Det)")

                ax.set_xticks([])
                ax.set_yticks([])
                row_idx += 1  # ✅ Move to next subplot row

            # 🚀 **Plot Stochastic Features**
            for f_idx, feature_name in enumerate(stochastic_features):
                for s_idx in range(num_samples):
                    ax = axes[row_idx, z] if num_z_slices > 1 else axes[row_idx]

                    feature_values = tensor_canonical[z, :, s_idx, num_deterministic + f_idx]

                    # should be across all samples and all time steps
                    vmin, vmax = np.min(tensor_canonical[:, :, s_idx, num_deterministic + f_idx]), np.max(tensor_canonical[:, :, s_idx, num_deterministic + f_idx])

                    scatter = ax.scatter(pg_id_positions % width, height - (pg_id_positions // width), 
                                         s=50, c=feature_values, cmap='viridis', vmin=vmin, vmax=vmax)

                    if row_idx == 0:
                        ax.set_title(f"Month {z+1}")

                    if z == 0:
                        ax.set_ylabel(f"{feature_name} (S={s_idx})")

                    ax.set_xticks([])
                    ax.set_yticks([])
                    row_idx += 1  # ✅ Move to next subplot row

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

