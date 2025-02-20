import os
import numpy as np
import logging
from typing import List, Optional, Tuple, Set
from matplotlib import pyplot as plt
import pandas as pd
import pickle

class ZStackManager:
    """
    Manages structured storage for stochastic and deterministic Z-stacks.

    Attributes:
    -----------
    shape_stochastic : tuple
        Shape of the stochastic Z-stack (e.g., (months, height, width, features, samples)).
    
    shape_deterministic : tuple
        Shape of the deterministic Z-stack (e.g., (months, height, width, features)).
        This remains fixed across all samples.
    
    zstack : dict
        Dictionary containing:
        - "stochastic" : np.ndarray -> Stores stochastic samples.
        - "deterministic" : np.ndarray -> Stores deterministic data (shared across samples).
    
    Methods:
    --------
    (To be implemented)
    - add_stochastic_zstack()
    - add_deterministic_zstack()
    - get_sample()
    - get_full_zstack()
    """

    def __init__(self, shape_stochastic: tuple, shape_deterministic: tuple):
        """
        Initializes the ZStackManager with pre-allocated memory for efficiency.

        Parameters:
        -----------
        shape_stochastic : tuple
            Expected shape for the stochastic Z-stack. Should be (months, height, width, features, samples).
        
        shape_deterministic : tuple
            Expected shape for the deterministic Z-stack. Should be (months, height, width, features).
        
        Raises:
        -------
        ValueError:
            - If the first four dimensions of shape_stochastic and shape_deterministic do not match.
        """

        # Initialize logger
        self.logger = logging.getLogger(__name__) # Create logger

        # Validate input shapes
        if shape_stochastic[:3] != shape_deterministic[:3]:
            raise ValueError(
                f"❌ Shape mismatch: Deterministic shape {shape_deterministic} must match "
                f"stochastic shape {shape_stochastic} (excluding feature and samples dimension)."
            )
        
        self.shape_stochastic = shape_stochastic
        self.shape_deterministic = shape_deterministic

        self.n_months = shape_stochastic[0]
        self.height = shape_stochastic[1]
        self.width = shape_stochastic[2]
        self.n_stochastic_features = shape_stochastic[3]
        self.n_deterministic_features = shape_deterministic[3]
        self.n_features = self.n_stochastic_features + self.n_deterministic_features
        self.n_samples = shape_stochastic[-1]

        # Pre-allocate memory for efficiency
        self.zstack = {
            "stochastic": np.zeros(shape_stochastic, dtype=np.float32),
            "deterministic": np.zeros(shape_deterministic, dtype=np.float32)
        }

        # Metadata to store feature names
        self.metadata = {
            "stochastic_features": None,
            "deterministic_features": None
        }

        init_msg = (
            f"\n✅ ZStackManager initialized successfully with matching shapes:\n"
            f"   ├── {shape_stochastic} (stochastic)\n"
            f"   ├── {shape_deterministic} (deterministic)\n"
            f"   ├── Number of months: {self.n_months}\n"
            f"   ├── Height: {self.height}\n"
            f"   ├── Width: {self.width}\n"
            f"   ├── Number of stochastic features: {self.n_stochastic_features}\n"
            f"   ├── Number of deterministic features: {self.n_deterministic_features}\n"
            f"   └── Number of samples: {self.n_samples}"
        )

        self.logger.info(init_msg)


    def set_stochastic_zstack(self, data: np.ndarray):
        
        """Stores the entire stochastic Z-stack at once."""

        if data.shape != self.shape_stochastic:
            raise ValueError(f"Expected {self.shape_stochastic}, got {data.shape}.")

        self.zstack["stochastic"] = data

        self.logger.info(f"\n✅Stochastic Z-stack set successfully.")

    def add_stochastic_zstack(self, data: np.ndarray):

        """ Add new stochastic features to the Z-stack."""

        # check if month, height, and width match
        if data.shape[0:3] != self.shape_stochastic[0:3]:
            raise ValueError(
                f"❌ Expected months, height, and width: {self.shape_stochastic[0:3]}, got {data.shape[0:3]}."
            )
        
        # check if number of samples match
        if data.shape[-1] != self.n_samples:
            raise ValueError(
                f"❌ Expected number of samples: {self.n_samples}, got {data.shape[-1]}."
            )
        
        # Log how many features are being added
        n_new_features = data.shape[3]
        self.logger.info(f"\n➕ Adding {n_new_features} new stochastic features to the Z-stack.")

        # concatenate new data along the feature axis
        self.zstack["stochastic"] = np.concatenate([self.zstack["stochastic"], data], axis=3)

        # update the number of stochastic features and total features
        self.n_stochastic_features += n_new_features
        self.n_features += n_new_features

        self.logger.info(f"\n✅ New stochastic zstack added successfully.\nNew shape: {self.zstack['stochastic'].shape}")


    def set_deterministic_zstack(self, data: np.ndarray):
        """
        Stores the entire deterministic Z-stack at once.

        Args:
        - data (np.ndarray): The deterministic Z-stack with shape (months, height, width, features).

        Raises:
        - ValueError: If the shape of `data` does not match the expected shape.
        """
        if data.shape != self.shape_deterministic:
            raise ValueError(f"❌ Expected {self.shape_deterministic}, got {data.shape}.")

        self.zstack["deterministic"] = data

        self.logger.info(f"\n✅ Deterministic Z-stack set successfully.")


    def add_deterministic_zstack(self, data: np.ndarray):
        """
        Adds new deterministic features to the Z-stack.

        Args:
        - data (np.ndarray): A numpy array of deterministic features to be added.  
                             Expected shape: (months, height, width, new_features).

        Raises:
        - ValueError: If the (months, height, width) dimensions do not match the existing deterministic Z-stack.
        """

        # Check if (months, height, width) match
        if data.shape[:3] != self.shape_deterministic[:3]:
            raise ValueError(
                f"❌ Expected (months, height, width): {self.shape_deterministic[:3]}, got {data.shape[:3]}."
            )

        # Log how many features are being added
        n_new_features = data.shape[3]
        self.logger.info(f"\n➕ Adding {n_new_features} new deterministic features to the Z-stack.")

        # Concatenate new deterministic features along the feature axis
        self.zstack["deterministic"] = np.concatenate([self.zstack["deterministic"], data], axis=3)

        # Update the number of deterministic features and total features
        self.n_deterministic_features += n_new_features
        self.n_features += n_new_features

        self.logger.info(f"\n✅ New deterministic Z-stack added successfully.\nNew shape: {self.zstack['deterministic'].shape}")


    def set_feature_names(self, stochastic_features: Optional[List[str]] = None, deterministic_features: Optional[List[str]] = None):
        """
        Sets and validates feature names for the stochastic and deterministic Z-stacks.

        Ensures:
        - No duplicate names within stochastic or deterministic features.
        - No overlap between stochastic and deterministic feature names.
        - Correct number of feature names provided.
        - Case-insensitive uniqueness.

        Args:
        -----
        - stochastic_features (List[str], optional): Names for stochastic features.
        - deterministic_features (List[str], optional): Names for deterministic features.

        Raises:
        -------
        - ValueError: If duplicates are found within or across feature sets, or if the provided names
                      do not match the expected number of features.
        """

        def clean_and_validate(names: Optional[List[str]], expected_count: int, feature_type: str) -> Tuple[Tuple[str, ...], Set[str]]:
            """Helper function to clean feature names and perform validation checks."""
            if names is None:
                return (), set()  # Return empty tuple and set

            # Clean names (strip spaces, lowercase for uniqueness check)
            cleaned_names = {name.strip().lower() for name in names}

            # Ensure uniqueness within the category
            if len(cleaned_names) != len(names):
                raise ValueError(f"❌ Duplicate {feature_type} feature names detected: {names}")

            # Ensure correct number of features
            if len(names) != expected_count:
                raise ValueError(
                    f"❌ Mismatch: Expected {expected_count} {feature_type} feature names, got {len(names)}."
                )

            return tuple(names), cleaned_names  # Preserve original casing for logging

        # Validate and assign feature names
        self.metadata["stochastic_features"], cleaned_stochastic = clean_and_validate(
            stochastic_features, self.shape_stochastic[3], "stochastic"
        )
        self.metadata["deterministic_features"], cleaned_deterministic = clean_and_validate(
            deterministic_features, self.shape_deterministic[3], "deterministic"
        )

        # Check for duplicates across feature types
        common_names = cleaned_stochastic & cleaned_deterministic
        if common_names:
            raise ValueError(f"❌ Feature names overlap between stochastic and deterministic: {common_names}")

        # Logging
        if self.metadata["stochastic_features"]:
            self.logger.info(f"✅ Stochastic feature names set successfully: {self.metadata['stochastic_features']}")
        if self.metadata["deterministic_features"]:
            self.logger.info(f"✅ Deterministic feature names set successfully: {self.metadata['deterministic_features']}")



    def zstack_to_df_wide(self, exclude_zstack_columns: Optional[bool] = None) -> pd.DataFrame:
        """
        Converts the Z-stacks into a wide-format Pandas DataFrame.

        Args:
        -----
        exclude_columns (List[str], optional): 
            List of column names to exclude from the final DataFrame.
            Options: ["height", "width", "Z"].

        Returns:
        --------
        pd.DataFrame:
            A DataFrame where:
            - Stochastic feature columns contain lists of sample values.
            - Deterministic feature columns contain scalar values.
            - Includes (z_month_id, height, width) as explicit coordinate columns.
        """

        self.logger.info("📦 Converting Z-stack to DataFrame (Wide Format)...")

        # Extract shapes
        months, height, width, n_stochastic_features, n_samples = self.shape_stochastic
        _, _, _, n_deterministic_features = self.shape_deterministic

        # Compute total rows (flattening spatial-temporal dimensions)
        total_rows = months * height * width

        # Generate coordinate indices (months, height, width)
        z_months, heights, widths = np.meshgrid(
            np.arange(months), np.arange(height), np.arange(width), indexing="ij"
        )

        # Flatten these to create coordinate columns
        z_months = z_months.flatten()
        heights = heights.flatten()
        widths = widths.flatten()

        # Flatten and reshape stochastic Z-stack to (rows, features, samples)
        stochastic_flat = self.zstack["stochastic"].reshape(total_rows, n_stochastic_features, n_samples)

        # Convert to lists per feature (axis 1 = features)
        stochastic_feature_lists = np.moveaxis(stochastic_flat, 1, 0).tolist()  # (features, rows) → list of lists

        # Flatten deterministic Z-stack to (rows, features)
        deterministic_flat = self.zstack["deterministic"].reshape(total_rows, n_deterministic_features)

        # Retrieve feature names from metadata or use defaults
        stochastic_columns = (
            self.metadata["stochastic_features"]
            if self.metadata["stochastic_features"]
            else [f"stochastic_feature_{i:03d}" for i in range(n_stochastic_features)]
        )

        deterministic_columns = (
            self.metadata["deterministic_features"]
            if self.metadata["deterministic_features"]
            else [f"deterministic_feature_{i:03d}" for i in range(n_deterministic_features)]
        )

        # Construct DataFrame
        df_dict = {
            "z_months": z_months,
            "height": heights,
            "width": widths,
            **{stochastic_columns[i]: stochastic_feature_lists[i] for i in range(n_stochastic_features)},
            **{deterministic_columns[i]: deterministic_flat[:, i] for i in range(n_deterministic_features)},
        }

        df = pd.DataFrame(df_dict)

        # If "month_id" exists in deterministic features, rename it to avoid confusion
        if "month_id" in df.columns:
            df.rename(columns={"month_id": "universal_month_id"}, inplace=True)

        # Apply zstack column exclusion if specified
        if exclude_zstack_columns:
            exclude_columns = ["height", "width", "z_months"]
            df.drop(columns=exclude_columns, inplace=True)

        self.logger.info(f"✅ DataFrame created successfully with shape: {df.shape}")

        return df


    def store_dataframe(self, filepath: str, format: str = "parquet"):
        """
        Stores the generated DataFrame in the specified format.

        Args:
        -----
        - filepath (str): The file path to store the DataFrame.
        - format (str, optional): The format to save the file. Defaults to "parquet".
                                  Options: "parquet", "csv".

        Raises:
        -------
        - ValueError: If the DataFrame does not exist or the format is unsupported.
        """

        if not hasattr(self, "df") or self.df is None:
            raise ValueError("❌ No DataFrame found. Ensure it is generated before saving.")

        format = format.lower()
        valid_formats = {"parquet", "csv", "pkl"}
        if format not in valid_formats:
            raise ValueError(f"❌ Unsupported format: {format}. Choose from {valid_formats}.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save DataFrame
        if format == "parquet":
            self.df.to_parquet(filepath, index=False)
        elif format == "csv":
            self.df.to_csv(filepath, index=False)
        elif format == "pkl":
            self.df.to_pickle(filepath)

        self.logger.info(f"✅ DataFrame successfully saved to {filepath} ({format}).")


    def store_zstack(self, filepath: str, format: str = "npz"):
        """
        Stores the Z-stack (both stochastic & deterministic) in the specified format.

        Args:
        -----
        - filepath (str): The file path to store the Z-stack.
        - format (str, optional): The format to save the file. Defaults to "npz".
                                  Options: "npz", "pkl".

        Raises:
        -------
        - ValueError: If no Z-stack data exists.
        """

        if not self.zstack or not any(self.zstack.values()):
            raise ValueError("❌ No Z-stack data available. Ensure data is set before saving.")
        
        format = format.lower()
        valid_formats = {"npz", "pkl"}
        if format not in valid_formats:
            raise ValueError(f"❌ Unsupported format: {format}. Choose from {valid_formats}.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save zstack
        if format ==  "npz":
            np.savez_compressed(filepath, **self.zstack)

        elif format == "pkl":
            with open(filepath, "wb") as f:
                pickle.dump(self.zstack, f)


        self.logger.info(f"✅ Z-stack successfully saved to {filepath} (.npz format).")



    def __repr__(self):
        """Returns a summary of the ZStackManager instance."""
        return (
            f"ZStackManager(\n"
            f"  Stochastic Shape    : {self.shape_stochastic}\n"
            f"  Deterministic Shape : {self.shape_deterministic}\n"
            f")"
        )




































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

        self.logger.info(f"\n✅ Synthetic zstack created with shape: {zstack.shape}")
        return zstack, feature_metadata
    















































class ZStackPlotter:

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