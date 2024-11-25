from typing import Optional
import datetime
from stepshift.views import StepshiftedModels
from views_partitioning import DataPartitioner
from views_schema.models import ModelMetadata
import pandas as pd
import pickle
# import validation

class ViewsRun():
    """
    ViewsRun
    ========

    parameters:
        partitioner: views_partitioning.DataPartitioner
        models: stepshift.views.StepshiftedModels

    This class combines modelling and partitioning, delivering predictions in a
    familiar format and concise API.

    Takes two arguments, a views_partitioning.DataPartitioner instance, and a
    stepshift.StepshiftedModels instance. These instances determine how data is
    subset, and how modelling happens.
    """

    def __init__(self, partitioner: DataPartitioner, models: StepshiftedModels):
        self._models = models

        # Partitioner is shifted to include training data covering N steps
        # behind the specified training period.
        self._shifted_partitioner = partitioner.shift_left(self._models._steps_extent)
        self._partitioner = partitioner

    # @validation.views_validate
    def fit(self, partition_name: str, timespan_name: str, data: pd.DataFrame)-> None:
        """
        fit
        ===

        parameters:
            partition_name (str)
            timespan_name (str)
            data (pandas.DataFrame)

        Fit the model to the named partition. partition_name must be present in
        the DataPartitioner.
        """

        self._models.fit(self._shifted_partitioner(partition_name,timespan_name,data))

    # @validation.views_validate
    def predict(self, partition_name: str, timespan_name: str, data: pd.DataFrame, proba: bool = False)-> pd.DataFrame:
        """
        predict
        =======

        parameters:
            partition_name (str)
            timespan_name (str)
            data (pandas.DataFrame)

        Predict with data from the named partition. partition_name must be
        present in the DataPartitioner. Returns predictions within the selected
        timespan.
        """

        if proba:
            predictions = self._models.predict_proba(
                self._shifted_partitioner(partition_name, timespan_name, data),
                combine=False)
        else:
            predictions = self._models.predict(
                self._shifted_partitioner(partition_name, timespan_name, data),
                combine=False)

        predictions.index.names = data.index.names
        data = data.merge(predictions, how = "left", left_index = True, right_index = True)
        return self._partitioner(partition_name,timespan_name,data)

    # @validation.views_validate
    def future_predict(self, partition_name: str, timespan_name: str, data: pd.DataFrame, keep_specific: bool = False, proba: bool = False) -> pd.DataFrame:
        """
        future_predict
        ==============

        parameters:
            partition_name (str)
            timespan_name (str)
            data (pandas.DataFrame)
            keep_specific (bool) = False: Return step-specific predictions as well

        returns:
            pandas.DataFrame: Dataframe containing future predictions

        Predict into the future using data from the last point in time in the
        selected partition. Returns step-combined predictions.
        """

        if proba:
            predictions = self._models.predict_proba(
                self._shifted_partitioner(partition_name, timespan_name, data),
                combine=False)
        else:
            predictions = self._models.predict(
                self._shifted_partitioner(partition_name, timespan_name, data),
                combine=False)

        end = self._shifted_partitioner.partitions.partitions[partition_name].timespans[timespan_name].end + 1
        future_preds = predictions.loc[end:]
        if not keep_specific:
            future_preds = future_preds[["step_combined"]]

        return future_preds

    def future_point_predict(self, time: int, data: pd.DataFrame, keep_specific: bool = False, proba: bool = False) -> pd.DataFrame:
        """
        future_point_predict
        ====================

        parameters:
            time (int): Point in time to predict from, given in views-month (months since 1979-12)
            data (pandas.DataFrame): Data to predict from
            keep_specific (bool) = False: Return step-specific predictions as well

        returns:
            pandas.DataFrame

        Predict into the future from a point in time.
        """

        if proba:
            predictions = self._models.predict_proba(
                data.loc[time - self._models._steps_extent: time],
                combine=False
            )
        else:
            predictions = self._models.predict(
                data.loc[time - self._models._steps_extent: time],
                combine=False
                )

        if not keep_specific:
            predictions = predictions[["step_combined"]]

        return predictions.loc[time+1 : time + self._models._steps_extent]

    def create_model_metadata(self,
            author: str,
            queryset_name: str,
            training_partition_name: str,
            training_timespan_name: str = "train",
            training_date: Optional[datetime.datetime] = None) -> ModelMetadata:
        """
        create_model_metadata
        ===============

        parameters:
            author (str): The author of the model (you)
            queryset_name (str): The name of the queryset used to train the model
            training_partition_name (str): The name of the partition containing the training timespan
            training_timespan_name (str): The name of the training timespan = "train"
            training_date (Optional[datetime.datetime]) = None (defaults to datetime.datetime.now())
        returns:
            views_schema.models.ModelMetadata

        Create a metadata instance based on provided metadata and data from the
        model and partitioner objects associated with the run.
        """

        training_date = training_date if training_date is not None else datetime.datetime.now()
        train_start, train_end = self._partitioner.partitions.partitions[training_partition_name].timespans[training_timespan_name]

        return ModelMetadata(
                author        = author,
                queryset_name = queryset_name,
                steps         = self._models._steps,
                train_start   = train_start,
                train_end     = train_end,
                training_date = training_date,
                )


    def save(self, path: str) -> None:
        with open(path, "wb") as file:
            pickle.dump(self, file)


    @property
    def models(self):
        return self._models.models

    @property
    def steps(self):
        return [*self._models.models.keys()]
    

if __name__ == "__main__":
    from ingester3.ViewsMonth import ViewsMonth
    from lightgbm import LGBMRegressor

    def get_partitioner_dict(partition, step=36):

        """Returns the partitioner_dict for the given partition."""

        if partition == 'calibration':

            partitioner_dict = {"train":(121,396),"predict":(397,444)} # calib_partitioner_dict - (01/01/1990 - 12/31/2012) : (01/01/2013 - 31/12/2015)

        if partition == 'testing':

            partitioner_dict = {"train":(121,444),"predict":(445,492)} 

        if partition == 'forecasting':

            month_last = ViewsMonth.now().id - 2 # minus 2 because the current month is not yet available. Verified but can be tested by chinging this and running the check_data notebook.

            partitioner_dict = {"train":(121, month_last),"predict":(month_last +1, month_last + 1 + step)}  # is it even meaningful to have a predict partition for forecasting? if not you can remove steps

        print('partitioner_dict', partitioner_dict) 

        return partitioner_dict


    partitioner_dict = get_partitioner_dict('forecasting')
    partition = DataPartitioner({'forecasting': partitioner_dict})
    dataset = pd.read_parquet('../raw_forecasting.parquet')
    dataset_f = dataset[dataset.index.get_level_values(0).isin(range(partitioner_dict['train'][0], partitioner_dict['train'][1] + 1))]
    stepshift_def = StepshiftedModels(LGBMRegressor(), [*range(1, 36 + 1, 1)], 'ln_ged_sb_dep')
    stepshift_model = ViewsRun(partition, stepshift_def)
    stepshift_model.fit('forecasting', "train", dataset_f)
    pred = stepshift_model.future_point_predict(partitioner_dict['predict'][0]-1, dataset, keep_specific=True)
