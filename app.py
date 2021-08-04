#!/usr/bin/env python3
# project template
# Copyright(C) 2010 Red Hat, Inc.
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""This is the main script of the solver-error-classifier."""
from pandas.core.frame import DataFrame
from template.version import __version__
import logging
import random
from typing import List
from typing import Optional
from typing import Tuple
import pathlib
import pickle

import numpy as np
import pandas as pd
import json
from pandas import json_normalize
from nltk.corpus import stopwords
import click
from thoth.common import init_logging
from thoth.common import __version__ as thoth_common_version
from thoth.storages import __version__ as thoth_storages_version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster

init_logging()
_LOGGER = logging.getLogger("thoth.solver_project_url")
_DATE_FORMAT = "%Y-%m-%d"

__component_version__ = f"{__version__}+" f"storages.{thoth_storages_version}.common.{thoth_common_version}"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def split_text(text: str) -> List:
    """Split/Tokenize text by new line character."""
    new_text = text.split("\n")
    return new_text


def get_error(text: str) -> str:
    """Extract error from command log."""
    value = "Command exited with non-zero status code (1):"
    if value in text:
        text = text.replace(value, "")
    text = " ".join(text.split())
    return text


def process_exit_status_error(text: str) -> str:
    """Process and simplify error exit status."""
    if "ERROR:" in text:
        text = text.replace("ERROR: ", "")
    if "Command errored out with exit status 1:" in text:
        text = text.replace("Command errored out with exit status 1:", "exitStatus1")
    if "Could not find a version that satisfies the requirement" in text:
        text = text.replace("Could not find a version that satisfies the requirement", "CouldNotFindVersionRequirement")
    if "Failed to successfully execute function" in text:
        text = text.replace(
            "Failed to successfully execute function in Python interpreter", "FailedToExcecuteFunction/Module"
        )
    if " (from versions:" in text:
        text_arr = text.split(" (from versions:")
        if len(text_arr) == 2:
            text = text_arr[0]
    return text


def process_error_info(text: str) -> str:
    """Process and simplify error info."""
    if isinstance(text, str):
        text = " ".join(text.split())
        if "ERROR:" in text:
            text = text.replace("ERROR: ", "")
        if " [Errno 2] No such file or directory: " in text:
            text = text.replace(" [Errno 2] No such file or directory: ", "DirectoryNotFound: ")
        if "NameError:" in text:
            text = text.replace(" name ", " ")
            text = text.replace(" is not defined", "")
        if "ModuleNotFoundError:" in text:
            text = text.replace("No module named ", "")
        if "Check the logs" in text:
            text = text.replace("Check the logs for full command output.", "CheckTheLogs")
        if "No matching distribution found" in text:
            text = text.replace("No matching distribution found", "NoMatchingDistributionError")
    return text


def check_is_instance(text: str) -> str:
    """Check if text is a string instance."""
    if not isinstance(text, str):
        text = ""
    return text


# Stopwords
def remove_stopwords(text: str) -> str:
    """Remove stopwords from text, stopwords set from nltk."""
    text_arr = text.split()
    words = [w for w in text_arr if w not in stopwords.words("english")]
    return " ".join(words)


def preprocess_document(solver_path: str) -> DataFrame:
    """Preprocess document with all helperfunctions listed above to output dataframe with error_processed column."""
    error_documents = []
    for path in pathlib.Path(solver_path).iterdir():
        if path.is_file():
            document_id = str(path)[8:]
            _LOGGER.debug("Processing solver document %r", document_id)
            current_file = open(path, "r")
            data = json.load(current_file)
            if data["result"]["errors"] != []:
                error_documents.append(data)
            current_file.close()
    random.shuffle(error_documents)
    _LOGGER.debug("number of docs: %r", len(error_documents))
    df = json_normalize(error_documents, sep="_")

    df_errors = pd.DataFrame()
    df_errors["result_errors"] = df["result_errors"]
    df_errors["message"] = df_errors.apply(lambda row: row.result_errors[0]["details"]["message"], axis=1)
    df_errors["tokenized_message"] = df_errors.apply(lambda row: split_text(row.message), axis=1)
    df_errors["exit_status"] = df_errors.apply(lambda row: row.tokenized_message[0], axis=1)
    df_errors["exit_status_ERROR"] = df_errors.apply(lambda row: get_error(row.exit_status), axis=1)
    df_errors["command"] = df_errors["tokenized_message"].str[1]
    df_errors["ERROR"] = df_errors["tokenized_message"].str[-2]
    df_errors["error_info"] = df_errors["tokenized_message"].str[-4]
    df_errors["error_label"] = df_errors["tokenized_message"].str[-5]
    df_errors["cwd_info"] = df_errors["tokenized_message"].str[2:-6]
    df_errors["error_for_analysis"] = df_errors["error_info"] + df_errors["error_label"]

    df_errors["exit_status_ERROR_processed"] = df_errors.apply(
        lambda row: process_exit_status_error(row.exit_status_ERROR), axis=1
    )
    df_errors["error_info_processed"] = df_errors.apply(lambda row: process_error_info(row.error_info), axis=1)
    df_errors["message_processed"] = df_errors["exit_status_ERROR_processed"] + " " + df_errors["error_info_processed"]
    df_errors["message_processed"] = df_errors.apply(lambda row: check_is_instance(row.message_processed), axis=1)
    df_errors["message_processed"] = df_errors.apply(lambda row: remove_stopwords(row.message_processed), axis=1)
    return df_errors


def cluster_errors(*, solver_path: str) -> Tuple[KNeighborsClassifier, TfidfVectorizer]:
    """Create model and train it on solver errors, model can then classify errors from solver."""
    df_errors = preprocess_document(solver_path)
    vectorizer = TfidfVectorizer(stop_words={"english"})
    x = vectorizer.fit_transform(df_errors["message_processed"])
    x_arr = x.toarray()
    z = linkage(x_arr, "ward")
    max_d = 3
    clusters = fcluster(z, max_d, criterion="distance")
    _LOGGER.debug("Clusters: %r", set(clusters))

    def get_data_from_cluster(df, clusters, cluster_number):
        indices = [i for i, x in enumerate(clusters) if x == cluster_number]
        df_grouped = df.iloc[indices]
        return df_grouped

    for i in range(1, len(set(clusters)) + 1):
        _LOGGER.debug("Cluster %r", i)
        _LOGGER.debug(get_data_from_cluster(df_errors, clusters, i))

    knn = KNeighborsClassifier()

    knn.fit(x_arr, clusters)
    return knn, vectorizer


def classification(*, model: KNeighborsClassifier, vectorizer: TfidfVectorizer, classify_dataset: str) -> List:
    """Classify new data at classify_dataset with knn model."""
    df_errors = preprocess_document(classify_dataset)
    x = vectorizer.transform(df_errors["message_processed"])
    x_arr = x.toarray()
    result = model.predict(x_arr)
    return result


def _print_version(ctx: click.Context, _, value: str):
    """Print version and exit."""
    if not value or ctx.resilient_parsing:
        return

    click.echo(__component_version__)
    ctx.exit()


@click.command()
@click.pass_context
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    envvar="THOTH_SOLVER_ERROR_CLASSIFIER_DEBUG",
    help="Be verbose about what's going on.",
)
@click.option(
    "--version",
    is_flag=True,
    is_eager=True,
    callback=_print_version,
    expose_value=False,
    help="Print version and exit.",
)
@click.option(
    "--classify",
    is_flag=True,
    envvar="THOTH_SOLVER_ERROR_CLASSIFIER_CLASSIFY",
    help="Classifying data from model file.",
)
@click.option(
    "--train",
    is_flag=True,
    envvar="THOTH_SOLVER_ERROR_CLASSIFIER_TRAIN",
    help="Training model based on solver document file.",
)
@click.option(
    "--model-path",
    help="path to the model file if classifying.",
    metavar="FILE",
    envvar="THOTH_SOLVER_ERROR_CLASSIFIER_MODEL_PATH",
    type=str,
)
@click.option(
    "--vectorizer-path",
    help="path to the tfidf vectorizer to preprocess words with the model.",
    metavar="FILE",
    envvar="THOTH_SOLVER_ERROR_CLASSIFIER_VECTORIZER_PATH",
    type=str,
)
@click.option(
    "--train-dataset",
    help="path to the solver dataset to cluster on.",
    metavar="FOLDER",
    envvar="THOTH_SOLVER_ERROR_CLASSIFIER_TRAIN_DATASET",
    type=str,
)
@click.option(
    "--predict-dataset",
    help="path to the solver dataset to predict error.",
    metavar="FOLDER",
    envvar="THOTH_SOLVER_ERROR_CLASSIFIER_CLASSIFY_DATASET",
    type=str,
)
@click.option(
    "--output",
    help="Store result to a file.",
    metavar="FILE",
    envvar="THOTH_SOLVER_ERROR_CLASSIFIER_OUTPUT",
    type=str,
)
def cli(
    _: click.Context,
    verbose: bool = False,
    classify: bool = False,
    train: bool = False,
    model_path: Optional[str] = None,
    vectorizer_path: Optional[str] = None,
    train_dataset: Optional[str] = None,
    predict_dataset: Optional[str] = None,
    output: str = None,
):
    """Aggregate Github URLs for GitHub hosted projects on PyPI."""
    if verbose:
        _LOGGER.setLevel(logging.DEBUG)

    _LOGGER.debug("Debug mode is on")
    _LOGGER.info("Version: %s", __component_version__)

    if classify:
        if train:
            _LOGGER.error("Only one of either train or classify can be selected.")
            return
        if model_path:
            if vectorizer_path:
                model = pickle.load((open(model_path, "rb")))
                vectorizer = pickle.load(open(vectorizer_path, "rb"))
                if output:
                    if predict_dataset:
                        _LOGGER.debug(
                            classification(model=model, vectorizer=vectorizer, classify_dataset=predict_dataset)
                        )
                else:
                    _LOGGER.error("No output file given.")
            else:
                _LOGGER.error("No vectorizer path given.")
        else:
            _LOGGER.error("No model path given.")
    elif train:
        if train_dataset:
            if output:
                model, vectorizer = cluster_errors(solver_path=train_dataset)
                knn_pickle = open(output, "wb")
                vectorizer_pickle = open("vectorizer_" + output, "wb")
                pickle.dump(model, knn_pickle)
                pickle.dump(vectorizer, vectorizer_pickle)
            else:
                _LOGGER.error("No output file given.")
        else:
            _LOGGER.error("No dataset path given.")


__name__ == "__main__" and cli()
