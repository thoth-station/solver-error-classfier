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

from template.version import __version__
import logging
import sys
import random
from datetime import date
from datetime import datetime
from typing import Dict
from typing import Any
from typing import List
from typing import Optional
import pathlib

import numpy as np
import pandas as pd
import json
from pandas import json_normalize
from nltk.corpus import stopwords
import nltk
import click
import yaml
from thoth.common import init_logging
from thoth.common import __version__ as thoth_common_version
from thoth.storages import SolverResultsStore
from thoth.storages import __version__ as thoth_storages_version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt

init_logging()
_LOGGER = logging.getLogger("thoth.solver_project_url")
_DATE_FORMAT = "%Y-%m-%d"

__component_version__ = f"{__version__}+" f"storages.{thoth_storages_version}.common.{thoth_common_version}"

def cluster_errors():
    """Create model and train it on solver errors, model can then classify errors from solver."""
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    error_documents = []
    for path in pathlib.Path("solvers").iterdir():
        if path.is_file():
            document_id = str(path)[8:]
            _LOGGER.debug("Processing solver document %r", document_id)
            current_file = open(path, "r")
            data = json.load(current_file)
            if data["result"]["errors"] != []:
                error_documents.append(data)
            current_file.close()
    random.shuffle(error_documents)
    _LOGGER.debug("number of docs: %r",len(error_documents))
    df = json_normalize(error_documents, sep = "_")

    def split_text(text):
        new_text = text.split('\n')
        return new_text

    def get_ERROR(text):
        value = "Command exited with non-zero status code (1):"
        if value in text:
            text = text.replace(value, "")
        text = ' '.join(text.split())
        return text

    def process_exit_status_ERROR(text):
        if "ERROR:" in text:
            text = text.replace("ERROR: ", "")
        if "Command errored out with exit status 1:" in text:
            text = text.replace("Command errored out with exit status 1:", "exitStatus1")
        if "Could not find a version that satisfies the requirement" in text:
            text = text.replace("Could not find a version that satisfies the requirement", "CouldNotFindVersionRequirement")
        if "Failed to successfully execute function" in text:
            text = text.replace("Failed to successfully execute function in Python interpreter", "FailedToExcecuteFunction/Module")
        if " (from versions:" in text:
            text = text.split(" (from versions:")
        
        if len(text) == 2:
            text = text[0]
        return text

    def process_error_info(text):
        if isinstance(text, str):
            text = ' '.join(text.split())
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

    def check_is_instance(text):
        if not isinstance(text, str):
            text = ""
        return text

    #Stopwords
    def remove_stopwords(text):
        text = text.split()
        words = [w for w in text if w not in stopwords.words('english')]
        return " ".join(words)


    df_errors = pd.DataFrame()
    df_errors['result_errors'] = df['result_errors']
    df_errors['message'] = df_errors.apply(lambda row: row.result_errors[0]['details']['message'], axis=1)
    df_errors['tokenized_message'] = df_errors.apply(lambda row: split_text(row.message),axis = 1)
    df_errors['exit_status'] = df_errors.apply(lambda row: row.tokenized_message[0],axis = 1)
    df_errors['exit_status_ERROR'] = df_errors.apply(lambda row: get_ERROR(row.exit_status), axis = 1)
    df_errors['command'] = df_errors["tokenized_message"].str[1]
    df_errors['ERROR'] = df_errors["tokenized_message"].str[-2]
    df_errors['error_info'] = df_errors["tokenized_message"].str[-4]
    df_errors['error_label'] = df_errors["tokenized_message"].str[-5]
    df_errors['cwd_info'] = df_errors["tokenized_message"].str[2:-6]
    df_errors['error_for_analysis'] = df_errors['error_info'] + df_errors['error_label']

    df_errors['exit_status_ERROR_processed'] = df_errors.apply(lambda row: process_exit_status_ERROR(row.exit_status_ERROR),axis = 1)
    df_errors['error_info_processed'] = df_errors.apply(lambda row: process_error_info(row.error_info),axis = 1)
    df_errors["message_processed"] = df_errors["exit_status_ERROR_processed"] + " " + df_errors["error_info_processed"]
    df_errors["message_processed"] = df_errors.apply(lambda row: check_is_instance(row.message_processed),axis = 1)
    
    vectorizer = TfidfVectorizer(stop_words={'english'})
    X = vectorizer.fit_transform(df_errors['message_processed'])
    x_arr = X.toarray()
    Z = linkage(x_arr, 'ward')
    c, coph_dists = cophenet(Z, pdist(x_arr))
    max_d = 3
    clusters = fcluster(Z, max_d, criterion='distance')
    _LOGGER.debug("Clusters: %r",set(clusters))
    
    def get_data_from_cluster(df, clusters, cluster_number):
        indices = [i for i, x in enumerate(clusters) if x == cluster_number]
        df_grouped = df.iloc[indices]
        return df_grouped

    for i in range(1,len(set(clusters))+1):
        _LOGGER.debug("Cluster %r", i)
        _LOGGER.debug(get_data_from_cluster(df_errors, clusters, i))


    
    def add_clusters_to_frame(original_data, clusters):  # train a KNN classifier on this new labelled data
        or_frame = pd.DataFrame(data=original_data)
        or_frame_labelled = pd.concat([or_frame, pd.DataFrame(clusters)], axis=1)
        return(or_frame_labelled)
    
    supervised_df = add_clusters_to_frame(X, clusters)
    supervised_df.columns = ['message_processed_vectorized', 'labels']

    X = X.toarray()
    indices = np.random.permutation(len(X))
    X_train = X[indices[:-10]]
    y_train = clusters[indices[:-10]]
    X_test  = X[indices[-10:]]
    y_test  = clusters[indices[-10:]]

    knn = KNeighborsClassifier()

    knn.fit(X_train, y_train) 

    res = knn.predict(X_test)
    _LOGGER.debug("Predicted: %r", res)
    _LOGGER.debug("Actual: %r", y_test)
    return knn

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
    envvar="THOTH_SOLVER_ERROR_CLASSIFIER",
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
    "--output",
    help="Store result to a file or print to stdout (-).",
    metavar="FILE",
    envvar="THOTH_SOLVER_ERROR_CLASSIFIER_OUTPUT",
    type=str,
)
def cli(
    _: click.Context,
    verbose: bool = False,
    output: Optional[str] = None,
):
    """Aggregate Github URLs for GitHub hosted projects on PyPI."""
    if verbose:
        _LOGGER.setLevel(logging.DEBUG)

    _LOGGER.debug("Debug mode is on")
    _LOGGER.info("Version: %s", __component_version__)

    model = cluster_errors()

    # if output == "-" or not output:
    #     yaml.safe_dump(model, sys.stdout)
    # else:
    #     _LOGGER.info("Writing results computed to %r", output)
    #     with open(output, "w") as f:
    #         yaml.safe_dump(model, f)


__name__ == "__main__" and cli()
