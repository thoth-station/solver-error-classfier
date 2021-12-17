#!/usr/bin/env sh
#
# This script is run by OpenShift's s2i. Here we guarantee that we run desired
# sub-command based on env-variables configuration.
#

case $THOTH_SOLVER_ERROR_CLASSIFIER_CLASSIFY in
    'classify')
        exec /opt/app-root/bin/python3 app.py classify
        ;;
    'train')
        exec /opt/app-root/bin/python3 app.py train
        ;;
    *)
        echo "Application configuration error - no solver-error-classifier-job subcommand specified." >&2
        exit 1
        ;;
esac
