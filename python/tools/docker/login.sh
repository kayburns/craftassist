#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates.


$(aws ecr get-login --region us-east-2 | sed 's|-e none https://||')
