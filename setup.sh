#!/bin/bash

# install helpful tools not in Dockerfile
apt -y install vim

# unpack models
tar -xvf models_folder.tar.gz
mv models/semantic_parser/ttad/ python/craftassist/models/ttad
mv models/semantic_parser/ttad_bert_updated/ python/craftassist/models/ttad_bert_updated
mkdir python/craftassist/models/ttad_bert_updated/model/
mv python/craftassist/models/ttad_bert_updated/* python/craftassist/models/ttad_bert_updated/model/
curl -OL http://craftassist.s3-us-west-2.amazonaws.com/pubr/collected_dataset.tar.gz -o collected_dataset.tar.gz
tar -xzvf collected_dataset.tar.gz
