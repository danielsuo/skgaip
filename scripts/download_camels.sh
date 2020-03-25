#!/usr/bin/env bash

# Download and expand data
CAMELS=basin_timeseries_v1p2_metForcing_obsFlow.zip
CAMELS_ATTRIBUTES=camels_attributes_v2.0.zip
MAURER_EXTENDED=maurer_extended.zip

mkdir -p data
pushd data
wget https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/$CAMELS
wget https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/$CAMELS_ATTRIBUTES
wget https://www.hydroshare.org/resource/17c896843cf940339c3c3496d0c1c077/data/contents/$MAURER_EXTENDED

unzip $CAMELS
unzip $CAMELS_ATTRIBUTES
unzip $MAURER_EXTENDED

rm *.zip
rm __MACOSX

$CAMELS_ROOT=basin_dataset_public_v1p2

mv camels_attributes_v2 $CAMELS_ROOT
mv maurer_extended $CAMELS_ROOT/basin_mean_forcing

popd data

# Create environment / update / install correct version of pytorch cuda

# train model

