#!/bin/bash

# create the train.txt/test.txt/val.txt/trainval.txt files based on default.txt

BASE_DIR=${1}
IMAGE_SETS_DIR="${BASE_DIR}/ImageSets/Main"
IMAGE_SETS_DEFAULT="${IMAGE_SETS_DIR}/default.txt"
IMAGE_SETS_TRAIN="${IMAGE_SETS_DIR}/train.txt"
IMAGE_SETS_TRAINVAL="${IMAGE_SETS_DIR}/trainval.txt"
IMAGE_SETS_TEST="${IMAGE_SETS_DIR}/test.txt"
IMAGE_SETS_VAL="${IMAGE_SETS_DIR}/val.txt"
IMAGES_DIR="${BASE_DIR}/JPEGImages"

IMAGE_NAMES=$(cat ${IMAGE_SETS_DIR}/default.txt)
IMAGE_COUNT=$(echo "${IMAGE_NAMES}" | wc -l)
VAL_COUNT=$((IMAGE_COUNT / 10))
TEST_COUNT=$((IMAGE_COUNT / 10))
TRAIN_COUNT=$((IMAGE_COUNT - VAL_COUNT - TEST_COUNT))

echo "IMAGE_COUNT: ${IMAGE_COUNT}"
echo "TRAIN_COUNT: ${TRAIN_COUNT}"
echo "TEST_COUNT COUNT: ${TEST_COUNT}"
echo "VAL_COUNT: ${VAL_COUNT}"

IMAGE_NAMES_RANOMD=$(sort -R ${IMAGE_SETS_DEFAULT})
echo $IMAGE_NAMES_RANOMD

# reset the files
rm -fr ${IMAGE_SETS_TRAIN}
rm -fr ${IMAGE_SETS_TRAINVAL}
rm -fr ${IMAGE_SETS_TEST}
rm -fr ${IMAGE_SETS_VAL}

# populate test/train/val files
i=1
for name in ${IMAGE_NAMES_RANOMD}; do
    if (( i <= VAL_COUNT )); then
        echo "${name}" >> ${IMAGE_SETS_VAL}
        echo "${name}" >> ${IMAGE_SETS_TRAINVAL}
    elif (( i > VAL_COUNT && i <= (TEST_COUNT+VAL_COUNT) )); then
        echo "${name}" >> ${IMAGE_SETS_TEST}
    else
        echo "${name}" >> ${IMAGE_SETS_TRAIN}
        echo "${name}" >> ${IMAGE_SETS_TRAINVAL}
    fi
    i=$((i+1))
done