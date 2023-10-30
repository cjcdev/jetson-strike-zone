#!/bin/bash

TEST_DIR="test_data"

INPUT="csi://0"
# INPUT="--input-loop=-1 ${TEST_DIR}/batting.mp4"
# INPUT="${TEST_DIR}/batting-pose.png"
# INPUT="${TEST_DIR}/standing-pose.png"

OUTPUT="rtp://chris-lin-xps15:8554"
# OUTPUT="out.jpg"

#EXEC_BIN="/jetson-inference/build/aarch64/bin/posenet"
EXEC_BIN="./strike_zone.py"

${EXEC_BIN} --headless "${INPUT}" "${OUTPUT}"

