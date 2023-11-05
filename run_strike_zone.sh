#!/bin/bash

INPUT="csi://0"

OUTPUT="rtp://chris-lin-xps15:8554"
# OUTPUT="out.jpg"

EXEC_BIN="./strike_zone.py"

${EXEC_BIN} --headless "${INPUT}" "${OUTPUT}"

