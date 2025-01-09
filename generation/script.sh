#!/bin/bash
if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&2
	exit 1
fi

set -e  # 当任何一行命令执行失败时，退出脚本
set -x

FILE=""
INPUT_PATH=""
BASE_MODEL=""
ALIGNER_MODEL=""

while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
		--input-file)
			INPUT_PATH=$1
			shift
			;;
        --target-file)
            FILE=$1
            shift
            ;;
        --base-model)
            BASE_MODEL=$1
            shift
            ;;
        --aligner-model)
            ALIGNER_MODEL=$1
            shift
            ;;
		*)
			echo "Unknown parameter passed: '${arg}'" >&2
			exit 1
			;;
	esac
done

if [ -z "${INPUT_PATH}" ]; then
    echo "Please specify the input file path with --input-file." >&2
    exit 1
fi

# first load dataset
mkdir -p data/$FILE
python load_dataset.py \
    --input_file "${INPUT_PATH}" \
    --file ${FILE}

for i in {0..15}
do
    echo "Round ${i} start"
    if [ ! -f "./data/$FILE/r${i}_base.json" ]; then
        echo "base data not found, generating..."
        { time python generation.py \
            --model_name_or_path ${BASE_MODEL} \
            --round ${i} \
            --file ${FILE} \
            --model_type "llama3" \
            2>&1 1>&3 | tee -a generation_timing.log; } 3>&1
    fi
    # if [ ! -f "./data/$FILE/r${i}_tf.json" ]; then
    #     echo "tf data not found, generating..."
    python tf_annotate.py \
        --round ${i} \
        --file ${FILE}
    # fi
    if [ ! -f "./data/$FILE/r${i}_annotated.json" ]; then
        export CONFIG_SET=1
        echo "annotated data not found, generating with ${ALIGNER_MODEL}..."
        { time python annotation.py \
            --model_name_or_path ${ALIGNER_MODEL} \
            --round ${i} \
            --file ${FILE} \
            --model_type "gemma" \
            2>&1 1>&3 | tee -a generation_timing.log; } 3>&1
        # bash gpt4/script.sh \
        # --input-file "../data/r${i}_tf.json" \
        # --round ${i}
    fi
    python tf_input.py \
        --round ${i} \
        --file ${FILE}
done