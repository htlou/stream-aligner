if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&2
	exit 1
fi

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
SCRIPT_NAME=$(basename "$0")
SCRIPT_NAME_WITHOUT_EXTENSION="${SCRIPT_NAME%.sh}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"
INPUT_PATH=""
OUTPUT_PATH=""
OUTPUT_FOLDER=""
OUTPUT_NAME=""
MODEL=""
END_NAME="correction.json"
PLATFORM="openai"
TYPE=""
declare -a INPUT_PATHS

while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
		--input-file)
            INPUT_PATHS+=("$1")  # Add the file path to the array
            shift
            ;;
		--folder-name)
			folder_names=("")
			shift
			;;
		*)
			echo "Unknown parameter passed: '${arg}'" >&2
			exit 1
			;;
	esac
done


MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
	comm -23 \
		<(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
		<(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
		shuf | head -n 1
)"

for input_file in "${INPUT_PATHS[@]}"; do
    search_dir=${SCRIPT_DIR}/${input_file}
    find "$search_dir" -type f -name "${END_NAME}" | while read file_path; do
        python3 main.py --debug \
            --openai-api-key-file ${SCRIPT_DIR}/config/openai_api_keys.txt \
            --input-file ${file_path} \
            --output-dir ${file_path}.output.json \
            --cache-dir ${SCRIPT_DIR}/.cache/${input_file}${SCRIPT_NAME_WITHOUT_EXTENSION} \
            --num-workers 30 \
            --type $TYPE \
            --shuffle 
    done
done