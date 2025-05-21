NGPU=${NGPU:-"1"}
LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/debug_model.toml"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"./outputs/checkpoint/"}
PROMPT=${PROMPT:-""}

torchrun --standalone \
	--nproc_per_node="${NGPU}" \
	--local-ranks-filter="${LOG_RANK}" \
	-m scripts.benchmark.gsm8k \
	--config="${CONFIG_FILE}" \
	--checkpoint="${CHECKPOINT_DIR}"