NGPU=${NGPU:-"1"}
LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"/nfs/model_config/llama3_1.5b.toml"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/nfs/llama1.5_ckpt/step-86949/"}

torchrun --standalone \
	--nproc_per_node="${NGPU}" \
	--local-ranks-filter="${LOG_RANK}" \
	-m scripts.benchmark.gsm8k \
	--config="${CONFIG_FILE}" \
	--checkpoint="${CHECKPOINT_DIR}"