#!/usr/bin/env bash
#
# KNN-experiment.sh
#
# Build and run the KNN protocols released in this repository.
# Results and per-run logs are written to KNN-experiment-res-<TAG>/.
#
# Usage:
#   ./KNN-experiment.sh [TAG] [PROTOCOL ...]
#
# Examples:
#   ./KNN-experiment.sh LAN
#   ./KNN-experiment.sh RTT5ms OHeapKNN
#   ./KNN-experiment.sh WAN OHeapKNN OHeapKNN_b Heap_based
#
# Network shaping is intentionally separate from this script.
# Configure the network first with the included helpers:
#   ./LAN.sh 0       # LAN
#   ./LAN.sh 5       # 5 ms RTT
#   ./WAN.sh         # WAN profile
# Then run this script with a matching TAG.
#
# Environment overrides:
#   JOBS=8 PORT=10000 HOST=localhost SKIP_CLEAN=0
#
# Run this script from the Garnet root directory containing the Makefile.

set -u
set -o pipefail

DEFAULT_PROTOCOLS=(OHeapKNN OHeapKNN_b Heap_based)
SUPPORTED_PROTOCOLS=(OHeapKNN OHeapKNN_b Heap_based)

usage() {
    cat <<'EOF'
Usage: ./KNN-experiment.sh [TAG] [PROTOCOL ...]

  TAG        Result label, e.g. LAN, RTT5ms, WAN (default: LAN)
  PROTOCOL   One or more of: OHeapKNN OHeapKNN_b Heap_based
             (default: all three)

Examples:
  ./KNN-experiment.sh LAN
  ./KNN-experiment.sh RTT5ms OHeapKNN
  ./KNN-experiment.sh WAN OHeapKNN OHeapKNN_b Heap_based

Network shaping is separate:
  ./LAN.sh 0       # LAN
  ./LAN.sh 5       # 5 ms RTT
  ./WAN.sh         # WAN

Environment overrides:
  JOBS=8 PORT=10000 HOST=localhost SKIP_CLEAN=0
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

TAG="${1:-LAN}"
if (( $# > 0 )); then
    shift
fi

if [[ ! "$TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "错误：TAG 只能包含字母、数字、点、下划线和短横线。" >&2
    exit 2
fi

if (( $# > 0 )); then
    PROTOCOLS=("$@")
else
    PROTOCOLS=("${DEFAULT_PROTOCOLS[@]}")
fi

if (( ${#PROTOCOLS[@]} == 0 )); then
    echo "错误：至少需要一个协议。" >&2
    exit 2
fi

for protocol in "${PROTOCOLS[@]}"; do
    valid=0
    for supported in "${SUPPORTED_PROTOCOLS[@]}"; do
        if [[ "$protocol" == "$supported" ]]; then
            valid=1
            break
        fi
    done
    if (( valid == 0 )); then
        echo "错误：不支持的协议 '$protocol'。" >&2
        echo "可用协议：${SUPPORTED_PROTOCOLS[*]}" >&2
        exit 2
    fi
done

OUT_DIR="./KNN-experiment-res-${TAG}"
JOBS="${JOBS:-8}"
PORT="${PORT:-10000}"
HOST="${HOST:-localhost}"
SKIP_CLEAN="${SKIP_CLEAN:-0}"

if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误：JOBS 必须为正整数。" >&2
    exit 2
fi

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( 10#$PORT < 1 || 10#$PORT > 65535 )); then
    echo "错误：PORT 必须为 1-65535 之间的整数。" >&2
    exit 2
fi

if [[ ! -f Makefile ]]; then
    echo "错误：当前目录没有 Makefile，请在 Garnet 根目录运行本脚本。" >&2
    exit 1
fi

if ! command -v make >/dev/null 2>&1; then
    echo "错误：找不到 make 命令。" >&2
    exit 1
fi

if ! mkdir -p "$OUT_DIR"; then
    echo "错误：无法创建结果目录 $OUT_DIR。" >&2
    exit 1
fi

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

PIDS=()
cleanup() {
    if (( ${#PIDS[@]} > 0 )); then
        local pid
        for pid in "${PIDS[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
    fi
}
on_signal() {
    cleanup
    exit 130
}
trap cleanup EXIT
trap on_signal INT TERM

SUMMARY="$OUT_DIR/summary.txt"
{
    echo "Experiment tag: ${TAG}"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Protocols: ${PROTOCOLS[*]}"
    echo "Host: ${HOST}"
    echo "Port: ${PORT}"
    echo "----------------------------------------"
} > "$SUMMARY"

log "======================================"
log "开始 OHeapKNN 实验"
log "实验组: ${TAG}"
log "结果目录: ${OUT_DIR}"
log "协议: ${PROTOCOLS[*]}"
log "======================================"

# Step 1: clean and build.
if [[ "$SKIP_CLEAN" == "1" ]]; then
    log "SKIP_CLEAN=1，跳过 make clean。"
else
    log "清理旧构建..."
    if ! make clean >"$OUT_DIR/build-clean.log" 2>&1; then
        log "警告：make clean 失败，继续尝试编译（日志：$OUT_DIR/build-clean.log）。"
    fi
fi

for protocol in "${PROTOCOLS[@]}"; do
    log "开始编译 ${protocol}..."
    build_log="$OUT_DIR/build_${protocol}.log"

    if ! make -j"$JOBS" "${protocol}.x" >"$build_log" 2>&1; then
        log "错误：${protocol} 编译失败。"
        log "日志：${build_log}"
        tail -n 40 "$build_log" || true
        echo "BUILD FAILED ${protocol}" >> "$SUMMARY"
        exit 1
    fi

    if [[ ! -x "./${protocol}.x" ]]; then
        log "错误：编译完成但找不到可执行文件 ./${protocol}.x"
        echo "MISSING BINARY ${protocol}" >> "$SUMMARY"
        exit 1
    fi

    log "${protocol} 编译完成。"
    echo "BUILD OK ${protocol}" >> "$SUMMARY"
done

# Step 2: run both parties and wait for them.
overall_status=0

for protocol in "${PROTOCOLS[@]}"; do
    log "======================================"
    log "开始执行 ${protocol}..."
    log "======================================"

    p0_log="${OUT_DIR}/outputP0_${protocol}.log"
    p1_log="${OUT_DIR}/outputP1_${protocol}.log"
    : > "$p0_log"
    : > "$p1_log"

    "./${protocol}.x" 0 -pn "$PORT" -h "$HOST" >"$p0_log" 2>&1 &
    pid0=$!
    PIDS+=("$pid0")

    "./${protocol}.x" 1 -pn "$PORT" -h "$HOST" >"$p1_log" 2>&1 &
    pid1=$!
    PIDS+=("$pid1")

    log "${protocol}: P0 PID=${pid0}, P1 PID=${pid1}"

    status=0
    wait "$pid0" || status=1
    wait "$pid1" || status=1
    PIDS=()

    if (( status != 0 )); then
        log "错误：${protocol} 执行失败。"
        log "请检查日志："
        log "  ${p0_log}"
        log "  ${p1_log}"
        echo "RUN FAILED ${protocol}" >> "$SUMMARY"
        overall_status=1
        break
    fi

    log "${protocol} 执行完毕。"
    echo "RUN OK ${protocol}" >> "$SUMMARY"
done

{
    echo "----------------------------------------"
    echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
    if (( overall_status == 0 )); then
        echo "Overall: SUCCESS"
    else
        echo "Overall: FAILED"
    fi
} >> "$SUMMARY"

if (( overall_status != 0 )); then
    log "实验未全部成功，请检查日志：${OUT_DIR}"
    exit 1
fi

log "======================================"
log "实验全部完成！"
log "协议: ${PROTOCOLS[*]}"
log "实验结果保存在: ${OUT_DIR}"
log "======================================"
