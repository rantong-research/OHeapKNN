#!/bin/bash

# 实验标签，例如 LAN、RTT5ms、RTT10ms
TAG=${1:-LAN}

# 自动创建对应实验结果目录
OUT_DIR="./KNN-experiment-res-${TAG}"
mkdir -p "$OUT_DIR"

# 需要运行的协议；默认只跑 OHeapKNN
# 如需运行其他版本，可改为：
# PROTOCOLS=("OHeapKNN" "OHeapKNN_b" "Heap_based")
PROTOCOLS=("OHeapKNN")

echo "======================================"
echo "开始 OHeapKNN 实验"
echo "实验组: ${TAG}"
echo "结果目录: ${OUT_DIR}"
echo "======================================"

# 第一步：清理并编译
make clean

for PROTOCOL in "${PROTOCOLS[@]}"; do
    echo "开始编译 ${PROTOCOL}..."

    if ! make -j8 "${PROTOCOL}.x" > /dev/null 2>&1; then
        echo "${PROTOCOL} 编译失败，退出脚本。"
        exit 1
    fi

    echo "${PROTOCOL} 编译完成！"
done

# 第二步：依次运行协议
for PROTOCOL in "${PROTOCOLS[@]}"; do
    echo "======================================"
    echo "开始执行 ${PROTOCOL}..."
    echo "======================================"

    "./${PROTOCOL}.x" 0 -pn 10000 -h localhost \
        > "${OUT_DIR}/outputP0_${PROTOCOL}.log" 2>&1 &
    PID0=$!

    "./${PROTOCOL}.x" 1 -pn 10000 -h localhost \
        > "${OUT_DIR}/outputP1_${PROTOCOL}.log" 2>&1 &
    PID1=$!

    # 等待双方计算进程结束
    STATUS=0

    wait "$PID0" || STATUS=1
    wait "$PID1" || STATUS=1

    if [ "$STATUS" -ne 0 ]; then
        echo "${PROTOCOL} 执行过程中存在错误，请检查日志。"
        exit 1
    fi

    echo "${PROTOCOL} 的所有进程执行完毕！"
done

echo "======================================"
echo "${PROTOCOLS[*]} 实验全部完成！"
echo "实验结果保存在: ${OUT_DIR}"
echo "======================================"
