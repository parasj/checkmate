export TF_FORCE_GPU_ALLOW_GROWTH=true


MODEL_NAMES=(  "VGG16" "InceptionV3" "DenseNet121" "ResNet101" "MobileNet" "VGG19" "ResNet50")
PLATFORM=${1:-'platform'}
RUNS=${2:-16}

echo "$PLATFORM"

for NAME in ${MODEL_NAMES[@]}
do
    for EXP in {4..10}
    do
        DEVICE=$((EXP-3))
        CUDA_VISIBLE_DEVICES=$DEVICE
        BATCH_SIZE=$((2**EXP))
        COST_FILE_NAME="${NAME}_${BATCH_SIZE}_${PLATFORM}"
        python src/profile_keras.py \
            -n $NAME \
            -b $BATCH_SIZE \
            -f "profiles/$NAME/"\
            -o $COST_FILE_NAME\
            -c $RUNS &> "profiles/debug/$NAME/$COST_FILE_NAME.log" &

        done
        wait
        echo "$NAME"
    done
