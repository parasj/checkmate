export TF_FORCE_GPU_ALLOW_GROWTH=true


MODEL_NAMES=( "ResNet50" "VGG16" "InceptionV3" "DenseNet121" "ResNet101" "MobileNet" "VGG19" )
PLATFORM=${1:-'platform'}
RUNS=${2:-16}

for NAME in ${MODEL_NAMES[@]}
do
    for EXP in {0..10}
    do
        BATCH_SIZE=$((2**EXP))
        COST_FILE_NAME="${NAME}_${BATCH_SIZE}_${PLATFORM}"
        python src/profile_keras.py \
            -n $NAME \
            -b $BATCH_SIZE \
            -f "profiles/$NAME/"\
            -o $COST_FILE_NAME\
            -c $RUNS &> "profiles/debug/$NAME/$COST_FILE_NAME.log"

        done
    done
