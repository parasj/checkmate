export TF_FORCE_GPU_ALLOW_GROWTH=true

MODEL_NAMES=("vgg_unet", "vgg_pspnet", "fcn_8_vgg", "Xception", "fcn_8", "pspnet", "segnet", "resnet50_segnet", "unet")
MODEL_NAMES=( "ResNet50" "VGG16" "InceptionV3" "DenseNet121" "ResNet101" "MobileNet" "VGG19" )
PLATFORM=${1:-'platform'}
RUNS=${2:-16}

for EXP in {0..5}
do
    for NAME in ${MODEL_NAMES[@]}
    do
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
    done
for EXP in {6..10}
do
    for NAME in $MODEL_NAMES
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
