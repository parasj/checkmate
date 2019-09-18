# export TF_FORCE_GPU_ALLOW_GROWTH=true

PLATFORM="p32xlarge"
RUNS=8000
MEMS=(100 1000  2000)
for MEM in ${MEMS[@]}
do
    SIZES=(64 128 256 512)
    STRATS=("GRIEWANK_LOGN" "CHEN_SQRTN_NOAP" "CHEN_GREEDY_NOAP" "OPTIMAL_ILP_GC" "CHECKPOINT_ALL")
    MODEL_NAMES=("VGG16" "MobileNet") #"vgg_unet")
    for NAME in ${MODEL_NAMES[@]} 
    do
        for BATCH_SIZE in ${SIZES[@]}
        do
            for STRAT in ${STRATS[@]}
            do
                echo "python src/execute_one.py --model-name $NAME \
--model-version v1 \
--platform $PLATFORM \
--num-runs $RUNS \
-b $BATCH_SIZE \
--strategy $STRAT \
--buffer-mem-mb $MEM &> logs/\"${NAME}_${BATCH_SIZE}_${STRAT}_${MEM}_gradless_eagerfalse.log\""

                python src/execute_one.py --model-name $NAME \
                    --model-version v1 \
                    --platform $PLATFORM \
                    --num-runs $RUNS \
                    -b $BATCH_SIZE \
                    --strategy $STRAT \
                    --buffer-mem-mb $MEM &> logs/"${NAME}_${BATCH_SIZE}_${STRAT}_${MEM}_gradless_eagerfalse.log"
            done
        done
    done



    STRATS=("GRIEWANK_LOGN" "CHEN_SQRTN_NOAP" "CHEN_GREEDY_NOAP" "OPTIMAL_ILP_GC" "CHECKPOINT_ALL" "CHEN_SQRTN" "CHEN_GREEDY")
    MODEL_NAMES=( "ResNet50" )
    for NAME in ${MODEL_NAMES[@]} 
    do
        for BATCH_SIZE in ${SIZES[@]}
        do
            for STRAT in ${STRATS[@]}
            do
                echo "python src/execute_one.py --model-name $NAME \
--model-version v1 \
--platform $PLATFORM \
--num-runs $RUNS \
-b $BATCH_SIZE \
--strategy $STRAT \
--buffer-mem-mb $MEM &> logs/\"${NAME}_${BATCH_SIZE}_${STRAT}_${MEM}_gradless_eagerfalse.log\""

                python src/execute_one.py --model-name $NAME \
                    --model-version v1 \
                    --platform $PLATFORM \
                    --num-runs $RUNS \
                    -b $BATCH_SIZE \
                    --strategy $STRAT \
                    --buffer-mem-mb $MEM &> logs/"${NAME}_${BATCH_SIZE}_${STRAT}_${MEM}_gradless_eagerfalse.log"
            done
        done
    done



    SIZES=(8 16 32 64 128 256)
    MODEL_NAMES=( "vgg_unet" )
    for NAME in ${MODEL_NAMES[@]} 
    do
        for BATCH_SIZE in ${SIZES[@]}
        do
            for STRAT in ${STRATS[@]}
            do
                echo "python src/execute_one.py --model-name $NAME \
--model-version v1 \
--platform $PLATFORM \
--num-runs $RUNS \
-b $BATCH_SIZE \
--strategy $STRAT \
--buffer-mem-mb $MEM &> logs/\"${NAME}_${BATCH_SIZE}_${STRAT}_${MEM}_gradless_eagerfalse.log\""

                python src/execute_one.py --model-name $NAME \
                    --model-version v1 \
                    --platform $PLATFORM \
                    --num-runs $RUNS \
                    -b $BATCH_SIZE \
                    --strategy $STRAT \
                    --buffer-mem-mb $MEM &> logs/"${NAME}_${BATCH_SIZE}_${STRAT}_${MEM}_gradless_eagerfalse.log"
            done
        done
    done
done
