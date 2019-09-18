BATCH_SIZE=128
NAME='ResNet50'
PLATFORM='p32xlarge'
COST_FILE_NAME="${NAME}_${BATCH_SIZE}_${PLATFORM}" 

mkdir "profiles/$NAME/"

python src/profile_keras.py \
    -n $NAME \
    -b $BATCH_SIZE \
    -f "profiles/$NAME/"\
    -o $COST_FILE_NAME\

#python evaluation.py \
#    --model-name $NAME \
#    -b $BATCH_SIZE \
#    --cost-file "$PLATFORM/$COST_FILE_NAME.npy" 
#
#
