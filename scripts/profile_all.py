import argparse
import logging
import os
import pathlib
import sys

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "../../src")))
#from integration.tf2.extraction import get_input_shape

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# UNCOMMENT FOR FIRST PROFILING MACHINE
#BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
#MODEL_NAMES = ["vgg_unet", "vgg_pspnet", "fcn_8_vgg", "Xception", "fcn_8", "pspnet", "segnet", "resnet50_segnet", "unet"]

# UNCOMMENT FOR SECOND PROFILING MACHINE
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
MODEL_NAMES = ['ResNet50', 'VGG16', "InceptionV3", "DenseNet121", "ResNet101", 'MobileNet', 'VGG19']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model profiler")
    parser.add_argument('platform', help="Hardware platform (e.g. p2xlarge)")
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('-c', '--num-runs', type=int, default=1)
    args = parser.parse_args()

    logger.info(f"Profiling for platform {args.platform}")
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    for model_name in tqdm(MODEL_NAMES, desc="Model"):
        #input_shape = list(get_input_shape(model_name)[0])
        for batch_size in tqdm(BATCH_SIZES, desc="Batch size"):
            input_shape = [batch_size]
            input_shape_str = "-".join(map(str, input_shape))
            folder = os.path.join(root_dir, "profiles", model_name)
            debug_folder = os.path.join(root_dir, "profiles", "debug", model_name)
            cost_file = f"is{input_shape_str}_{args.platform}"
            cd_cmd = f"cd \"{root_dir}\""
            cmd = f"python src/profile_keras.py -n \"{model_name}\" -b {batch_size} -f \"{folder}\" -o \"{cost_file}\" -c {args.num_runs}"
            pipe_file = f"{debug_folder}/{cost_file}.log"
            final_cmd = f"{cd_cmd}; {cmd} > {pipe_file} 2>&1"
            if args.dry_run:
                logger.info('\n\t' + final_cmd)
            else:
                pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
                pathlib.Path(debug_folder).mkdir(parents=True, exist_ok=True)
                status = os.system(final_cmd)
                assert status != -1
