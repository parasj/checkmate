import logging
import os
import subprocess
import sys

from colorlog import colorlog


SCRIPT_ROOT = os.path.dirname(os.path.realpath(__file__))


def get_logger():
    _log = logging.getLogger("git_sync")
    _log.propagate = False  # due to bug with duplicate outputs
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    _log.setLevel(logging.DEBUG)
    handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(levelname)s:%(name)s:\t%(message)s'))
    _log.addHandler(handler)
    _log.info("Loaded logger!")
    return _log


def run_shell(cmd: str):
    cmd_result = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout
    with cmd_result as f:
        return f.read().decode()


if __name__ == "__main__":
    logger = get_logger()
    current_branch = run_shell("git rev-parse --abbrev-ref HEAD")
    logger.info(f"Current branch on this node: {current_branch}")

    if os.system("test -z \"$(git status --porcelain)\"") > 0:
        logger.error("You have dirty changes in your path! exiting...")
        sys.exit(1)

    with open(os.path.join(SCRIPT_ROOT, 'machines.txt')) as f:
        machine_list = list(filter(len, f.read().split('\n')))

    for machine in machine_list:
        logger.info(f'Syncing with node {machine}')
        run_shell(f"ssh {machine} \"/data/paras/optimalcheckpointing/scripts/git_sync_local.sh {current_branch}\"")
