import logging
import colorlog


def setup_logger(name: str, log_file: str = None):
    logging._warn_preinit_stderr = 0
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # due to bug with duplicate outputs

    if not len(logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(levelname)s:%(name)s:\t%(message)s'))
        logger.addHandler(handler)

        if log_file is not None:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)

    return logger
