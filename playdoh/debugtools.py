from userpref import *
import logging
import os.path
import time
import traceback
import sys


def setup_logging(level):
    if level == logging.DEBUG:
        logging.basicConfig(level=level,
                            format="%(asctime)s,%(msecs)03d  \
%(levelname)-7s  P:%(process)-4d  \
T:%(thread)-4d  %(message)s",
                            datefmt='%H:%M:%S')
    else:
        logging.basicConfig(level=level,
                            stream=sys.stdout,
                            format="%(message)s")
    return logging.getLogger('playdoh')


def get_caller():
    tb = traceback.extract_stack()[-3]
    module = os.path.splitext(os.path.basename(tb[0]))[0].ljust(18)
    line = str(tb[1]).ljust(4)
#    func = tb[2].ljust(18)
#    return "L:%s  %s  %s" % (line, module, func)
    return "L:%s  %s" % (line, module)


def log_debug(obj):
    # HACK: bug fix for deadlocks when logger level is not debug
    time.sleep(.002)
    if level == logging.DEBUG:
        string = str(obj)
        string = get_caller() + string
        logger.debug(string)


def log_info(obj):
    if level == logging.DEBUG:
        obj = get_caller() + str(obj)
    logger.info(obj)


def log_warn(obj):
    if level == logging.DEBUG:
        obj = get_caller() + str(obj)
    logger.warn(obj)


def debug_level():
    logger.setLevel(logging.DEBUG)


def info_level():
    logger.setLevel(logging.INFO)


def warning_level():
    logger.setLevel(logging.WARNING)


# Set logging level to INFO by default
level = eval('logging.%s' % USERPREF['loglevel'])
logger = setup_logging(level)


__all__ = ['log_debug', 'log_info', 'log_warn',
           #'logging', 'setup_logging', 'logger',
           'debug_level', 'info_level', 'warning_level']


if __name__ == '__main__':
    log_debug("hello world")
    log_info("hello world")
