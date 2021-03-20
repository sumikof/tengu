import sys
from logging import getLogger

logger = getLogger(__name__)


def osx_queue(ctx):
    from tengu.drlfx.modules.multi_queue import MultiQueue
    return MultiQueue(ctx=ctx)  # OSX ではエラーが発生するために自作クラスに差し替え


def queue_factory(ctx):
    import platform
    pf = platform.system()

    if pf == 'Darwin':
        import multiprocessing as mp
        return mp.Queue()
    else:
        import multiprocessing as mp
        return mp.Queue()
