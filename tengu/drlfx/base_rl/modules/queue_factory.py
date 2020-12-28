def osx_queue(ctx):
    from tengu.drlfx.base_rl.modules.multi_queue import MultiQueue
    return MultiQueue(ctx=ctx)  # OSX ではエラーが発生するために自作クラスに差し替え


def queue_factory(ctx):
    import platform
    pf = platform.system()
    if pf == 'Darwin':
        return osx_queue(ctx)
    else:
        import multiprocessing as mp
        return mp.Queue()
