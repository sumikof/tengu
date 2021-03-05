import multiprocessing.queues


class SharedCounter(object):

    def __init__(self, n=0):
        self.count = multiprocessing.Value('i', n)

    def increment(self, n=1):
        """ Increment the counter by n (default = 1) """
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self):
        """ Return the value of the counter """
        return self.count.value


class MultiQueue(multiprocessing.queues.Queue):

    def __init__(self, maxsize=0, *, ctx):
        super(MultiQueue, self).__init__(maxsize=maxsize, ctx=ctx)
        self.size = SharedCounter(0)

    def put(self, *args, **kwargs):
        self.size.increment(1)
        super(MultiQueue, self).put(*args, **kwargs)

    def get(self, *args, **kwargs):
        self.size.increment(-1)
        return super(MultiQueue, self).get(*args, **kwargs)

    def qsize(self):
        """ Reliable implementation of multiprocessing.Queue.qsize() """
        return self.size.value

    def empty(self):
        """ Reliable implementation of multiprocessing.Queue.empty() """
        return not self.qsize()

    def clear(self):
        """ Remove all elements from the Queue. """
        while not self.empty():
            self.get()


if __name__ == '__main__':
    import numpy as np

    a = [[np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])],
         [np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])]]
    a = np.reshape(a, (2, 4, 4))
    print(a)
