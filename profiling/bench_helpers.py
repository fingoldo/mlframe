def repeatManyTimes(func,*args,nRepetitions=1000,**kwargs):
    # A monotonic clock: wall-clock datetime.now() can jump (NTP, DST) in the middle of a measurement.
    from time import perf_counter
    startTime=perf_counter()
    for i in range(nRepetitions):
        func(*args,**kwargs)
    return perf_counter()-startTime
