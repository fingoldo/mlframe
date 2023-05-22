def repeatManyTimes(func,*args,nRepetitions=1000,**kwargs):
    from datetime import datetime
    startTime=datetime.now()
    for i in range(nRepetitions):
        func(*args,**kwargs)
    endTime=datetime.now()
    return (endTime-startTime).total_seconds()