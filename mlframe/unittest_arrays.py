import unittest
import numpy as np
import mlframe.arrays as m
import mlframe.tests as tests

minElem=50;maxElem=1000;arrSize=1000000

def baselineArgSort(vals):
    return np.argsort(vals)
def baselineArgSortIndexed(vals,indices):
    fr=vals[indices]
    return indices[np.argsort(fr)]
            
class Test(unittest.TestCase):    
    def test_arrayMinMax(self):
        #Validity
        self.assertEqual(m.arrayMinMax(np.random.randint(minElem,maxElem,arrSize)),(minElem, maxElem-1))
        self.assertEqual(m.arrayMinMax(np.arange(20),10,15), (10,14))
    def test_arrayMinMaxParallel(self):
        #Validity
        self.assertEqual(m.arrayMinMaxParallel(np.random.randint(minElem,maxElem,arrSize)), (minElem, maxElem-1))
        self.assertEqual(m.arrayMinMaxParallel(np.arange(20),10,15), (10,14))
        #Performance
        x = np.random.rand(arrSize)
        def baselineMinMax(x):
            return x.min(), x.max()

        baselineTime=tests.repeatManyTimes(baselineMinMax,x)    
        print("Baseline x.min(), x.max() test took %s" % baselineTime)
        
        baselineTime=tests.repeatManyTimes(m.arrayMinMax,x)    
        print("Baseline arrayMinMax test took %s" % baselineTime)        
        
        optimizedTime=tests.repeatManyTimes(m.arrayMinMaxParallel,x,maxThreads=2)
        print("Optimized arrayMinMaxParallel test took %s" % optimizedTime)
        
        #self.assertTrue(optimizedTime<baselineTime)
    def test_arrayCountingSort(self):
        x=np.random.randint(minElem,maxElem,arrSize)
        #Validity        
        self.assertTrue((m.arrayCountingSort(x,maxElem)==np.sort(x)).all())
        #Performance
        baselineTime=tests.repeatManyTimes(np.sort,x)    
        print("Baseline np.sort test took %s" % baselineTime)
        
        optimizedTime=tests.repeatManyTimes(m.arrayCountingSort,x,maxElem)
        print("Optimized arrayCountingSort test took %s" % optimizedTime)
        self.assertTrue(optimizedTime<baselineTime)
    def test_arrayCountingArgSort(self):
        x=np.random.randint(minElem,maxElem,arrSize)
        #Validity
        #    whole array
        self.assertTrue((x[m.arrayCountingArgSort(x,maxElem)]==x[np.argsort(x)]).all())
        #    indexed array
        indices=np.random.choice(x,arrSize//5,replace=False)
        self.assertTrue((x[m.arrayCountingArgSort(x,maxElem,indices)]==x[indices[np.argsort(x[indices])]]).all())
        #Performance

        #    whole array
        baselineTime=tests.repeatManyTimes(baselineArgSort,x)    
        print("Baseline np.argsort test for whole array took %s" % baselineTime)        
        optimizedTime=tests.repeatManyTimes(m.arrayCountingArgSort,x,maxElem)
        print("Optimized arrayCountingArgSort test for whole array took %s" % optimizedTime)
        self.assertTrue(optimizedTime<baselineTime)
        #    indexed array
        baselineTime=tests.repeatManyTimes(baselineArgSortIndexed,x,indices)    
        print("Baseline np.argsort test for indexed array took %s" % baselineTime)        
        optimizedTime=tests.repeatManyTimes(m.arrayCountingArgSort,x,maxElem,indices)
        print("Optimized arrayCountingArgSort test for indexed array took %s" % optimizedTime)
        self.assertTrue(optimizedTime<baselineTime)        
    def test_arrayCountingArgSortParallel(self):
        x=np.random.randint(minElem,maxElem,arrSize)
        #Validity
        #    whole array
        self.assertTrue((x[m.arrayCountingArgSortThreaded(x,maxElem)]==x[np.argsort(x)]).all())
        #    indexed array
        indices=np.random.choice(x,arrSize//5,replace=False)
        self.assertTrue((x[m.arrayCountingArgSortThreaded(x,maxElem,indices)]==x[indices[np.argsort(x[indices])]]).all())
        #Performance        
        #    whole array
        baselineTime=tests.repeatManyTimes(baselineArgSort,x)    
        print("Baseline np.argsort test for whole array took %s" % baselineTime)
        
        optimizedTime=tests.repeatManyTimes(m.arrayCountingArgSortThreaded,x,maxElem)
        print("Optimized arrayCountingArgSortThreaded test for whole array took %s" % optimizedTime)
        self.assertTrue(optimizedTime<baselineTime)        
        #    indexed array    
        baselineTime=tests.repeatManyTimes(baselineArgSortIndexed,x,indices)    
        print("Baseline np.argsort test for indexed array took %s" % baselineTime)
        
        optimizedTime=tests.repeatManyTimes(m.arrayCountingArgSortThreaded,x,maxElem,indices)
        print("Optimized arrayCountingArgSortThreaded test for indexed array took %s" % optimizedTime)
        self.assertTrue(optimizedTime<baselineTime)        
    def test_arrayCountingArgSortAndUniqueValues(self):
        x=np.random.randint(minElem,maxElem,arrSize)
        #Validity
        #    whole array
        self.assertTrue((x[m.arrayCountingArgSortAndUniqueValues(x,maxElem)[2]]==x[np.argsort(x)]).all())
        #    indexed array
        indices=np.random.choice(x,arrSize//5,replace=False)
        self.assertTrue((x[m.arrayCountingArgSortAndUniqueValues(x,maxElem,indices)[2]]==x[indices[np.argsort(x[indices])]]).all())
        #Performance
        #    whole array
        baselineTime=tests.repeatManyTimes(baselineArgSort,x)    
        print("Baseline np.argsort test for whole array took %s" % baselineTime)
        
        optimizedTime=tests.repeatManyTimes(m.arrayCountingArgSortAndUniqueValues,x,maxElem)
        print("Optimized arrayCountingArgSortAndUniqueValues test for whole array took %s" % optimizedTime)
        self.assertTrue(optimizedTime<baselineTime)
        #    indexed array
        baselineTime=tests.repeatManyTimes(baselineArgSortIndexed,x,indices)    
        print("Baseline np.argsort test for indexed array took %s" % baselineTime)
        
        optimizedTime=tests.repeatManyTimes(m.arrayCountingArgSortAndUniqueValues,x,maxElem,indices)
        print("Optimized arrayCountingArgSortAndUniqueValues test for indexed array took %s" % optimizedTime)
        self.assertTrue(optimizedTime<baselineTime)         
        
if __name__ == '__main__':    
    unittest.main()