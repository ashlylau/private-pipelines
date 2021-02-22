# Usage

```python
from pystatdp import pystatdp
#import the algorithm you want to validate
from pydp.algorithms.laplacian import BoundedMean

psd = pystatdp()
#Use the main method from pystatdp to validate the algorithm
psd.main(BoundedMean, tuple((-15, 15)), tuple((0.9,)), e_iter=1000,
        d_iter=5000, test_range= 0.5)
```

### Arguments
```bash
algo         algorithm
param        parameters for the algorithm
privacy      privacy budget
```
### Optional Arguments
```bash
e_iter       event iterations; default: 100000
d_iter       detection iterations; default: 500000
test_range   absolute difference between two consecutive test privacy budgets; default = 0.1
n_checks     Number of test privacy budgets to validate for; default: 3
```
