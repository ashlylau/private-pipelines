
from collections import deque
from logging import getLogger, basicConfig, INFO

logger = getLogger(__name__)
basicConfig(level=INFO)

def arr_n_check(privacy, test_range, n_checks):
   """
   returns the test_privacy tuple
   seek for it to be symmetrical, therefore, equal number of observations on
   either side of the original privacy budget.
   As a result n_checks should always be an odd number.
   test_range: it is the absolute difference between consecutive test privacy
   budget being returned
   """
   if privacy < 0:
       p_old = privacy
       privacy = -privacy
       logger.info(f"value provided for privacy(={p_old}) argument is negative; \
       using instead privacy={privacy}")
   if n_checks < 0:
       check_old = n_checks
       n_checks = -1*n_checks
       logger.info(f"value provided for n_checks(={check_old}) argument is negative; \
       using instead n_checks={n_checks}")
   if n_checks % 2 == 0:
       check_old = n_checks
       n_checks = n_checks-1 if n_checks > 2 else 3
       logger.info(f"value provided for n_checks(={check_old}) argument is even or \
       smaller than 3; using instead n_checks={n_checks}")

   return_tuple = deque([privacy])
   for i in range(1, n_checks//2 + 1):
       l = privacy - (i*test_range)
       if l < 0:
           logger.info(f"value for test privacy reached below 0 on {i-1}th iteration\
           terminating further population of the tuple")
           break
       r = privacy + (i*test_range)
       return_tuple.appendleft(l)
       return_tuple.append(r)
   return_tuple = tuple(return_tuple)
   logger.info(f"final test_privacy tuple at PyStatDP/pystatdp/utils.py; \
   arr_n_check() = str{return_tuple}")
   return return_tuple
