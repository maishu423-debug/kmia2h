import time, sys
import pandas as pd
from kmia2h.residual import refit
from kmia2h.forcing import model_cfg
t = time.time()
cfg = model_cfg()
res, diag, X = refit.main(sys.argv[1], sys.argv[2], sys.argv[3], cfg)
print('X stats', X.x.describe().round(3).to_dict())
pd.set_option('display.width', 250)
print(pd.DataFrame(diag).T.to_string())
print('regimes', len(res['regimes']), 'step check', res['step_check_f'], 'took', time.time() - t)
