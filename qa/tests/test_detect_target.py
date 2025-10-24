
import pandas as pd
from ml.utils import detect_target_and_task

def test_detect_target_and_task():
    df = pd.DataFrame({'a':[1,2,3,4], 'b':['x','y','x','y'], 'label':[0,1,0,1]})
    t, task = detect_target_and_task(df, None)
    assert t == 'label'
    assert task == 'classification'
