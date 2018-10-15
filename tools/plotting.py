import matplotlib
import matplotlib.pyplot as plt

def plt_act_pred(act, pred, dates, title):
    act.set_index(dates, inplace=True)
    act.plot(label='Actual', c='b', title=title)
    pred.set_index(dates, inplace=True)
    pred['Predicted'].plot(label='Predicted', c='r', linestyle='--')

