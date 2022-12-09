from spinup.utils.plot import make_plots


make_plots( all_logdirs=[r'D:\WS\RLcodes\spinningup-master\data\ppo'],
            xaxis='Epoch',
            values='AverageEpRet',
            # values='LossV',
            smooth=10)

