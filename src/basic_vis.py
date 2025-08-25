import matplotlib.pyplot as plt
import os

def plot_variable_histograms(col_names, df):
    
    # Filter columns based on col_names
    df_filtered = df[col_names]
    
    outPath = './plots'
    
    # Plot some of the data, just to make sure it looks ok
    for c, vals in df_filtered.items():
        n = vals.dropna().count()
        if n < 2: continue

        # get median, variance, skewness
        med = vals.dropna().median()
        var = vals.dropna().var()
        skew = vals.dropna().skew()

        # plot
        fig = plt.figure(figsize=(13, 6))
        plt.subplots(figsize=(13,6))
        vals.dropna().plot.hist(bins=100, label='HIST (n={})'.format(n))

        # fake plots for KS test, median, etc
        plt.plot([], label=' ',color='lightgray')
        plt.plot([], label='Median: {}'.format(format(med,'.2f')),
                 color='lightgray')
        plt.plot([], label='Variance: {}'.format(format(var,'.2f')),
                 color='lightgray')
        plt.plot([], label='Skew: {}'.format(format(skew,'.2f')),
                 color='lightgray')

        # add title, labels etc.
        plt.title('{} measurements in ICU '.format(str(c)))
        plt.xlabel(str(c))
        plt.legend(loc="upper left", bbox_to_anchor=(1,1),fontsize=12)
        plt.xlim(0, vals.quantile(0.99))
        
        #fig.savefig(os.path.join(outPath, (str(c) + '_HIST_.png')), bbox_inches='tight')