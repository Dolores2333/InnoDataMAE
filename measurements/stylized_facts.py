import visualize as vs
import numpy as np
import stats
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#volatility clustering
def acf(x,file_name,for_abs=True,multiple=False,fit=False,scale='log',max_lag=1000):
    if for_abs:
        x = np.abs(x)
    if multiple:
        res = np.zeros(max_lag)
        for e in x:
            res += stats.acf(e)
        res /= x.shape[0]
    else:
        res = stats.acf(x)
    vs.acf(res,file_name,scale=scale)

def leverage_effect(x,file_name,multiple=True,min_lag=1,max_lag=100):
    def compute_levs(x,x_abs):
        Z = (np.mean(x_abs**2))**2
        second_term = np.mean(x)*np.mean(x_abs**2)
        def compute_for_t(t):
            if t == 0:
                first_term = np.mean(x*(x_abs)**2)
            elif t > 0:
                first_term = np.mean(x[:-t]*(x_abs[t:]**2))
            else:
                first_term = np.mean(x[-t:]*(x_abs[:t]**2) )
            return (first_term-second_term)/Z
        levs = [compute_for_t(t) for t in range(min_lag,max_lag)]
        return np.array(levs)

    x_abs = np.abs(x)
    if multiple:
        levs = np.zeros(max_lag-min_lag)
        for e1,e2 in zip(x,x_abs):
            levs += compute_levs(e1,e2)
        levs /= x.shape[0]
    else:
        levs = compute_levs(x,x_abs)
    vs.leverage_effect([i for i in range(min_lag,max_lag)],levs,file_name)
    return levs


#fat-tail
def distribution(x,file_name,scale='linear',multiple=False,normalize=True,granuality=100):
    #preprocessing
    if multiple:
        x = np.reshape(x,x.size)
    if normalize:
        x = normalize_time_series(x)
    if scale is 'linear':
        dist_x,dist_y = linear_pdf(x,granuality=granuality)
        vs.distribution(dist_x, dist_y, file_name, 'linear')
        return dist_x, dist_y
    elif scale is 'log':
        dist_x,dist_y = linear_pdf(x,granuality=granuality)
        vs.distribution(dist_x, dist_y, file_name, 'log')
        pass
    else:
        pass

def culmulative_distribution(x,scale='linear',normalize=True):
    pass

def normalize_time_series(x):
    mean = np.mean(x)
    std = np.std(x)
    x = (x-mean)/std
    return x

def linear_pdf(x,dist_x=None,granuality=100):
    if dist_x is None:
        x_max = 5.
        x_min = -5.
    dist_x = np.linspace(x_min,x_max,granuality)
    diff = dist_x[1]-dist_x[0]
    dist_x_visual = (dist_x + diff)[:-1]
    dist_y = np.zeros(granuality-1)
    for e,(x1,x2) in enumerate(zip(dist_x[:-1],dist_x[1:])):
        dist_y[e] = x[np.logical_and(x > x1,x < x2)].size
    dist_y /= x.size
    return dist_x_visual,dist_y


def log_pdf(x,dist_x=None,granuality=100):
    pass

def cdf(x,scale='linear'):
    pass

#coarse_fine_volatility_correlaion

def Coarse_fine(x,file_name,T=5):
    corr,corr_diff=stats.Correlation(x,T=T)
    vs.coarse_fine_volatility_corr(corr,corr_diff,file_name)

def Gain_or_loss_asymtric(x,file_name,theta=0.1):
    gain_prob,loss_prob = stats.Gain_or_loss_probability(x,theta = theta)
    vs.Gain_or_loss_asymtric_distribution(gain_prob,loss_prob,file_name)





# %% PCA Analysis

def PCA_Analysis(dataX, dataX_hat,file_name):
    # Analysis Data Size
    Sample_No = 1000

    # Data Preprocessing
    for i in range(Sample_No):
        if (i == 0):
            arrayX = np.reshape(np.mean(np.asarray(dataX[0]), 1), [1, len(dataX[0][:, 0])])
            arrayX_hat = np.reshape(np.mean(np.asarray(dataX_hat[0]), 1), [1, len(dataX[0][:, 0])])
        else:
            arrayX = np.concatenate((arrayX, np.reshape(np.mean(np.asarray(dataX[i]), 1), [1, len(dataX[0][:, 0])])))
            arrayX_hat = np.concatenate(
                (arrayX_hat, np.reshape(np.mean(np.asarray(dataX_hat[i]), 1), [1, len(dataX[0][:, 0])])))

    # Parameters
    No = len(arrayX[:, 0])
    colors = ["red" for i in range(No)] + ["blue" for i in range(No)]

    # PCA Analysis
    pca = PCA(n_components=2)
    pca.fit(arrayX)
    pca_results = pca.transform(arrayX)
    pca_hat_results = pca.transform(arrayX_hat)

    # Plotting
    plt.figure(dpi=150)

    plt.scatter(pca_results[:, 0], pca_results[:, 1], c=colors[:No], alpha=0.2, label="Original")
    plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1], c=colors[No:], alpha=0.2, label="Synthetic")

    plt.title('PCA plot')
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    plt.legend()
    plt.savefig(file_name+'.png',transparent=True)
    plt.close()



# %% TSNE Analysis

def tSNE_Analysis(dataX, dataX_hat,file_name):
    # Analysis Data Size
    Sample_No = 1000
    # Preprocess
    for i in range(Sample_No):
        if (i == 0):
            arrayX = np.reshape(np.mean(np.asarray(dataX[0]), 1), [1, len(dataX[0][:, 0])])
            arrayX_hat = np.reshape(np.mean(np.asarray(dataX_hat[0]), 1), [1, len(dataX[0][:, 0])])
        else:
            arrayX = np.concatenate((arrayX, np.reshape(np.mean(np.asarray(dataX[i]), 1), [1, len(dataX[0][:, 0])])))
            arrayX_hat = np.concatenate(
                (arrayX_hat, np.reshape(np.mean(np.asarray(dataX_hat[i]), 1), [1, len(dataX[0][:, 0])])))

    # Do t-SNE Analysis together
    final_arrayX = np.concatenate((arrayX, arrayX_hat), axis=0)

    # Parameters
    No = len(arrayX[:, 0])
    colors = ["red" for i in range(No)] + ["blue" for i in range(No)]

    # TSNE anlaysis
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(final_arrayX)

    # Plotting
    plt.figure(dpi=150)

    plt.scatter(tsne_results[:No, 0], tsne_results[:No, 1], c=colors[:No], alpha=0.2, label="Original")
    plt.scatter(tsne_results[No:, 0], tsne_results[No:, 1], c=colors[No:], alpha=0.2, label="Synthetic")


    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.legend()
    plt.savefig(file_name+'.png',transparent=True)
    plt.close()
