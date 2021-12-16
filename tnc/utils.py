import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from torch.utils import data
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import hdbscan
from scipy.cluster.hierarchy import dendrogram

def create_simulated_dataset(window_size=50, path='./data/simulated_data/', batch_size=100):
    if not os.listdir(path):
        raise ValueError('Data does not exist')
    x = pickle.load(open(os.path.join(path, 'x_train.pkl'), 'rb'))
    y = pickle.load(open(os.path.join(path, 'state_train.pkl'), 'rb'))
    x_test = pickle.load(open(os.path.join(path, 'x_test.pkl'), 'rb'))
    y_test = pickle.load(open(os.path.join(path, 'state_test.pkl'), 'rb'))

    n_train = int(0.8*len(x))
    n_valid = len(x) - n_train
    n_test = len(x_test)
    x_train, y_train = x[:n_train], y[:n_train]
    x_valid, y_valid = x[n_train:], y[n_train:]

    datasets = []
    for set in [(x_train, y_train, n_train), (x_test, y_test, n_test), (x_valid, y_valid, n_valid)]:
        T = set[0].shape[-1]
        windows = np.split(set[0][:, :, :window_size * (T // window_size)], (T // window_size), -1)
        windows = np.concatenate(windows, 0)
        labels = np.split(set[1][:, :window_size * (T // window_size)], (T // window_size), -1)
        labels = np.round(np.mean(np.concatenate(labels, 0), -1))
        datasets.append(data.TensorDataset(torch.Tensor(windows), torch.Tensor(labels)))

    trainset, testset, validset = datasets[0], datasets[1], datasets[2]
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = data.DataLoader(validset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


def plot_heatmap(sample, clustering_model, encoder, normalization_specs, path, hm_file_name, device, signal_list, length_of_hour, window_size, sliding_gap):
    _, num_features, seq_len = sample.shape

    
    encodings = []
    with torch.no_grad():
        encoder.to(device)
        encoder.eval()
        for t in range(0, seq_len-window_size+1, sliding_gap):
            window = sample[:, :, t:t+window_size]

            encodings.append(encoder(torch.unsqueeze(window, 0).to(device)).view(-1,)) # Add a dimension of size 1 at beginning so we have a 'batch' of size 1 for the encoder to work with.
        
    encodings = torch.stack(encodings, 0).to('cpu').detach().float()

    


    # Since encodings have now been generated, we can un normalize our data for plotting
    means, stds = normalization_specs[0], normalization_specs[1]
    means = means.reshape(-1, 1)
    stds = stds.reshape(-1, 1)
    sample[0][sample[1]==1] = (sample[0] * stds)[sample[1]==1] # At the places in the data where the map is 1, multiply that data by std
    sample[0][sample[1]==1] = (sample[0] + means)[sample[1]==1] # Then add mean


    if 'waveform' in path:
        f, axs = plt.subplots(3)
        f.set_figheight(12)
        f.set_figwidth(27)
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 7])
        axs[0] = plt.subplot(gs[0])
        axs[1] = plt.subplot(gs[1])
        axs[2] = plt.subplot(gs[2])
        sns.lineplot(np.arange(0,sample.shape[1]/250, 1./250), sample[0], ax=axs[0])
        sns.lineplot(np.arange(0,sample.shape[1]/250, 1./250), sample[1], ax=axs[1])
        axs[1].margins(x=0)
        axs[1].grid(False)
        axs[1].xaxis.set_tick_params(labelsize=22)
        axs[1].yaxis.set_tick_params(labelsize=22)

    else: # We'll have num_features plots + 1 for the heatmap
        f, axs = plt.subplots(num_features + 1)  #
        f.set_figheight(num_features*7)
        f.set_figwidth(23)
        for feat in range(num_features):
            sns.lineplot(np.arange(seq_len), sample[0][feat].to('cpu'), ax=axs[feat])

    for i in range(len(signal_list)):
        axs[i].set_title(signal_list[i], fontsize=30, fontweight='bold')

    
    num_hours = int(seq_len/length_of_hour)
    for i in range(sample.shape[-2] + 1):
        axs[i].set_xlabel('Time (Hours)', fontsize=28)
        
        axs[i].set_xticks(np.arange(num_hours)*length_of_hour)
        axs[i].set_xticklabels(np.arange(num_hours))
    
        axs[i].xaxis.set_tick_params(labelsize=22)
        axs[i].yaxis.set_tick_params(labelsize=22)
        
        axs[i].margins(x=0)
        axs[i].grid(False)

    try:
        cluster_labels = hdbscan.approximate_predict(clustering_model, [encoding for encoding in encodings]) # labels are the labels that the clustering model gives to each encoding in the sample
        if 'ICU' in hm_file_name:
            for i in range(num_features):
                t_0 = 0
                for t in range(1, cluster_labels.shape[-1]):
                    if cluster_labels[t]==cluster_labels[t-1] and t < cluster_labels.shape[-1] -1: # If the label is the same as the last time step and we're not at the end yet
                        continue
                    else:
                        axs[i].axvspan((t_0)*window_size, (t+1)*window_size, facecolor=['g', 'r', 'b', 'y', 'm', 'c', 'k', 'w'][int(cluster_labels[t_0])], alpha=0.25)
                        t_0 = t+1
    except:
        pass
    
        
        
    axs[-1].set_title('Encoding Trajectory', fontsize=30, fontweight='bold')
    axs[-1].set_xlabel('Time (Hours)', fontsize=28) 
    axs[-1].set_xticks(np.arange(num_hours)*length_of_hour)
    axs[-1].set_xticklabels(np.arange(num_hours))

    sns.heatmap(encodings.detach().cpu().numpy().T, cbar=False, linewidth=0.5, ax=axs[-1], linewidths=0.05, xticklabels=False)
    f.tight_layout()
    

    # sns.heatmap(encodings.detach().cpu().numpy().T, linewidth=0.5)
    plt.savefig(os.path.join(path, hm_file_name))

    # windows = np.split(sample[:, :window_size * (T // window_size)], (T // window_size), -1)
    # windows = torch.Tensor(np.stack(windows, 0)).to(encoder.device)
    # windows_label = np.split(label[:window_size * (T // window_size)], (T // window_size), -1)
    # windows_label = torch.Tensor(np.mean(np.stack(windows_label, 0), -1 ) ).to(encoder.device)
    # encoder.to(encoder.device)
    # encodings = encoder(windows)


def plot_pca_trajectory(sample, encoder, window_size, device, sliding_gap, kmeans_model, path, pca_file_name):

    _, num_features, seq_len = sample.shape

    
    encodings = []
    with torch.no_grad():
        encoder.to(device)
        encoder.eval()
        for t in range(0, seq_len-window_size+1, sliding_gap):
            window = sample[:, :, t:t+window_size]

            encodings.append(encoder(torch.unsqueeze(window, 0).to(device)).view(-1,)) # Add a dimension of size 1 at beginning so we have a 'batch' of size 1 for the encoder to work with.
        
    encodings = torch.stack(encodings, 0).to('cpu').detach().float()
    




    pca = PCA(n_components=2)
    embedding = pca.fit_transform(encodings.detach().cpu().numpy())
    d = {'f1':embedding[:,0], 'f2':embedding[:,1], 'Time to arrest':np.arange(len(embedding)-1, -1, -1)}#, 'label':windows_label}
    df = pd.DataFrame(data=d)

    fig, ax = plt.subplots()
    ax.set_title("Trajectory")
    # sns.jointplot(x="f1", y="f2", data=df, kind="kde", size='time', hue='label')
    sns.scatterplot(x="f1", y="f2", data=df, hue='Time to arrest')
    plt.savefig(os.path.join(path, pca_file_name))

def plot_distribution(x_test, y_test, encoder, window_size, path, device, title="", augment=4, cv=0):
    checkpoint = torch.load('./ckpt/%s/checkpoint_%d.pth.tar'%(path, cv))
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder = encoder.to(device)
    n_test = len(x_test)
    inds = np.random.randint(0, x_test.shape[-1] - window_size, n_test * augment)
    windows = np.array([x_test[int(i % n_test), :, ind:ind + window_size] for i, ind in enumerate(inds)])
    windows_state = [np.round(np.mean(y_test[i % n_test, ind:ind + window_size], axis=-1)) for i, ind in
                     enumerate(inds)]
    encodings = encoder(torch.Tensor(windows).to(device))

    tsne = TSNE(n_components=2)
    embedding = tsne.fit_transform(encodings.detach().cpu().numpy())
    # pca = PCA(n_components=2)
    # embedding = pca.fit_transform(encodings.detach().cpu().numpy())
    # original_embedding = PCA(n_components=2).fit_transform(windows.reshape((len(windows), -1)))
    original_embedding = TSNE(n_components=2).fit_transform(windows.reshape((len(windows), -1)))


    df_original = pd.DataFrame({"f1": original_embedding[:, 0], "f2": original_embedding[:, 1], "state": windows_state})
    df_encoding = pd.DataFrame({"f1": embedding[:, 0], "f2": embedding[:, 1], "state": windows_state})
    # df_encoding = pd.DataFrame({"f1": embedding[:, 0], "f2": embedding[:, 1], "state": y_test[np.arange(4*n_test)%n_test, inds]})


    # Save plots
    if not os.path.exists(os.path.join("./plots/%s"%path)):
        os.mkdir(os.path.join("./plots/%s"%path))
    # plt.figure()
    fig, ax = plt.subplots()
    ax.set_title("Origianl signals TSNE", fontweight="bold")
    # sns.jointplot(x="f1", y="f2", data=df_original, kind="kde", hue='state')
    sns.scatterplot(x="f1", y="f2", data=df_original, hue="state")
    plt.savefig(os.path.join("./plots/%s"%path, "signal_distribution.pdf"))

    fig, ax = plt.subplots()
    # plt.figure()
    ax.set_title("%s"%title, fontweight="bold", fontsize=18)
    if 'waveform' in path:
        sns.scatterplot(x="f1", y="f2", data=df_encoding, hue="state", palette="deep")
    else:
        sns.scatterplot(x="f1", y="f2", data=df_encoding, hue="state")
    # sns.jointplot(x="f1", y="f2", data=df_encoding, kind="kde", hue='state')
    plt.savefig(os.path.join("./plots/%s"%path, "encoding_distribution_%d.pdf"%cv))


def model_distribution(x_train, y_train, x_test, y_test, encoder, window_size, path, device):
    checkpoint = torch.load('./ckpt/%s/checkpoint.pth.tar'%path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    # n_train = len(x_train)
    n_test = len(x_test)
    augment = 100

    # inds = np.random.randint(0, x_train.shape[-1] - window_size, n_train * 20)
    # windows = np.array([x_train[int(i % n_train), :, ind:ind + window_size] for i, ind in enumerate(inds)])
    # windows_label = [np.round(np.mean(y_train[i % n_train, ind:ind + window_size], axis=-1))
    #                  for i, ind in enumerate(inds)]
    inds = np.random.randint(0, x_test.shape[-1] - window_size, n_test * augment)
    x_window_test = np.array([x_test[int(i % n_test), :, ind:ind + window_size] for i, ind in enumerate(inds)])
    y_window_test = np.array([np.round(np.mean(y_test[i % n_test, ind:ind + window_size], axis=-1)) for i, ind in
                     enumerate(inds)])
    train_count = []
    if 'waveform' in path:
        encoder.to('cpu')
        x_window_test = torch.Tensor(x_window_test)
    else:
        encoder.to(device)
        x_window_test = torch.Tensor(x_window_test).to(device)

    encodings_test = encoder(x_window_test).detach().cpu().numpy()

    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(encodings_test, y_window_test)
    # preds = neigh.predict(encodings_test)
    _, neigh_inds = neigh.kneighbors(encodings_test)
    neigh_ind_labels = [np.mean(y_window_test[ind]) for ind in (neigh_inds)]
    label_var = [(y_window_test[ind]==y_window_test[i]).sum() for i, ind in enumerate(neigh_inds)]
    dist = (label_var)/10


def confidence_ellipse(mean, cov, ax, n_std=1.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, edgecolor='navy',
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def trend_decompose(x, filter_size):
    df = pd.DataFrame(data=x.T)
    df = df.rolling(filter_size, win_type='triang').sum()
    s = df.loc[:, 0]
    f, axs = plt.subplots(1)
    print(s[filter_size-1:].shape, x[0,:-filter_size+1].shape)
    axs.plot(s[filter_size-1:], c='red')
    axs.plot(x[0,:-filter_size+1], c='blue')
    plt.show()


def dim_reduction_mixed_clusters(negative_encodings, positive_encodings, negative_cluster_labels, positive_cluster_labels, data_type, unique_name, unique_id):
    colors = ['g', 'r', 'b', 'y', 'm', 'c', 'k', 'w']
    
    # TRAIN ON MIXED, NOT JUST NEG AND POS

    mixed_encodings = np.vstack([positive_encodings, negative_encodings])
    mixed_cluster_labels = np.concatenate([positive_cluster_labels, negative_cluster_labels])
    pos_inds = np.arange(len(positive_encodings))
    neg_inds = np.arange(len(positive_encodings), len(mixed_encodings))

    # PLOTTING NORMAL PLOT
    embeddings = TSNE(n_components=2).fit_transform(mixed_encodings)
    labels = negative_cluster_labels

    reduced_negative_encodings = embeddings[neg_inds]
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_negative_encodings[:, 0], reduced_negative_encodings[:, 1], c=[colors[i] for i in labels], label=labels)
    plt.title('Normal Encodings Projected onto 2D')

    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.savefig('../DONTCOMMITplots/%s/%s/%s_negative_embeddings.pdf'%(data_type, unique_id, unique_name))


    # PLOTTING ARREST PLOT
    labels = positive_cluster_labels

    reduced_positive_encodings = embeddings[pos_inds]
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_positive_encodings[:, 0], reduced_positive_encodings[:, 1], c=[colors[i] for i in labels], label=labels)
    plt.title('Positive Encodings Projected onto 2D')

    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.savefig('../DONTCOMMITplots/%s/%s/%s_positive_embeddings.pdf'%(data_type, unique_id, unique_name))


    # PLOTTING MIXED PLOT
    labels = mixed_cluster_labels

    reduced_mixed_encodings = mixed_encodings
    plt.figure(figsize=(8, 6))    
    plt.scatter(reduced_mixed_encodings[:, 0], reduced_mixed_encodings[:, 1], c=[colors[i] for i in labels], label=labels)
    plt.title('Positive and Negative Encodings Projected onto 2D')

    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.savefig('../DONTCOMMITplots/%s/%s/%s_positive_and_negative_embeddings.pdf'%(data_type, unique_id, unique_name))
    
def dim_reduction_positive_clusters(positive_encodings, positive_cluster_labels, data_type, unique_name, unique_id):
    colors = ['g', 'r', 'b', 'y', 'm', 'c', 'k', 'w']
    
    reduced_positive_encodings = TSNE(n_components=2).fit_transform(positive_encodings)
    labels = positive_cluster_labels

    plt.figure(figsize=(8, 6))    
    plt.scatter(reduced_positive_encodings[:, 0], reduced_positive_encodings[:, 1], c=[colors[i] for i in labels], label=labels)
    plt.title('Arrest Encodings Projected onto 2D')

    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.savefig('../DONTCOMMITplots/%s/%s/%s_positive_trained_clustering_positive_embeddings.pdf'%(data_type, unique_id, unique_name))



def plot_dendrogram(model, title, file_name, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    plt.figure(figsize=(10, 5))
    plt.title(title)
    dendrogram(linkage_matrix, **kwargs)
    plt.savefig(file_name)



def plot_negative_and_mortality(negative_data_maps, mortality_data_maps, mortality_labels, encoder, device, window_size, data_type, unique_name, unique_id):
    mortality_samples = []

    for i in range(len(mortality_labels)):
        if 1 == mortality_labels[i]:
            low = np.random.randint(0, mortality_data_maps.shape[-1]-window_size)
            mortality_samples.append(torch.Tensor(mortality_data_maps[i, :, :, low: low+window_size]))

    negative_samples = []

    for i in range(len(negative_data_maps)):
        low = np.random.randint(0, negative_data_maps.shape[-1]-window_size)
        negative_samples.append(torch.Tensor(negative_data_maps[i, :, :, low: low+window_size]))

            
    mortality_samples = torch.stack(mortality_samples).to(device)
    negative_samples = torch.stack(negative_samples).to(device)

    mortality_encodings = encoder(mortality_samples).to('cpu').detach().numpy()
    negative_encodings = encoder(negative_samples).to('cpu').detach().numpy()
    
    encodings = np.vstack([negative_encodings, mortality_encodings])
    labels = np.ones(len(negative_encodings) + len(mortality_encodings))
    labels[0:len(negative_encodings)] = 0

    pca = PCA(n_components=2)
    pca.fit(encodings)
    labels = np.array(labels)
    #labels = np.reshape(labels, (labels.shape[0], 1))

    reduced_encodings = pca.transform(encodings)
    fig = plt.figure(figsize=(8, 6))

    colors = []
    for i in labels:
        if i == 0:
            colors.append("Black") # negative patients
        elif i == 1:
            colors.append("Red")  # Pre positive window patients
        

    scatter = plt.scatter(reduced_encodings[:, 0], reduced_encodings[:, 1], c=colors, label=labels)

    plt.title('Mortality and Normal Encodings Projected onto 2D')

    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.savefig('../DONTCOMMITplots/%s/%s/%s_mortality_and_negative_embeddings.pdf'%(data_type, unique_id, unique_name))
    plt.close()
