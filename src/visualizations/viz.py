#Visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

#Matplotlib preferences/fonts
plt.style.use('seaborn')
header_font = fm.FontProperties(fname='../src/visualizations/Fonts/Montserrat/Montserrat-Regular.ttf', size = 14)
text_font = fm.FontProperties(fname='../src/visualizations/Fonts/Lato/Lato-Regular.ttf', size = 14)
cbar_font = fm.FontProperties(fname='../src/visualizations/Fonts/Lato/Lato-Regular.ttf', size = 30)

def hist(var):
    """Simple histogram.

    Args:
        var (obj): Pandas series for desired histogram.

    Returns:
        matplotlib histogram.
    """
    plt.hist(var)

    return

def class_word_count_plots(rows, cols, counter_dict):
    """Generates subplots for top words in each grouping variable.

    Args:
        rows, cols (int): row/column layout for subplots.
        counter_dict (dict): dictionary with keys representing classes
          and values representing
    Returns:
        A figure countaing horizontal barchart subplots
        of the 10 most common words for each class in counter_dict.
    """
    fig, ax = plt.subplots(rows, cols, figsize = (20, 10))
    ax = ax.flatten()

    for idx,class_ in enumerate(counter_dict.keys()):
        words = [word for word, freq in counter_dict[class_]][::-1]
        freq = [freq for word, freq in counter_dict[class_]][::-1]
        #Plotting
        ax[idx].barh(words, freq)
        ax[idx].set_title(class_)

    return

def word_sims_plots(rows,
                    cols,
                    words,
                    model,
                    num_sim_words,
                    colors
                    ):
    """Generates horizontal bar charts for word similarities.

    Args:
        rows, cols (int): Row/column layout for subplots.
        words (list): Words to find similarities to. Minimum is 2.
        model (obj): Trained Doc2Vec model.
        num_sim_words (int): Number of most similar words to find.
        colors: Colors used for each plot. Number of colors
          should be equal to number of `words`.

    Returns:
        A figure countaing horizontal barchart subplots of the 10
        most common words for each class in counter_dict.
    """
    assert len(words) >= 2, 'Must input at least 2 words.'

    word_params = []
    for idx, val in enumerate(words):
        word_params.append((words[idx], model.wv.most_similar(words[idx], topn = num_sim_words), colors[idx]))

    fig, ax = plt.subplots(rows, cols, figsize = (25,7))
    fig.subplots_adjust(wspace = .5)
    fig.set_facecolor('w')

    for idx, val in enumerate(word_params):
        words = [pair[0] for pair in val[1]][::-1]
        similarities = [pair[1] for pair in val[1]][::-1]
        ax[idx].grid(True,
                     linewidth=.2,
                     color = '#000000'
                    )
        ax[idx].set_facecolor('w')
        ax[idx].barh(words,
                     similarities,
                     height = .4,
                     color = val[2]
                    )
        ax[idx].set_xlabel('Similarity Score', font_properties = text_font)
        ax[idx].set_xlim(0,1)
        ax[idx].set_xticks(np.round(np.arange(0,1.1,.2),2))
        ax[idx].set_xticklabels(ax[idx].get_xticks(),
                                font_properties = text_font,
                                size = 12
                               )
        ax[idx].set_yticklabels(words, font_properties = text_font)
        ax[idx].set_title(val[0].capitalize(),
                          font_properties = header_font,
                          size = 20
                         )
    return

def model_comparison(df):
    """Plots model F1 scores in descending order.

    Args:
        df (obj): Pandas dataframe.

    Returns:
        A horizontal barchart of model performances ordered
        by descending F1 scores.
    """
    #Sorting, turning labels into actual English
    replace_jargon = {'PV-DBOW': 'Doc Embeddings', 'TFIDF': 'Tf-idf',
                      'MOWE': 'Word Embeddings', 'IDF-MOWE': 'Weighted Embeddings'                  }
    sorted_initial_eval = df.sort_values('F1', ascending = False).replace(replace_jargon)

    #OCD formatting
    feature_mods = list(zip(sorted_initial_eval['Data'].tolist()[:10],
                            sorted_initial_eval['Model'].tolist()[:10]))
    feat_mods_list = [f'{i[0]} ({i[1]})' for i in feature_mods]

    #Color-coding to highlight top 4 models
    sorted_initial_eval['color'] = ['#FF7E4E' if i < 4 else '#C0C0C0' for i in range(sorted_initial_eval.shape[0])]

    fig, ax = plt.subplots(figsize = (25,20))
    ax.set_facecolor('w')
    ax.set_title('Initial Validation Models - Top 10',
                 font_properties = header_font,
                 size = 40,
                 pad = 40
                )
    ax.grid(True,
            linewidth=.2,
            color = '#000000'
            )
    ax.barh(feat_mods_list[::-1],
            sorted_initial_eval['F1'].tolist()[:10][::-1],
            height = .6,
            color = sorted_initial_eval['color'].tolist()[:10][::-1]
           )
    ax.set_xlabel('F1 Score',
                  font_properties = text_font,
                  size = 30,
                  labelpad = 30
                 )
    ax.set_xticklabels(np.round(ax.get_xticks(),2),
                       font_properties = text_font,
                       size = 25
                      )
    ax.set_yticklabels(feat_mods_list[::-1],
                       font_properties = text_font,
                       size = 30
                      )

    return

def confusion_heatmap(y_test, preds, groups):
    """Generates a custom confusion matrix.

    Code adapted from: https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix

    Args:
        y_test, preds (arr): Array of numeric values. Length of y_test should equal preds.
        groups (list): Names of groups in y_test/preds.
    Returns:
        A Seaborn heatmap representing a scikit-learn confusion matrix.
    """
    array = confusion_matrix(y_test, preds)
    df_cm = pd.DataFrame(array, index = [i for i in groups],
                         columns = [i for i in groups]
                        )
    fig, ax = plt.subplots(figsize = (25,25))

    color_map = sns.light_palette('#ff4500', n_colors = 50)
    ax = sns.heatmap(df_cm,
                     annot=True,
                     cmap = color_map,
                     fmt = 'g',
                     annot_kws = {'font_properties':text_font, 'fontsize':35}
                    )
    ax.set_xlabel('Actual Class',
                  font_properties = text_font,
                  size = 40,
                  labelpad = 50
                 )
    ax.set_xticklabels(df_cm.columns,
                       font_properties = header_font,
                       size = 35
                      )
    ax.set_ylabel('Predicted Class',
                  font_properties = header_font,
                  size = 40,
                  labelpad = 50
                 )
    ax.set_yticklabels(df_cm.index,
                       font_properties = text_font,
                       size = 35
                      )
    cbar = ax.collections[0].colorbar
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),
                            font_properties = text_font,
                            size = 30
                           );

    return
