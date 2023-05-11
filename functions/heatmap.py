import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
%config InlineBackend.figure_format = "retina"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score_file",
        type=str,
        help="Path to input mutation score file.",
    )
    
    parser.add_argument(
        "--save_file_dir",
        type=str,
        help="Path of the directory for saving files.",
    )

    parser.add_argument(
        "--wt_seq",
        type=str,
        help="Wild type sequence for mutation.",
    )

    args = parser.parse_args()
    
    df = pd.read_csv(args.score_file + "/mutations.tsv", delimiter='\t')['diff']
    row_list = df.values.tolist()
    print(row_list)
    plot_matrix = []
    for i in range(0, 473, 4):
      plot_matrix.append(row_list[i:i+4])
    #print(plot_matrix) # ATGC
    array_plot = np.transpose(np.array(plot_matrix))
    #print(array_plot)
    sns.set(font_scale=3)
    plt.figure(figsize=(100, 5.4), dpi=300)
    cmap = sns.cubehelix_palette(10, start=2, rot=0, dark=0, light=.95, reverse=True, gamma = .5)
    ax = sns.heatmap(array_plot,cmap=cmap,xticklabels= list(args.wt_seq), 
                yticklabels=['A', 'U', 'G', 'C'])
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.savefig(args.save_file_dir + "/heatmap.jpg")


if __name__ == "__main__":
    main()
