from .utils import (
    init_module,
    init_dataset,
    dataset_to_df,
    group_users_by_preferences,
    analyze_user)
from recbole.config import Config
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from statsmodels.stats.api import DescrStatsW

def generate_report(df, output_folder: str, alternative: str = 'auto', confidence: float = 0.99):
    """
    Generates a report based on the user analysis DataFrame and saves it to the specified output folder.

    Args:
        df (pd.DataFrame): The DataFrame containing user analysis results.
        output_folder (str): The folder where the report will be saved.
        alternative (str): The alternative hypothesis for statistical testing.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save the DataFrame to a CSV file
    output_file = os.path.join(output_folder, 'user_analysis_data.csv')
    df.to_csv(output_file, index=False)
    print(f"Report generated and saved to {output_file}")

    sns.kdeplot(data=df, x="interactions", hue='group', multiple="stack", log_scale=True)
    plt.savefig(os.path.join(output_folder, 'user_analysis_report.pdf'))
    plt.close()
    with open(os.path.join(output_folder, 'report.txt'), 'w', encoding='utf-8') as f:
        print("===== Group Popularity Report =====", file=f)
        for i in range(df['group'].max() + 1):
            print(f"Group {i} statistics:", file=f)
            group_data = df[df['group'] == i]['popularity_score']
            mean = np.mean(group_data)
            std_dev = np.std(group_data, ddof=1)
            count = len(group_data)
            ds = DescrStatsW(df[df['group'] == i]['popularity_score'].values)
            print(f"Mean: {mean}, Std Dev: {std_dev}, Count: {count}", file=f)
            print(f"{confidence*100:.2f}% Confidence interval mean: {ds.tconfint_mean(alpha=1-confidence)}", file=f)
        print("\n===== Group Interactions Report =====", file=f)
        for i in range(df['group'].max() + 1):
            print(f"Group {i} statistics:", file=f)
            group_data = df[df['group'] == i]['interactions']
            mean = np.mean(group_data)
            std_dev = np.std(group_data, ddof=1)
            count = len(group_data)
            ds = DescrStatsW(df[df['group'] == i]['interactions'].values)
            print(f"Mean: {mean}, Std Dev: {std_dev}, Count: {count}", file=f)
            print(f"{confidence*100:.2f}% Confidence interval mean: {ds.tconfint_mean(alpha=1-confidence)}", file=f)
        print("\n===== Welch's t-test Results: difference interactions between groups =====", file=f)
        for i in range(df['group'].max() + 1):
            for j in range(i):
                # Example data: replace with your actual data
                group_A = df[df['group']==i]['interactions']
                group_B = df[df['group']==j]['interactions']
                
                # Calculate means and standard deviations
                mean_A = np.mean(group_A)
                mean_B = np.mean(group_B)
                
                std_A = np.std(group_A, ddof=1)
                std_B = np.std(group_B, ddof=1)
                
                n_A = len(group_A)
                n_B = len(group_B)
                
                # Perform Welch’s t-test (independent two-sample t-test, unequal variances)
                if alternative == 'auto':
                    # Default to two-sided test if not specified
                    alternative = 'two-sided'
                    t_stat, p_value = stats.ttest_ind(group_A, group_B, equal_var=False, alternative=alternative)  
                    if p_value < (1 - confidence) / 2:
                        alternative = 'less'
                        t_stat, p_value = stats.ttest_ind(group_A, group_B, equal_var=False, alternative=alternative)
                        if p_value > 1 - confidence:
                            alternative = 'greater'
                            # Recalculate with the new alternative hypothesis anyway

                t_stat, p_value = stats.ttest_ind(group_A, group_B, equal_var=False, alternative=alternative)  # one-tailed test
                
                # Degrees of freedom for Welch's t-test
                degree_fredom = (std_A**2/n_A + std_B**2/n_B)**2 / ((std_A**2/n_A)**2/(n_A-1) + (std_B**2/n_B)**2/(n_B-1))
                # Equivalente to: 
                # df = (std_A**2/n_A + std_B**2/n_B)**2 / (std_A**4/(n_A**2*(n_A - 1)) + std_B**4/(n_B**2*(n_B-1))) 
                # See: https://en.wikipedia.org/wiki/Welch%27s_t-test#Degrees_of_freedom

                # 99% confidence interval for the difference in means
                diff_means = mean_A - mean_B if alternative == 'greater' else mean_A - mean_B
                se_diff = np.sqrt(std_A**2/n_A + std_B**2/n_B)
                ci_lower = diff_means - stats.t.ppf(confidence, degree_fredom) * se_diff
                ci_upper = diff_means + stats.t.ppf(confidence, degree_fredom) * se_diff
                
                # Output
                if alternative == 'greater':
                    print(f"Welch's t-test results for {i} > {j}:", file=f)
                elif alternative == 'less':
                    print(f"Welch's t-test results for {i} < {j}:", file=f)
                else:
                    print(f"Welch's t-test results for {i} != {j}:", file=f)
                print(f"t-statistic: {t_stat:.4f}", file=f)
                print(f"p-value (one-tailed): {p_value:.4f}", file=f)
                print(f"99% Confidence interval for difference in means: ({ci_lower:.4f}, {ci_upper:.4f})", file=f)



def main():
    argparser = argparse.ArgumentParser(description='User Analysis Module')
    argparser.add_argument('-d', '--dataset', type=str, required=True, 
                           help='Name of the dataset to analyze')
    argparser.add_argument('-a', '--alternative', type=str, required=False, default='auto',
                           choices=["two-sided", "greater", "less", "auto"], 
                           help='''Alternative hypothesis to test. 
                           The more uncommon users has [Options] interactions than the more common. 
                           Options: "two-sided", "greater", "less", "auto" (default: "auto").
                           ''')
    argparser.add_argument('-c', '--confidence', type=float, required=False, default=0.99, 
                           help='De ault confidence level for statistical tests (default: 0.99)')
    argparser.add_argument('-u', '--user_agregation', type=str, required=False, default='median',
                           choices=["mediam", "mean", "popularity"], 
                           help='''Method to aggregate user popularity scores.
                           Options: "median", "mean", "popularity" (default: "median").
                           "popularity" is defined in: https://arxiv.org/abs/1907.13286
                           ''')
    argparser.add_argument('-g', '--groups', type=str, required=False, default='4', 
                           help='Number of groups to divide users into based on preferences or list of limits, e.g. [0, 0.5, 1] (default: 4)')
    argparser.add_argument('-o', '--output', type=str, required=False, default='results', 
                           help='Output folder for analysis results (default: "results")')
    argparser.add_argument('-f', '--config_file', type=str, required=False, default=None, 
                           help='Path to the configuration file (default: None)')
    
    args = argparser.parse_args()
    # Initialize the configuration and logger
    config = Config(model='BPR', 
                    dataset=args.dataset,
                    config_file_list=args.config_file)
    
    # Initialize the module with the configuration
    init_module(config)

    # Initialize the dataset
    print("Initializing dataset...")
    train_dataset, valid_dataset, test_dataset = init_dataset(config)

    # Convert dataset to DataFrame
    print("Converting dataset to DataFrame...")
    df = dataset_to_df(train_dataset)
    print("Grouping users by preferences...")
    groups = eval(args.groups) if args.groups.startswith('[') else int(args.groups)
    if isinstance(groups, list):
        valid = groups[0] == 0 and groups[-1] == 1
        for i in range(1, len(groups)):
            if not valid:
                break
            valid = groups[i] > groups[i-1] and groups[i] <= 1
        if not valid:
            raise ValueError("Invalid groups limits. Must be a list of increasing values between 0 and 1, e.g. [0, 0.5, 1]")
    elif isinstance(groups, int) and groups < 1:
        raise ValueError("Invalid number of groups. Must be an integer greater than 0.")
    df = group_users_by_preferences(df, groups=groups, method=args.user_agregation)
    print("Analyzing user preferences...")
    df = analyze_user(df)

    print("Generating report...")
    output_folder = os.path.join(args.output, 
                                 f"{args.dataset}_{args.user_agregation}_groups_{args.groups}_{args.confidence}")
    generate_report(df, output_folder, alternative=args.alternative, confidence=args.confidence)
    print(f"User analysis completed. Results saved to {output_folder}")

if __name__ == '__main__':
    main()