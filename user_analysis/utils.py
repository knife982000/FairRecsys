import RecboleRunner #Needed for circular imports

from recbole.data.dataset import Dataset
from recbole.config import Config
from recbole.utils import init_seed
import seaborn as sns
import pandas as pd
from typing import Tuple

user_field = 'user_id'
item_field = 'item_id'


def init_module(config: Config) -> None:
    """
    Initializes global user and item field names from the provided configuration.

    Args:
        config (Config): The configuration object containing field names.
    """
    global user_field, item_field
    user_field = config['USER_ID_FIELD']
    item_field = config['ITEM_ID_FIELD']
    pass


def init_dataset(config: Config) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Initializes the random seed and builds the dataset using the provided configuration.

    Args:
        config (Config): The configuration object for dataset creation.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: The training, validation, and test datasets.
    """
    init_seed(config['seed'], config['reproducibility'])
    dataset = Dataset(config=config)
    return dataset.build()


def dataset_to_df(dataset: Dataset):
    """
    Converts a RecBole Dataset object into a pandas DataFrame with user and item columns.

    Args:
        dataset (Dataset): The dataset to convert.

    Returns:
        pd.DataFrame: DataFrame containing user and item interactions.
    """
    global user_field, item_field
    data = []
    for u in dataset:
        data.append((u[user_field].item(), u[item_field].item()))
    df = pd.DataFrame(data=data, columns=[user_field, item_field])
    return df


def group_users_by_preferences(df: pd.DataFrame, groups: int = 4, 
                               method: str = 'median', return_popularity=False) -> pd.DataFrame:
    """
    Groups users based on the average popularity of items they interact with.

    Args:
        df (pd.DataFrame): DataFrame with user-item interactions.
        groups (int, optional): Number of user groups to create. Defaults to 4.
        method (str): Method to calculate user popularity score ('median' or 'mean').
        return_popularity (bool, optional): Whether to return item popularity as well. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with an added 'group' column for user group assignment.
        (optional) pd.Series: Item popularity scores if return_popularity is True.
    """
    global user_field, item_field
    item_popularity = df[item_field].value_counts(normalize=True) 
    #Assign Popularity Score to Each Interaction
    df['item_popularity'] = df[item_field].map(item_popularity)
    #Score aprox interaction
    if method == 'median':
        user_pop_score = df.groupby(user_field)['item_popularity'].median()
    elif method == 'mean':
        user_pop_score = df.groupby(user_field)['item_popularity'].mean()
    else:
        raise ValueError("Invalid method. Choose 'median' or 'mean'.")
    #Compute quartiles
    user_groups = pd.qcut(user_pop_score.values, groups, labels=False)
    #Assign group to each user based on popularity score
    user_group_df = pd.DataFrame({
        user_field: user_pop_score.index,
        'popularity_score': user_pop_score,
        'group': user_groups  # or user_clusters
    })
    df = df.join(user_group_df[['group']], on="user_id")
    if return_popularity:
        return df, item_popularity
    return df


def analyze_user(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes users by calculating their average item popularity score, interaction count, and group assignment.

    Args:
        df (pd.DataFrame): DataFrame with user-item interactions and group assignments.
    Returns:
        pd.DataFrame: DataFrame summarizing each user's popularity score, interaction count, and group.
    """
    user_popularity = df[[user_field, 'item_popularity']].\
                        groupby(user_field).mean().\
                        rename(columns={'item_popularity': 'popularity_score'})
    interaction_count =  df[[user_field, item_field]].groupby(user_field).\
                                    count().rename(columns={item_field: 'interactions'})
    df = df[[user_field, 'group']].drop_duplicates()
    user_description = df.sort_values(user_field).\
                        join(user_popularity.join(interaction_count), on='user_id').\
                            reset_index().drop(columns='index')
    return user_description