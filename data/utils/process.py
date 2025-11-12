import pandas as pd

def normalise_timestamps(df: pd.DataFrame, time_column: str, sample_rate: int) -> pd.DataFrame:
  """
  Both datasets are sub-second sampled with some gaps (e.g 1, 1, 1, 1, 2, 2, 3, 5, 5, 5).
  Normalizes timestamps to regular intervals based on sample_rate (in seconds).
  """
  return df.groupby(time_column).mean().reset_index()


def main():
  df = pd.read_csv('data/carOBD/obdiidata/drive1.csv')
  columns = df.columns.tolist()
  print(columns)
  df = normalise_timestamps(df, 'ENGINE_RUN_TINE ()', 1)
  print(df.head(25))

if __name__ == '__main__':
  main()