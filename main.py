from utils.data_utils import load_data_v2

if __name__ == "__main__":
    df_dataset = load_data_v2('train')
    df_dataset_2 = load_data_v2('validation')
    df_dataset_3 = load_data_v2('test')

