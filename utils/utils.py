import pandas as pd

def load_data(synt:str):

    match synt:
        case "real":
            all_data_set = pd.read_csv(r"processed_data//Processed_data_real.csv", index_col=0)
        case "lgbm":
            all_data_set = pd.read_csv(r"processed_data//Processed_data_all.csv", index_col=0)
        case "intra":
            all_data_set = pd.read_csv(r"processed_data//Intra_Pattern_Processed_data_all.csv", index_col=0)
        case "spline":
            all_data_set = pd.read_csv(r"processed_data//Spline_Processed_data_all.csv", index_col=0)
    

    return all_data_set