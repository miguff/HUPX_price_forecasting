import pandas as pd

def load_data(synt:str, country: str):

    match synt:
        case "real":
            all_data_set = pd.read_csv(f"processed_data//{country}//Processed_data_real.csv", index_col=0)
        case "lgbm":
            all_data_set = pd.read_csv(f"processed_data//{country}//Processed_data_all.csv", index_col=0)
        case "intra":
            all_data_set = pd.read_csv(f"processed_data//{country}//Intra_Pattern_Processed_data_all.csv", index_col=0)
        case "spline":
            all_data_set = pd.read_csv(f"processed_data//{country}//Spline_Processed_data_all.csv", index_col=0)
    

    return all_data_set