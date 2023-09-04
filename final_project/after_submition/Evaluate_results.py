import pandas as pd
import os
import shutil

optimizers = ['SGD', 'Adam', 'AdamW']
learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
momentums = [0.1, 0.5, 0.9]
results_base_path = os.path.join("/home/liranc6/Using_YOLOv8/runs/detect/")
dest_base_path = "/home/liranc6/Using_YOLOv8/results"

"""
reads the results from path and returns a dict of dataframes when key=optimizer, val=df
"""
def create_results_data_frames():
    df_dict = {}
    # loop through the optimizers
    for optimizer in optimizers:
        # create an empty data frame
        optimizer_df = pd.DataFrame(columns=momentums, index=learning_rates)

        optimizer_df = optimizer_df.fillna(0)

        for learning_rate in learning_rates:
            mAP_score = 0
            for momentum in momentums:
                # name path
                folder_name = f"optimizer={str(optimizer)}_learning_rate={str(learning_rate)}_momentum={str(momentum)}"

                full_folder_name = os.path.join(results_base_path, folder_name)
                        
                #get mAP50-95 results
                result_csv_path = os.path.join(full_folder_name, 'results.csv')
                if os.path.exists(result_csv_path):
                    df = pd.read_csv(result_csv_path)
                    mAP_score = df['    metrics/mAP50-95(B)'].iloc[-1]
                    optimizer_df.loc[learning_rate, momentum] = mAP_score

        # add the data frame to the dictionary with the optimizer name as the key
        df_dict[optimizer] = optimizer_df

    return df_dict

def print_df(df_dict):
    for optimizer, results in df_dict.items():
        print(optimizer + ":")
        print(results)
        print("\n\n")

def create_results_csv(df_dict):
    with open("results.csv", "w") as f:
        for op in optimizers:
            f.write(str(op) + ":\n")
            df_dict[op].to_csv(f, mode='a')
            # write two new lines after each DataFrame
            f.write("\n\n")

def find_best_params(df_dict):
    best_params = ""
    max = -1
    for op in optimizers:
        df = df_dict[op]
        # get the row and column labels of the maximum value
        max_row_label, max_col_label = df.stack().idxmax()
        # get the value of the maximum cell
        max_val = df.loc[max_row_label, max_col_label]
        if max_val>max:
            best_params = "best params are: " + str(op) + " learning_rates=" + str(max_row_label) + " momentum=" + str(max_col_label)\
                        + "max_val_after_5_ephocs=" + str(max_val)
            
    return best_params

def print_best_params(best_params):
    with open("results.txt", "a") as f:
        f.write(best_params)
    print(best_params)


def copy_results_csv_to_new_folder(dest_base_path):
    # Create the directory
    os.makedirs(dest_base_path, exist_ok=True)
    # loop through the optimizers
    for optimizer in optimizers:
        for learning_rate in learning_rates:
            for momentum in momentums:
                #rename folder
                folder_name = f"optimizer={str(optimizer)}_learning_rate={str(learning_rate)}_momentum={str(momentum)}"
                new_name = os.path.join(results_base_path, folder_name)

                #get mAP50-95 results
                result_csv_path = os.path.join(new_name, 'results.csv')
                if os.path.exists(result_csv_path):
                    # Set the source and destination paths
                    src_path = result_csv_path

                    dest_full_path = os.path.join(dest_base_path, folder_name+'.csv')

                    # Copy the file to the destination folder
                    shutil.copy(src_path, dest_full_path)
                else:
                    print("no folder found")

if __name__ == "__main__":
    df_dict = create_results_data_frames()

    create_results_csv(df_dict)

    copy_results_csv_to_new_folder(dest_base_path)

    print_df(df_dict)

    create_results_csv(df_dict)

    best_params = find_best_params(df_dict)

    print_best_params(best_params)



