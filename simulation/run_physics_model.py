import os
from matplotlib import pyplot as plt
from pyFM.interface import DLL
from pyFM.operation import Operation
from tqdm import tqdm
import argparse

plt.rcParams["axes.grid"] = True 

def run_instance(run_folder, operations_file, operations_meta):
    """
    Run the DLL model on the input in the specified run folder.
    """
    dll = DLL(run_folder, "Flowmodel.dll")
    op = Operation(os.path.join(run_folder, operations_file),
                   os.path.join(run_folder, operations_meta))
    
    pbar = tqdm(total=len(op), desc="Processing Operations", leave=True)
    for t in range(len(op)):
        pbar.update()
        
        input_table = op.get_RT_table(t)
        dll.setmoduletable("Input_RealTime", input_table)
        dll.runflowmodel()
    
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DLL model with operations.")
    parser.add_argument("run_folder", type=str, help="Path to the folder containing the DLL and input files.")
    parser.add_argument("operations_file", type=str, help="Name of the operations file.")
    parser.add_argument("operations_meta", type=str, help="Name of the operations meta file.")
    
    args = parser.parse_args()
    run_instance(args.run_folder, args.operations_file, args.operations_meta)
