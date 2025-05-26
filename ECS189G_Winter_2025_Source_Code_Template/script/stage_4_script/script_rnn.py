from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN import Method_RNN
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

#---- RNN script for both classification and generation ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('train', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/'
    
    # Choose task: 'text_classification' or 'text_generation'
    task = 'text_generation'  # Change this to switch tasks
    data_obj.dataset_source_file_name = task

    method_obj = Method_RNN('recurrent neural network', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/RNN_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    
    if task == 'text_classification':
        # Run classification task with train-test split
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        setting_obj.print_setup_summary()
        accuracy, _ = setting_obj.load_run_save_evaluate()
        print('************ Overall Performance ************')
        print('RNN Classification Accuracy:', accuracy)
    else:
        # For generation task, we don't need train-test split
        data_obj.load()
        method_obj.data = data_obj.data
        result = method_obj.run()
        
        # Generate text with three starting words
        start_words = ['once', 'upon', 'a']
        generated_text = result['model'].generate_text(start_words)
        
        print('\n************ Generated Text ************')
        print('Starting words:', ' '.join(start_words))
        print('Generated text:', generated_text)
    
    print('************ Finish ************')
    # ------------------------------------------------------
    

    