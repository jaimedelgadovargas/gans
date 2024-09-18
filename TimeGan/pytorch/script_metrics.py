import numpy as np
from metrics.discriminative_score_metrics import discriminative_score_metrics
from metrics.predictive_score_metrics import predictive_score_metrics
from readsettings import ReadSettings
import sys
def main():
    args = sys.argv[1:]
    print(args)
    data = ReadSettings(args[0])
    dataset_path = data["paths"]["dataset"]
    path_output = data["paths"]["output"]
    
    seq_len = data["model_parameters"]["seq_len"]
    n_seq = data["model_parameters"]["n_seq"]
    hidden_dim = data["model_parameters"]["hidden_dim"]
    gamma = data["model_parameters"]["gamma"]
    
    noise_dim = data["model_parameters"]["noise_dim"]
    dim = data["model_parameters"]["dim"]
    batch_size = data["model_parameters"]["batch_size"]
    
    log_step = data["model_parameters"]["log_step"]
    learning_rate = data["model_parameters"]["learning_rate"]
    train_steps = data["model_parameters"]["train_steps"]

    sample_size = data["model_parameters"]["sample_size"]
    gan_args = batch_size, learning_rate, noise_dim, 24, 2, (0, 1), dim
    #-----------------------------------------------------------------------------------------
    synthetic_sample=np.load(path_output+"synthetic_samplenpy.npy")
    real_sample=np.load(path_output+"real_samplenpy.npy")
    
    
    _, batch_size, _ = np.asarray(real_sample).shape
    metric_results = dict()
    metric_results['discriminative']  = discriminative_score_metrics(real_sample, synthetic_sample,batch_size)
    metric_results['predictive'] = predictive_score_metrics(real_sample, synthetic_sample,batch_size)
    print("Discriminative: %f +/- %f"%(metric_results['discriminative'][0],metric_results['discriminative'][1]))
    print("Predictive: %f +/- %f"%(metric_results['predictive'][0],metric_results['predictive'][1]))

if __name__ == '__main__':
    data = ReadSettings("settings.json")
    main()


    
    


    

#Full credits goes to [Fabiana Clemented](https://towardsdatascience.com/synthetic-time-series-data-a-gan-approach-869a984f2239) for this implementation.<br>
#Paper on [TimeGAN](https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf)
    
