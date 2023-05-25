import numpy as np
import pandas as pd

predicted_labels = np.load("pred_seq2seq.npy")
##print(data)



##predicted_labels = predicted_labels.detach().cpu().numpy()
index_list = []
for i in range(0,len(predicted_labels)):
    index_list.append(f"S{i}")
    
prediction = pd.DataFrame(columns=['ID', 'label'])

prediction["ID"] = index_list
prediction["label"] = predicted_labels

prediction = prediction.reset_index(drop=True)

prediction.to_csv('20221119_하준서_sent_class.pred.csv', index = False)

#index_list
prediction
