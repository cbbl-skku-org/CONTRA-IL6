from .architecture import CONTRA_IL6
import torch
from torch.nn.functional import softmax, pad
import os
from .feature_extractor import ESM, ProtTrans, SeqVec
from Bio import SeqIO
from tqdm import tqdm

class CONTRA_IL6_Predictor:
    def __init__(self, model_config, ckpt_dir, nfold, device):
        self.models = [CONTRA_IL6(**model_config) for _ in range(nfold)]
        self.device = device
        
        for ckpt_file, model in zip(sorted(os.listdir(ckpt_dir)), self.models):
            if ckpt_file.endswith('.pth'):
                checkpoint_path = os.path.join(ckpt_dir, ckpt_file)
                self._load_model_from_checkpoint(checkpoint_path, model)
        
        
    def _load_model_from_checkpoint(self, checkpoint_path, model):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint, strict=False)
        model.to(self.device)
        model.eval()
        
        
    def predict_one(self, f1, f2, f3, f4, threshold=0.46):
        all_probs = []
        for model in self.models:
            with torch.no_grad():
                output = model([f1, f2, f3, f4], None)
                prob = softmax(output, dim=-1)[:, 1]
                all_probs.append(prob.item())

        mean_prob = sum(all_probs) / len(all_probs)
        return 1 if mean_prob >= threshold else 0, mean_prob
        
    def __call__(self, f1s, f2s, f3s, f4s, threshold=0.46):
        all_preds = []
        all_probs = []
        
        for f1, f2, f3, f4 in zip(f1s, f2s, f3s, f4s):
            pred, prob = self.predict_one(f1, f2, f3, f4, threshold)
            all_preds.append(pred)
            all_probs.append(prob)
        
        return all_preds, all_probs
    
    
class CONTRA_IL6_Inferencer:
    def __init__(self, predictor, device='cpu'):
        self.predictor = predictor
        self.esm_1 = ESM(model_name='esm1_t34_670M_UR50S', device=device)
        self.esm_2 = ESM(model_name='esm2_t36_3B_UR50D', device=device)
        self.prot_t5 = ProtTrans(model_name='Rostlab/prot_t5_xl_uniref50', device=device)
        self.seqvec = SeqVec()
        
    @staticmethod
    def read_fasta_file(fasta_file):
        data_dict = {}
        for record in SeqIO.parse(fasta_file, 'fasta'):
            assert record.id not in data_dict, f'Duplicated ID: {record.id}'
            data_dict[record.id] = str(record.seq)
        return data_dict
    
    @staticmethod
    def write_output_file(output_dict, output_file):
        with open(output_file, 'w') as f:
            f.write('ID\tLabel\tProbability\n')
            for key, (label, prob) in output_dict.items():
                f.write(f'{key}\t{label}\t{prob:.2f}\n')
    
    def predict_sequences(self, data_dict, threshold=0.46, batch_size=4):
        keys = list(data_dict.keys())
        seqs = list(data_dict.values())
        total_batch_len = (len(seqs) // batch_size) + int(len(seqs) % batch_size == 0)
        esm_1_generator = self.esm_1.get_features_batch(seqs, batch_size=batch_size)
        esm_2_generator = self.esm_2.get_features_batch(seqs, batch_size=batch_size)
        prot_t5_generator = self.prot_t5.get_features_batch(seqs, batch_size=batch_size)
        seqvec_generator = self.seqvec.get_features_batch(seqs, batch_size=batch_size)
        
        ALL_LABELS = []
        ALL_PROBS = []
        for esm_2_features, esm_1_features, seqvec_features, prot_t5_features in tqdm(zip(
            esm_2_generator, esm_1_generator, seqvec_generator, prot_t5_generator
        ), total=total_batch_len, desc='Predicting'):
            # Padding to ensure all features have the same length 30, using pad, the shape of feature is (batch_size, seq_len, feature_dim
            esm_2_features = [pad(f, (0, 0, 0, 30 - f.shape[1]), value=0) for f in esm_2_features]
            esm_1_features = [pad(f, (0, 0, 0, 30 - f.shape[1]), value=0) for f in esm_1_features]
            seqvec_features = [pad(f, (0, 0, 0, 30 - f.shape[1]), value=0) for f in seqvec_features]
            prot_t5_features = [pad(f, (0, 0, 0, 30 - f.shape[1]), value=0) for f in prot_t5_features]
            
            labels, probs = self.predictor(
                esm_2_features, esm_1_features, seqvec_features, prot_t5_features, threshold=threshold
            )
            ALL_LABELS.extend(labels)
            ALL_PROBS.extend(probs)
        
        
        return {key: [label, prob] for key, label, prob in zip(keys, ALL_LABELS, ALL_PROBS)}
    

# if __name__ == "__main__":
#     import yaml
    
#     config = yaml.safe_load(open('configs/top_4_features.yaml'))
#     model_confg = config['model']
#     ckpt_path = '/data/GitHub_Repo/CONTRA-IL6/checkpoints/final_ckpts'
#     predictor = CONTRA_IL6_Predictor(model_confg, ckpt_path, nfold=10, device='cpu')
#     inferencer = CONTRA_IL6_Inferencer(predictor)
    
#     test_pos_seqs = inferencer.read_fasta_file('data/Validate_negative.txt')
    
#     output = inferencer.predict_sequences(test_pos_seqs, threshold=0.46)
    
#     neg_cnt, pos_cnt = 0, 0
#     for key, (label, prob) in output.items():
#         if label == 1:
#             pos_cnt += 1
#         else:
#             neg_cnt += 1
    
#     print(f'Positive: {pos_cnt}, Negative: {neg_cnt}')