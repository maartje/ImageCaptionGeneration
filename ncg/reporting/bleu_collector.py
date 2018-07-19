from nltk.translate.bleu_score import corpus_bleu
from ncg.nn.predict import predict

class BleuCollector:

    def __init__(self, image_features_loader, references, text_mapper, max_length):
        self.image_features_loader = image_features_loader
        self.SOS_index = text_mapper.SOS_index()
        self.max_length = max_length
        self.text_mapper = text_mapper
        self.references = self.reorder_reference_sentences(references)
        self.bleu_val = []
        
    def on_epoch_completed(self, epoch, trainer):
        predicted_indices = predict(
            trainer.decoder, self.image_features_loader, self.SOS_index, self.max_length)
        cleaned_predicted_indices = [
            self.text_mapper.remove_predefined_indices(s) for s in predicted_indices
        ]
        bleu = corpus_bleu(self.references, cleaned_predicted_indices)
        self.bleu_val.append(bleu)
        
        self.cleaned_predicted_indices = cleaned_predicted_indices
        
        
    def reorder_reference_sentences(self, references_by_file):
        references_by_sentence = [[ s[1:-1] for s in tup] for tup in zip(*references_by_file)]
        return references_by_sentence
        
        
    
