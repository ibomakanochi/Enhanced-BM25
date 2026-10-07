import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

class OfflineDocExpander:
    def __init__(self, model_name="doc2query/msmarco-t5-base-v1"):
        """
        Loads the sequence-to-sequence model used for predicting search queries.
        This is executed entirely offline prior to indexing.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Doc2Query model on {self.device.upper()}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def generate_synthetic_queries(self, document_text, num_queries=3):
        """
        Reads a document and predicts 'num_queries' synthetic questions users 
        might ask to find this document, bridging specialized jargon to layman terms.
        """
        if not document_text or not document_text.strip():
            return ""
        
        # Cap length to prevent memory overflows on massive documents
        input_ids = self.tokenizer.encode(
            document_text, 
            return_tensors='pt', 
            max_length=512, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=64,
                do_sample=True,
                top_k=10,
                num_return_sequences=num_queries
            )
        
        queries = [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        return " ".join(queries)

# Example usage for offline preprocessing:
# expander = OfflineDocExpander()
# corpus[doc_id]["expanded_queries"] = expander.generate_synthetic_queries(corpus[doc_id]["text"])