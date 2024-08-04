import torch
import transformers
from model import Transformer

tokenizers = transformers.AutoTokenizer.from_pretrained("quantumaikr/KoreanLM")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MAX_LEN =128

class Translator():
    def __init__(self, tokenizers, transformer):
        self.tokenizers = tokenizers
        self.transformer = transformer
        transformer.eval()

    def __call__(self, sentence):

        sentence = tokenizers.encode(sentence, padding="max_length",add_special_tokens=False,  max_length=MAX_LEN,truncation=True, return_tensors="pt")

        encoder_input = sentence.to(device=device)

        # as the target is english, the first token to the transformer should be the
        # english start token.
        start = torch.asarray([[tokenizers.bos_token_id]], device=device)
        end = torch.asarray([[tokenizers.eos_token_id]], device=device)

        # `tf.TensorArray` is required here (instead of a python list) so that the
        # dynamic-loop can be traced by `tf.function`.
        output = start

        for _ in range(MAX_LEN-1):
            predictions = self.transformer(encoder_input, output)
            # select the last token from the seq_len dimension
            predictions = torch.nn.functional.log_softmax(predictions, dim=1)  # (batch_size, vocab_size, 1)
            predictions = predictions[:, :, -1:]
            predicted_id = torch.argmax(predictions, axis=1)

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = torch.concat([output, predicted_id], axis=1)


            if predicted_id == end:
                break

        output = output.squeeze()

        return output
  
def print_translation(sentence, tokens):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens}')

    
if __name__ == "__main__":

    with torch.no_grad():
        model = Transformer(256 , 512, tokenizers.vocab_size, MAX_LEN, device).to(device=device)
        model.load_state_dict(torch.load("./Translator/model-30.pt"))
        model.eval()

        ts = Translator(tokenizers, model)

        sentence = input("입력 >> ")
        
        text = ts(sentence)

        text = tokenizers.decode(text)

        print_translation(sentence,text)