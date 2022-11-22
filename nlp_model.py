import os
import torch
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder

from fairseq.models.roberta import RobertaModel, RobertaHubInterface
from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE

from vncorenlp import VnCoreNLP

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TextClassifier():
    def __init__(self) -> None:
        self.max_seq_len = 256
        self.__load_model()
        self.__load_vocab()
        self.__load_label_encoder()
        self.__rdrsegmenter = VnCoreNLP(
            "vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size="-Xmx500m")

    def get_model(self) -> RobertaModel:
        return self.__model

    def __load_model(self):
        print('\nLoading NLP model...')
        self.__model = RobertaModel.from_pretrained(
            'PhoBERT_base_fairseq', checkpoint_file='model.pt')
        self.__model.to(DEVICE)
        self.__model.register_classification_head('new_task', num_classes=10)
        self.__model.load_state_dict(torch.load(
            os.path.join('model', 'model_3.bin'), map_location=DEVICE))

        class BPE():
            bpe_codes = 'PhoBERT_base_fairseq/bpe.codes'
        self.__model.bpe = self.__bpe = fastBPE(BPE())
        print('\nModel has loaded successfully.\n')

    def __load_vocab(self):
        print('\nLoading dictionary...')
        self.__vocab = Dictionary()
        self.__vocab.add_from_file('PhoBERT_base_fairseq/dict.txt')
        print('Dictionary has loaded')

    def __load_label_encoder(self):
        with open('model/labelEncoder.pkl', 'rb') as f:
            self.le: LabelEncoder = pickle.load(f)

    def encode(self, text) -> torch.LongTensor:
        text_tokenized = ' '.join([
            ' '.join(sent) for sent in self.__rdrsegmenter.tokenize(text)
        ])
        bpe_sentence = '<s> ' + self.__bpe.encode(text_tokenized) + ' <s>'
        tokens = self.__vocab.encode_line(
            bpe_sentence, append_eos=False, add_if_not_exist=False)
        # Index cac token cls (đầu câu), eos (cuối câu), padding (padding token)
        # cls_id = 0
        eos_id = 2
        pad_id = 1
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            tokens[-1] = eos_id
        else:
            tokens = torch.cat((tokens, torch.tensor(
                [pad_id, ] * (self.max_seq_len - len(tokens)))))
        return tokens.long()

    def predict(self, text):
        self.__model.eval()
        tokens = self.encode(text)
        logits = self.__model.predict('new_task', tokens, return_logits=True)
        y_pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
        return self.le.inverse_transform(y_pred)[0]


if __name__ == '__main__':
    text = '''
        Rụng lông mi nhiều Tôi 32 tuổi, sức khỏe bình thường. Mấy tháng gần đây lông mi ở mắt rất hay rụng. \
        Mỗi lần rửa mặt chạm trúng lông mi lại rụng vài sợi. Xin bác sĩ cho biết tôi có bệnh gì không, có ảnh \
        hưởng đến sức khỏe không? Một bạn đọc (TPHCM)- Bác sĩ Phạm Thị Bích Thủy, Bệnh viện Mắt TPHCM, trả lời: \
        Lông mi rụng nhiều và thường xuyên có thể là triệu chứng của viêm bờ mi. Bệnh có nhiều nguyên nhân như \
        nấm, mắt hột, dị ứng... Bạn nên đến cơ sở có chuyên khoa mắt để được bác sĩ khám trực tiếp, cho thuốc \
        điều trị và tư vấn cách chăm sóc vệ sinh mắt.
    '''
    classifier = TextClassifier()
    print(classifier.predict(text))
