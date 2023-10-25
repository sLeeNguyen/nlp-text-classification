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
            os.path.join('model', 'model_both.bin'), map_location=DEVICE))

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
        bpe_sentence = '<s> ' + self.__bpe.encode(text_tokenized) + ' </s>'
        tokens = self.__vocab.encode_line(
            bpe_sentence, append_eos=False, add_if_not_exist=False).long().tolist()

        eos_id = 2
        pad_id = 1
        cls_id = 0
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:64] + tokens[-(self.max_seq_len - 64):]
            tokens[0] = cls_id
            tokens[-1] = eos_id
        else:
            tokens = tokens + [pad_id, ] * (self.max_seq_len - len(tokens))
        return torch.LongTensor(tokens)

    def predict(self, text):
        self.__model.eval()
        tokens = self.encode(text)
        logits = self.__model.predict('new_task', tokens, return_logits=True)
        y_pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
        return self.le.inverse_transform(y_pred)[0]


if __name__ == '__main__':
    text = '''
        Thứ trưởng Văn hóa: Tặng quà Tết không nên nặng giá trị vật chất Theo bà Trịnh Thị Thủy, tặng quà dịp Tết là nét văn hóa tốt đẹp của người Việt Nam, mang giá trị tinh thần chứ không nên nặng về vật chất. Chiều 3/1, tại họp báo Chính phủ thường kỳ, Thứ trưởng Văn hóa Thể thao và Du lịch Trịnh Thị Thủy nêu quan điểm, Tết luôn có giá trị thiêng liêng trong tâm thức người dân Việt Nam. Truyền thống tặng quà, biếu quà dịp Tết thể hiện tấm lòng thành kính với bề trên, người cao tuổi, người giúp đỡ mình trong cuộc sống. Thời gian tới, Bộ Văn hóa Thể thao và Du lịch sẽ tăng cường truyền thông để người dân nhận thức đúng giá trị của văn hóa tặng quà Tết, từ đó thực hành đúng. Ngày 23/12/2022, Thủ tướng ban hành chỉ thị về tăng cường các biện pháp đón Tết Nguyên đán Quý Mão 2023 vui tươi, lành mạnh, an toàn, tiết kiệm. Các cơ quan trong hệ thống hành chính nhà nước thực hiện nghiêm việc không tổ chức đi thăm, chúc Tết cấp trên và lãnh đạo các cấp; không tổ chức đoàn của Trung ương thăm, chúc Tết cấp ủy, chính quyền các tỉnh, thành phố; nghiêm cấm biếu, tặng quà Tết cho lãnh đạo các cấp dưới mọi hình thức; không sử dụng ngân sách nhà nước, phương tiện, tài sản công trái quy định; không tham gia các hoạt động mê tín, dị đoan; chỉ dự lễ chùa, lễ hội khi được phân công. Người đứng đầu Chính phủ yêu cầu các địa phương tổ chức tốt việc thăm hỏi, chúc Tết thương binh, gia đình liệt sĩ, gia đình có công với nước, cán bộ lão thành cách mạng, mẹ Việt Nam anh hùng, nhân sĩ, trí thức, người có uy tín trong đồng bào dân tộc thiểu số, chức sắc tôn giáo tiêu biểu, các đơn vị lực lượng vũ trang, lực lượng thường trực làm nhiệm vụ trong ngày Tết, nhất là ở biên giới, hải đảo, vùng khó khăn. Bên cạnh đó, các bộ ngành, địa phương phải thực hiện tốt chính sách an sinh, xã hội, phát huy truyền thống đại đoàn kết, tinh thần tương thân, tương ái của dân tộc; chăm lo đời sống của vật chất và tinh thần của nhân dân, bảo đảm mọi người đều có điều kiện vui xuân, đón Tết.
    '''
    classifier = TextClassifier()
    print(classifier.predict(text))
