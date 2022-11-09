# <학습자료 8-1>
#
# nn.Module을 상속받아 모델 클래스를 정의함
#
# 1) Blcok 모델
#
# 2) ImageClassificer 모델


# model.py : Block 모델
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, input_size, output_size, use_batch_norm=True, dropout_p=.4):
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p

        super().__init__()

        def get_regularizer(use_batch_norm, size):
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)

        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            get_regularizer(use_batch_norm, output_size)
        )

        def forward(self, x):
            # |x| = (batch_szie, input_size)
            y = self.block(x)
            # |y| = (batch_szie, input_size)

            return y



# model.py : ImageClassifier 모델
# 심층신경망 역할
class ImageClassifier(nn.Module) :
  def __init__(self, input_size, output_size, hidden_sizes=[500,400,300,200,100], use_batch_norm=True, dropout_p=.3) :

    super().__init__()

    assert len(hidden_sizes) > 0, "You need to specify hidden layers"
    # assert 키워드 뒤에 [조건]을 입력하고 그 뒤에 콤마(,) [오류메시지]를 입력합니다.
    # 이 assert는 [조건]이 True인 경우 그대로 코드 진행, False인 경우 어설트에러를 발생하게 됩니다.

    last_hidden_size = input_size
    blocks = []
    for hidden_size in hidden_sizes :
      blocks += [Block(
        last_hidden_size,
        hidden_size,
        use_batch_norm,
        dropout_p
      )]
      last_hidden_size = hidden_size

    self.layers = nn.Sequential(
        *blocks,
        nn.Linear(last_hidden_size, output_size),
        nn.LogSoftmax(dim=-1)
    )

  def forward(self, x) :
    # |x| = (batch_szie, input_size)
    y = self.layers(x)
    # |y| = (batch_szie, input_size)

    return y





