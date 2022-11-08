{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyOByzfhJw3eJIbYEidwKR8S",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/bh8648/Deap-Learning/blob/main/model.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "<학습자료 8-1>\n",
        "\n",
        "nn.Module을 상속받아 모델 클래스를 정의함\n",
        "\n",
        "1) Blcok 모델\n",
        "\n",
        "2) ImageClassificer 모델\n"
      ],
      "metadata": {
        "id": "LPknGWfxA9wG"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# model.py : Block 모델\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "\n",
        "class Block(nn.Module) :\n",
        "  def __init__(self, input_size, output_size, use_batch_norm=True, dropout_p=.4) :\n",
        "    self.input_size = input_size\n",
        "    self.output_size = output_size\n",
        "    self.use_batch_norm = use_batch_norm\n",
        "    self.dropout_p = dropout_p\n",
        "\n",
        "    super().__init__()\n",
        "\n",
        "    def get_regularizer(use_batch_norm, size) :\n",
        "      return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)\n",
        "    \n",
        "    self.block = nn.Sequential(\n",
        "        nn.Linear(input_size, output_size),\n",
        "        nn.LeakyReLU(),\n",
        "        get_regularizer(use_batch_norm, output_size)\n",
        "    )\n",
        "\n",
        "    def forward(self, x) :\n",
        "      # |x| = (batch_szie, input_size)\n",
        "      y = self.block(x)\n",
        "      # |y| = (batch_szie, input_size)\n",
        "\n",
        "      return y"
      ],
      "metadata": {
        "id": "EL4-GIUfA1iH"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "class ImageClassifier(nn.Module) :\n",
        "  def __init__(self, input_size, output_size, hidden_sizes=[500,400,300,200,100], use_batch_norm=True, dropout_p=.3) :\n",
        "\n",
        "    super().__init__()\n",
        "\n",
        "    assert len(hidden_sizes) > 0, \"You need to specify hidden layers\"\n",
        "    # assert 키워드 뒤에 [조건]을 입력하고 그 뒤에 콤마(,) [오류메시지]를 입력합니다.\n",
        "    # 이 assert는 [조건]이 True인 경우 그대로 코드 진행, False인 경우 어설트에러를 발생하게 됩니다.\n",
        "\n",
        "    last_hidden_size = input_size\n",
        "    blocks = []\n",
        "    for hidden_size in hidden_sizes :\n",
        "      blocks += [Block(\n",
        "        last_hidden_size,\n",
        "        hidden_size,\n",
        "        use_batch_norm,\n",
        "        dropout_p  \n",
        "      )]\n",
        "      last_hidden_size = hidden_size\n",
        "\n",
        "    self.layers = nn.Sequential(\n",
        "        *blocks,\n",
        "        nn.Linear(last_hidden_size, output_size),\n",
        "        nn.LogSoftmax(dim=-1)\n",
        "    )\n",
        "\n",
        "  def forward(self, x) :\n",
        "    # |x| = (batch_szie, input_size)\n",
        "    y = self.layers(x)\n",
        "    # |y| = (batch_szie, input_size)\n",
        "\n",
        "    return y\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "AWNVlYEzEg6B"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}