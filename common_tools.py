import pandas as pd
import numpy as np

import json
import openai
import time

import re

from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr, kendalltau


pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.6f}'.format


MODEL_GPT3 = "gpt-3.5-turbo"
MODEL_GPT4 = "gpt-4"
MODEL_DAVINCI3 = "text-davinci-003"


API_PRICING_INPUT={
    MODEL_GPT3: 0.0015,
    MODEL_GPT4: 0.03,
    MODEL_DAVINCI3: 0.02
}

API_PRICING_OUTPUT={
    MODEL_GPT3: 0.002,
    MODEL_GPT4: 0.06,
    MODEL_DAVINCI3: 0.02
}

API_KEYS_FILE="../api_keys_20230324.json"
API_KEYS_FILE_2="../api_keys_20230612.json"


#
# Queries LLM evaluation definitions
#

EVALUATION_SYSTEM_ROLE_REV_2_9 = {
    'role': "system", 
    'content': "Você avalia se uma passagem de texto responde a uma pergunta, indicando uma pontuação de 0 à 2, onde 0 indica que a passagem não responde a pergunta; 1 indica que a passagem responde parcialmente a pergunta; e 2 que a passagem responde a pergunta de forma clara e completa. Sua resposta deve ser JSON, com o primeiro campo \"Razão\", explicando seu raciocínio, e segundo campo \"Pontuação\""
}



EVALUATION_SYSTEM_ROLE_REV_2_8 = {
    'role': "system", 
    'content': "Você avalia se uma passagem de texto é relevante para responder a uma pergunta, indicando uma pontuação de 0 à 2, onde 0 indica que a passagem não é relevante e não deveria ser relacionada à pergunta; 1 indica que a passagem é relevante, respondendo parcialmente a pergunta, mas contendo outro conteúdo não relevante; e 2 indica que a passagem é altamente relevante, respondendo claramente à pergunta. Sua resposta deve ser JSON, com o primeiro campo \"Razão\", explicando seu raciocínio, e segundo campo \"Pontuação\""
}



EVALUATION_SYSTEM_ROLE_REV_2_7 = {
    'role': "system", 
    'content': "Você avalia se uma passagem de texto responde a uma pergunta, indicando uma pontuação de 0 à 2, onde 0 indica que a passagem não está relacionada e não responde a pergunta; 1 indica que a passagem responde parcialmente a pergunta ou responde mas inclui muita informação não relacionada; e 2 que a passagem responde a pergunta de forma clara e direta, sem conter informações não relacionadas. Sua resposta deve ser JSON, com o primeiro campo \"Razão\", explicando seu raciocínio, e segundo campo \"Pontuação\""
}



EVALUATION_SYSTEM_ROLE_REV_2_6 = {
    'role': "system", 
    'content': "Task description:\nEvaluate how adequate a text passage is to answer a question, indicating a score from 0 to 3 for the passage adequacy.\n\nEvaluation criteria:\npassage adequacy: how clearly the passage answers to the question; adequate passage includes only relevant information which clarifies or enriches the answer; adequate passage does not include unnecessary or unrelated information, or any clutter (e.g. strange characters, extensive list of items)  that might hinder the answer comprehension.\n\nEvaluation Steps:\n1. Read the question carefully: Understand what the question is asking. Identify the key points or information that the answer should contain.\n\n2. Read the text passage: Read the passage thoroughly to understand its content. Pay attention to the details and the information provided in the passage.\n\n3. Compare the question and the passage: Compare the information in the passage with the key points identified in the question. Check if the passage contains the necessary information to answer the question.\n\n4. Evaluate the relevance of the information: Determine if the information in the passage is relevant to the question. The passage should not contain unnecessary or unrelated information that does not contribute to answering the question.\n\n5. Check for clarity: The passage should be clear and easy to understand. It should not contain any clutter such as strange characters or an extensive list of items that might hinder the comprehension of the answer.\n\n6. Score the passage: Based on the evaluation, score the passage on a scale of 0 to 3. A score of 0 indicates that the passage does not answer the question at all. A score of 1 indicates that the passage provides some information but it is not adequate to answer the question. A score of 2 indicates that the passage adequately answers the question but may contain some unnecessary information. A score of 3 indicates that the passage perfectly answers the question, providing only relevant information and no clutter.\n\nExpected JSON output:\n{\"Razão\": \"<>\", \"Pontuação\":  <>}"
}



EVALUATION_SYSTEM_ROLE_REV_2_5 = {
    'role': "system", 
    'content': "Task description:\nEvaluate whether a passage of text answers a question, indicating a score from 0 to 3 for the passage adequacy.\nEvaluation criteria:\npassage adequacy: how clearly the passage answers to the question, including only relevant information which clarifies or enriches the answer, avoiding unnecessary, unrelated information, or anything that might hinder the answer comprehension.\n\nEvaluation Steps:\n1. Read the question carefully: Understand what the question is asking. Identify the key points or information that the question is seeking.\n\n2. Read the passage: Read the passage thoroughly. Try to understand the main idea, the details, and the overall context of the passage.\n\n3. Compare the question and the passage: Compare the information in the passage with the question. Look for any information in the passage that directly answers the question.\n\n4. Evaluate the relevance of the information: Determine if the information in the passage that answers the question is relevant and directly related to the question. If the passage contains unnecessary or unrelated information, it may not be a good answer to the question.\n\n5. Evaluate the clarity of the information: Determine if the information in the passage is clear and easy to understand. If the passage is confusing or difficult to understand, it may not be a good answer to the question.\n\n6. Evaluate the completeness of the information: Determine if the passage fully answers the question. If the passage only partially answers the question, it may not be a good answer.\n\n7. Score the passage: Based on your evaluation, give the passage a score from 0 to 3. A score of 0 means the passage does not answer the question at all.\n\nExpected JSON output:\n{\"Razão\": \"<>\", \"Pontuação\":  <>}"
}



EVALUATION_SYSTEM_ROLE_REV_2_4 = {
    'role': "system", 
    'content': "Task description:\nEvaluate whether a passage of text answers a question, indicating a score from 0 to 3, where 0 = passage is inappropriate for the question; 1 = passage partially answers the question; 2 = passage answers the question but does not contain all the necessary information; and 3 = passage contains all the information to answer the question.\n\nEvaluation Steps:\n1. Read the question carefully: Understand what the question is asking. Identify the key points or information that the answer should contain.\n\n2. Read the passage: Read the passage thoroughly. Try to understand the main idea and the details provided in the passage.\n\n3. Compare the question and the passage: Compare the information in the passage with the requirements of the question. Identify if the passage contains the necessary information to answer the question.\n\n4. Score the passage: Based on your comparison, score the passage. If the passage is inappropriate for the question, score it 0. If the passage partially answers the question, score it 1. If the passage answers the question but does not contain all the necessary information, score it 2. If the passage contains all the information to answer the question, score it 3.\n\nExpected JSON output:\n{\"Razão\": \"<>\", \"Pontuação\":  <>}"
}



EVALUATION_SYSTEM_ROLE_REV_2_3 = {
    'role': "system", 
    'content': "Você avalia se uma passagem de texto responde a uma pergunta, atribuindo uma pontuação gradual de 0 a 3, onde 0 é passagem totalmente inadequada para a pergunta e 3 é passagem contém todas as informações para responder a pergunta”. Sua resposta deve ser JSON, com o primeiro campo \"Razão\", explicando seu raciocínio, e segundo campo \"Pontuação\""
}



EVALUATION_SYSTEM_ROLE_REV_2_2 = {
    'role': "system", 
    'content': "Você avalia se uma passagem de texto responde a uma pergunta, indicando uma pontuação de 0 à 3, onde 0 = passagem é inadequada para a pergunta; 1 = passagem responde parcialmente a pergunta; 2 = passagem responde a pergunta mas não contém todas as informações necessárias; e 3 = passagem contém todas as informações para responder a pergunta. Sua resposta deve ser JSON, com o primeiro campo \"Razão\", explicando seu raciocínio, e segundo campo \"Pontuação\""
}



EVALUATION_SYSTEM_ROLE_REV_2_1 = {
    'role': "system", 
    'content': "Você avalia se uma passagem de texto responde a uma pergunta, indicando uma pontuação de 0 à 3, onde 0 indica que a passagem não está relacionada com a pergunta; 1 indica que a passagem está no tema mas não responde a pergunta; 2 indica que a passagem responde parcialmente a pergunta ou responde mas inclui muita informação não relacionada; e 3 que a passagem responde a pergunta de forma clara e direta, sem conter informações não relacionadas. Sua resposta deve ser JSON, com o primeiro campo \"Razão\", explicando seu raciocínio, e segundo campo \"Pontuação\""
}



EVALUATION_FEW_SHOT_EXAMPLES_REV_2_9=[
    [
        {
            'role': "user",
            'content': "Passagem: \"Vanessa Redgrave nasceu em 30 de janeiro de 1937 em Londres. Ela fará falta, mas não será esquecida. Centenas de fãs imediatamente começaram a escrever suas mensagens de condolências na página do Facebook, expressando sua tristeza pela morte da talentosa atriz de 79 anos.\"\nPergunta: \"quantos anos tem vanessa redgrave?\"",
        },
        {
            'role': "assistant",
            'content': "{\"Razão\":\"A passagem responde a idade que Vanessa Redgrave tinha quando faleceu, e que ela nasceu em 1937. Com essas informações é possível determinar que ela morreu em 2016 e ter toda informação completa\",\"Pontuação\":2}"
        }
    ],

    [
        {
            'role': "user",
            'content': "Passagem: \"O Brasil possui muitas belezas naturais. Neste artigo vamos indicar os melhores lugares para passear no Brasil.\"\nPergunta: \"Onde passear no Brasil?\"",
        },
        {
            'role': "assistant",
            'content': "{\"Razão\":\"A passagem apenas indica que o Brasil tem muitas belezas naturais, mas não lista nenhum exemplo. Embora a passagem indique que artigo vai falar sobre lugares para passear no Brasil, o trecho apresentado não lista nenhum lugar específico para passear no Brasil\",\"Pontuação\":0}"
        }
    ],

    [
        {
            'role': "user",
            'content': "Passagem: \"conveniente no bairro mais moderno da cidade, o Radisson Blu Belo Horizonte, Savassi é um dos hotéis cinco estrelas mais desejados do Sudeste do Brasil. Passe o dia comprando produtos de marcas famosas no Pátio Savassi ou visitando a Pampulha, um Patrimônio Mundial da UNESCO maravilhosamente preservado, antes de voltar para um dos nossos quartos de hotel modernos. Quando estiver pronto para se aventurar novamente, caminhe pelo movimentado Mercado Central de Belo Horizonte para comprar souvenires ou frutas frescas e provar lanches tradicionais brasileiros, como o pão de queijo. Comece seu dia com o pé direito saboreando um café da manhã de cortesia na Pizzaria Olegário e volte mais tarde para um delicioso almoço ou jantar. Viajando a trabalho? Com um centro de negócios, estacionamento amplo e quatro espaços de\"\nPergunta: \"Onde passear no Brasil?\"",
        },
        {
            'role': "assistant",
            'content': "{\"Razão\":\"A passagem responde à pergunta indiretamente incluindo informação desnecessária, indicando que na região do hotel em Belo Horizonte fica um Patrimônio Mundial da UNESCO, que pode ser uma boa sugestão de passeio no Brasil\",\"Pontuação\":1}"
        }
    ], 
]



EVALUATION_FEW_SHOT_EXAMPLES_REV_2_8=[
    [
        {
            'role': "user",
            'content': "Passagem: \"Vanessa Redgrave nasceu em 30 de janeiro de 1937 em Londres. Ela fará falta, mas não será esquecida. Centenas de fãs imediatamente começaram a escrever suas mensagens de condolências na página do Facebook, expressando sua tristeza pela morte da talentosa atriz de 79 anos.\"\nPergunta: \"quantos anos tem vanessa redgrave?\"",
        },
        {
            'role': "assistant",
            'content': "{\"Razão\":\"A passagem é altamente relevante pois responde a idade que Vanessa Redgrave tinha quando faleceu, e que ela nasceu em 1937. Com essas informações é possível determinar que ela morreu em 2016 e ter toda informação completa\",\"Pontuação\":2}"
        }
    ],

    [
        {
            'role': "user",
            'content': "Passagem: \"O Brasil possui muitas belezas naturais. Neste artigo vamos indicar os melhores lugares para passear no Brasil.\"\nPergunta: \"Onde passear no Brasil?\"",
        },
        {
            'role': "assistant",
            'content': "{\"Razão\":\"A passagem é irrelevante para a pergunta, pois apenas indica que o Brasil tem muitas belezas naturais, mas não lista nenhum exemplo específico.\",\"Pontuação\":0}"
        }
    ],

    [
        {
            'role': "user",
            'content': "Passagem: \"conveniente no bairro mais moderno da cidade, o Radisson Blu Belo Horizonte, Savassi é um dos hotéis cinco estrelas mais desejados do Sudeste do Brasil. Passe o dia comprando produtos de marcas famosas no Pátio Savassi ou visitando a Pampulha, um Patrimônio Mundial da UNESCO maravilhosamente preservado, antes de voltar para um dos nossos quartos de hotel modernos. Quando estiver pronto para se aventurar novamente, caminhe pelo movimentado Mercado Central de Belo Horizonte para comprar souvenires ou frutas frescas e provar lanches tradicionais brasileiros, como o pão de queijo. Comece seu dia com o pé direito saboreando um café da manhã de cortesia na Pizzaria Olegário e volte mais tarde para um delicioso almoço ou jantar. Viajando a trabalho? Com um centro de negócios, estacionamento amplo e quatro espaços de\"\nPergunta: \"Onde passear no Brasil?\"",
        },
        {
            'role': "assistant",
            'content': "{\"Razão\":\"A passagem é relevante pois responde indiretamente à pergunta, incluindo muita informação desnecessária, mas indicando que na região do hotel em Belo Horizonte fica um Patrimônio Mundial da UNESCO, que pode ser uma boa sugestão de passeio no Brasil\",\"Pontuação\":1}"
        }
    ], 
]



EVALUATION_FEW_SHOT_EXAMPLES_REV_2_7=[
    [
        {
            'role': "user",
            'content': "Passagem: \"Vanessa Redgrave nasceu em 30 de janeiro de 1937 em Londres. Ela fará falta, mas não será esquecida. Centenas de fãs imediatamente começaram a escrever suas mensagens de condolências na página do Facebook, expressando sua tristeza pela morte da talentosa atriz de 79 anos.\"\nPergunta: \"quantos anos tem vanessa redgrave?\"",
        },
        {
            'role': "assistant",
            'content': "{\"Razão\":\"A passagem responde a idade que Vanessa Redgrave tinha quando faleceu, e que ela nasceu em 1937. Com essas informações é possível determinar que ela morreu em 2016 e ter toda informação completa\",\"Pontuação\":2}"
        }
    ],

    [
        {
            'role': "user",
            'content': "Passagem: \"O Brasil possui muitas belezas naturais. Neste artigo vamos indicar os melhores lugares para passear no Brasil.\"\nPergunta: \"Onde passear no Brasil?\"",
        },
        {
            'role': "assistant",
            'content': "{\"Razão\":\"A passagem apenas indica que o Brasil tem muitas belezas naturais, mas não lista nenhum exemplo. Embora a passagem indique que artigo vai falar sobre lugares para passear no Brasil, o trecho apresentado não lista nenhum lugar específico para passear no Brasil\",\"Pontuação\":0}"
        }
    ],

    [
        {
            'role': "user",
            'content': "Passagem: \"conveniente no bairro mais moderno da cidade, o Radisson Blu Belo Horizonte, Savassi é um dos hotéis cinco estrelas mais desejados do Sudeste do Brasil. Passe o dia comprando produtos de marcas famosas no Pátio Savassi ou visitando a Pampulha, um Patrimônio Mundial da UNESCO maravilhosamente preservado, antes de voltar para um dos nossos quartos de hotel modernos. Quando estiver pronto para se aventurar novamente, caminhe pelo movimentado Mercado Central de Belo Horizonte para comprar souvenires ou frutas frescas e provar lanches tradicionais brasileiros, como o pão de queijo. Comece seu dia com o pé direito saboreando um café da manhã de cortesia na Pizzaria Olegário e volte mais tarde para um delicioso almoço ou jantar. Viajando a trabalho? Com um centro de negócios, estacionamento amplo e quatro espaços de\"\nPergunta: \"Onde passear no Brasil?\"",
        },
        {
            'role': "assistant",
            'content': "{\"Razão\":\"A passagem responde à pergunta indiretamente incluindo informação desnecessária, indicando que na região do hotel em Belo Horizonte fica um Patrimônio Mundial da UNESCO, que pode ser uma boa sugestão de passeio no Brasil\",\"Pontuação\":1}"
        }
    ], 
]



EVALUATION_FEW_SHOT_EXAMPLES_REV_2_2=[
    [
        {
            'role': "user",
            'content': "Passagem: \"O cirurgião faz uma incisão no quadril, remove a articulação do quadril danificada e a substitui por uma articulação artificial que é uma liga metálica ou, em alguns casos, cerâmica. A cirurgia geralmente leva cerca de 60 a 90 minutos para ser concluída.\"\nPergunta: \"de que metal são feitas as próteses de quadril?\"", 
        },
        {  
            'role': "assistant",
            'content': "{\"Razão\":\"Passagem não responde completamente a pergunta, apenas indicando que a prótese pode ser de uma liga metálica, sem listar quais metais.\",\"Pontuação\":1}"
        }
    ],

    [
        {
            'role': "user",
            'content': "Passagem: \"Vanessa Redgrave nasceu em 30 de janeiro de 1937 em Londres. Ela fará falta, mas não será esquecida. Centenas de fãs imediatamente começaram a escrever suas mensagens de condolências na página do Facebook, expressando sua tristeza pela morte da talentosa atriz de 79 anos.\"\nPergunta: \"quantos anos tem vanessa redgrave?\"",
        },
        {
            'role': "assistant",
            'content': "{\"Razão\":\"A passagem responde a idade que Vanessa Redgrave tinha quando faleceu, e que ela nasceu em 1937. Com essas informações é possível determinar que ela morreu em 2016 e ter toda informação completa\",\"Pontuação\":3}"
        }
    ],

    [
        {
            'role': "user",
            'content': "Passagem: \"O Brasil possui muitas belezas naturais. Neste artigo vamos indicar os melhores lugares para passear no Brasil.\"\nPergunta: \"Onde passear no Brasil?\"",
        },
        {
            'role': "assistant",
            'content': "{\"Razão\":\"A passagem apenas indica que o Brasil tem muitas belezas naturais, mas não lista nenhum exemplo. Embora a passagem indique que artigo vai falar sobre lugares para passear no Brasil, o trecho apresentado não lista nenhum lugar específico para passear no Brasil\",\"Pontuação\":0}"
        }
    ],

    [
        {
            'role': "user",
            'content': "Passagem: \"conveniente no bairro mais moderno da cidade, o Radisson Blu Belo Horizonte, Savassi é um dos hotéis cinco estrelas mais desejados do Sudeste do Brasil. Passe o dia comprando produtos de marcas famosas no Pátio Savassi ou visitando a Pampulha, um Patrimônio Mundial da UNESCO maravilhosamente preservado, antes de voltar para um dos nossos quartos de hotel modernos. Quando estiver pronto para se aventurar novamente, caminhe pelo movimentado Mercado Central de Belo Horizonte para comprar souvenires ou frutas frescas e provar lanches tradicionais brasileiros, como o pão de queijo. Comece seu dia com o pé direito saboreando um café da manhã de cortesia na Pizzaria Olegário e volte mais tarde para um delicioso almoço ou jantar. Viajando a trabalho? Com um centro de negócios, estacionamento amplo e quatro espaços de\"\nPergunta: \"Onde passear no Brasil?\"",
        },
        {
            'role': "assistant",
            'content': "{\"Razão\":\"A passagem responde à pergunta indiretamente incluindo informação desnecessária, indicando que na região do hotel em Belo Horizonte fica um Patrimônio Mundial da UNESCO, que pode ser uma boa sugestão de passeio no Brasil\",\"Pontuação\":2}"
        }
    ], 
]



EVALUATION_FEW_SHOT_EXAMPLES_REV_2_1=[
    [
        {
            'role': "user",
            'content': "Passagem: \"O cirurgião faz uma incisão no quadril, remove a articulação do quadril danificada e a substitui por uma articulação artificial que é uma liga metálica ou, em alguns casos, cerâmica. A cirurgia geralmente leva cerca de 60 a 90 minutos para ser concluída.\"\nPergunta: \"de que metal são feitas as próteses de quadril?\"", 
        },
        {  
            'role': "assistant",
            'content': "{\"Razão\":\"Passagem contém informação que não responde a pergunta, apenas indicando que a prótese pode ser de uma liga metálica, sem listar quais metais; de qualquer forma, o assunto está relacionado com a pergunta.\",\"Pontuação\":1}"
        }
    ],

    [
        {
            'role': "user",
            'content': "Passagem: \"Vanessa Redgrave nasceu em 30 de janeiro de 1937 em Londres. Ela fará falta, mas não será esquecida. Centenas de fãs imediatamente começaram a escrever suas mensagens de condolências na página do Facebook, expressando sua tristeza pela morte da talentosa atriz de 79 anos.\"\nPergunta: \"quantos anos tem vanessa redgrave?\"",
        },
        {
            'role': "assistant",
            'content': "{\"Razão\":\"A passagem responde a idade que Vanessa Redgrave tinha quando faleceu, e que ela nasceu em 1937. Com essas informações é possível determinar que ela morreu em 2016 e ter toda informação completa\",\"Pontuação\":3}"
        }
    ],

    [
        {
            'role': "user",
            'content': "Passagem: \"O Brasil possui muitas belezas naturais. Neste artigo vamos indicar os melhores lugares para passear no Brasil.\"\nPergunta: \"Onde passear no Brasil?\"",
        },
        {
            'role': "assistant",
            'content': "{\"Razão\":\"A passagem apenas indica que o Brasil tem muitas belezas naturais, mas não lista nenhum exemplo. Embora a passagem indique que artigo vai falar sobre lugares para passear no Brasil, tema da pergunta, o trecho apresentado não lista nenhum lugar específico para passear no Brasil\",\"Pontuação\":0}"
        }
    ],

    [
        {
            'role': "user",
            'content': "Passagem: \"conveniente no bairro mais moderno da cidade, o Radisson Blu Belo Horizonte, Savassi é um dos hotéis cinco estrelas mais desejados do Sudeste do Brasil. Passe o dia comprando produtos de marcas famosas no Pátio Savassi ou visitando a Pampulha, um Patrimônio Mundial da UNESCO maravilhosamente preservado, antes de voltar para um dos nossos quartos de hotel modernos. Quando estiver pronto para se aventurar novamente, caminhe pelo movimentado Mercado Central de Belo Horizonte para comprar souvenires ou frutas frescas e provar lanches tradicionais brasileiros, como o pão de queijo. Comece seu dia com o pé direito saboreando um café da manhã de cortesia na Pizzaria Olegário e volte mais tarde para um delicioso almoço ou jantar. Viajando a trabalho? Com um centro de negócios, estacionamento amplo e quatro espaços de\"\nPergunta: \"Onde passear no Brasil?\"",
        },
        {
            'role': "assistant",
            'content': "{\"Razão\":\"A passagem responde à pergunta indiretamente incluindo informação desnecessária, indicando que na região do hotel em Belo Horizonte fica um Patrimônio Mundial da UNESCO, que pode ser uma boa sugestão de passeio no Brasil\",\"Pontuação\":2}"
        }
    ], 
]


#
# Define which version to use
#

EVALUATION_FEW_SHOT_EXAMPLES=None
EVALUATION_SYSTEM_ROLE=EVALUATION_SYSTEM_ROLE_REV_2_6




EVALUATION_FEW_SHOT_EXAMPLES_FIRST_ANALYSIS=[
    [
        {
            'role': "user",
            'content': "Passagem: \"O cirurgião faz uma incisão no quadril, remove a articulação do quadril danificada e a substitui por uma articulação artificial que é uma liga metálica ou, em alguns casos, cerâmica. A cirurgia geralmente leva cerca de 60 a 90 minutos para ser concluída.\"\nPergunta: \"de que metal são feitas as próteses de quadril?\"", 
        },
        {  
            'role': "assistant",
            'content': "A passagem não responde a pergunta de forma clara, pois apenas indica indiretamente que a prótese pode ser de uma liga metálica, mas não explicita quais metais; de qualquer forma, o assunto da passagem é relacionado com a pergunta. Pontuação: 1."
        }
    ],

    [
        {
            'role': "user",
            'content': "Passagem: \"Vanessa Redgrave nasceu em 30 de janeiro de 1937 em Londres. Ela fará falta, mas não será esquecida. Centenas de fãs imediatamente começaram a escrever suas mensagens de condolências na página do Facebook, expressando sua tristeza pela morte da talentosa atriz de 79 anos.\"\nPergunta: \"quantos anos tem vanessa redgrave?\"",
        },
        {
            'role': "assistant",
            'content': "A passagem responde a idade que Vanessa Redgrave tinha quando faleceu, e que ela nasceu em 1937. Com essas informações é possível determinar que ela morreu em 2016 e ter toda informação completa. Pontuação: 3."
        }
    ],

    [
        {
            'role': "user",
            'content': "Passagem: \"O Brasil possui muitas belezas naturais. Neste artigo vamos indicar os melhores lugares para passear no Brasil.\"\nPergunta: \"Onde passear no Brasil?\"",
        },
        {
            'role': "assistant",
            'content': "A passagem apenas indica que o Brasil tem muitas belezas naturais, mas não indica nenhum exemplo. Embora a passagem indique que artigo vai falar sobre lugares para passear no Brasil, tema da pergunta, o trecho apresentado não lista nenhum lugar específico para passear no Brasil. Pontuação: 0."
        }
    ],

    [
        {
            'role': "user",
            'content': "Passagem: \"conveniente no bairro mais moderno da cidade, o Radisson Blu Belo Horizonte, Savassi é um dos hotéis cinco estrelas mais desejados do Sudeste do Brasil. Passe o dia comprando produtos de marcas famosas no Pátio Savassi ou visitando a Pampulha, um Patrimônio Mundial da UNESCO maravilhosamente preservado, antes de voltar para um dos nossos quartos de hotel modernos. Quando estiver pronto para se aventurar novamente, caminhe pelo movimentado Mercado Central de Belo Horizonte para comprar souvenires ou frutas frescas e provar lanches tradicionais brasileiros, como o pão de queijo. Comece seu dia com o pé direito saboreando um café da manhã de cortesia na Pizzaria Olegário e volte mais tarde para um delicioso almoço ou jantar. Viajando a trabalho? Com um centro de negócios, estacionamento amplo e quatro espaços de\"\nPergunta: \"Onde passear no Brasil?\"",
        },
        {
            'role': "assistant",
            'content': "A passagem responde à pergunta indiretamente, indicando que na região do hotel em Belo Horizonte fica um Patrimônio Mundial da UNESCO, que pode ser uma boa sugestão de passeio no Brasil. Pontuação: 2."
        }
    ], 
]

EVALUATION_ADDITIONAL_FEW_SHOT_EXAMPLES=[
    [
        {
            'role': "user",
            'content': "Passagem: \"Uma prótese feita de metal e plástico são os implantes de substituição do quadril mais comumente usados. Tanto a bola quanto o soquete da articulação do quadril são substituídos por um implante de metal e um espaçador de plástico é colocado entre eles. Os metais mais comumente usados incluem titânio e aço inoxidável.\"\nPergunta: \"de que metal são feitas as próteses de quadril?\"",
        },
        {
            'role': "assistant",
            'content': "Pontuação: 10; Razão: a passagem responde à pergunta de forma clara e direta, e ainda acrescenta informações relevantes sobre próteses de quadril."
        }
    ],

    [
        {
            'role': "user",
            'content': "Passagem: \"conveniente no bairro mais moderno da cidade, o Radisson Blu Belo Horizonte, Savassi é um dos hotéis cinco estrelas mais desejados do Sudeste do Brasil. Passe o dia comprando produtos de marcas famosas no Pátio Savassi ou visitando a Pampulha, um Patrimônio Mundial da UNESCO maravilhosamente preservado, antes de voltar para um dos nossos quartos de hotel modernos. Quando estiver pronto para se aventurar novamente, caminhe pelo movimentado Mercado Central de Belo Horizonte para comprar souvenires ou frutas frescas e provar lanches tradicionais brasileiros, como o pão de queijo. Comece seu dia com o pé direito saboreando um café da manhã de cortesia na Pizzaria Olegário e volte mais tarde para um delicioso almoço ou jantar. Viajando a trabalho? Com um centro de negócios, estacionamento amplo e quatro espaços de\"\nPergunta: \"Onde passear no Brasil?\"",
        },
        {
            'role': "assistant",
            'content': "Pontuação: 5; Razão: a passagem responde à pergunta muito indiretamente, indicando que na região do hotel em Belo Horizonte fica um Patrimônio Mundial da UNESCO, que pode ser uma boa sugestão de passeio no Brasil."
        }
    ],
    
    [
        {
            'role': "user",
            'content': "Passagem: \"ideal para a sua viagem à bela Cidade Jardim do Brasil. Quando suas reuniões do dia terminarem, você pode caminhar até o restaurante Dona Lucinha para saborear a culinária local e, depois, visitar as lojas no Pátio Savassi. Se preferir passar algum tempo admirando a paisagem exuberante da área, há praças públicas bem cuidadas, como a Praça da Liberdade e a Praça da Savassi a uma curta distância a pé. Parque Municipal Américo Renné Giannetti 1,23 mi / 1,97 km do hotel Deixe-se envolver pela natureza neste belo parque no centro da cidade de Belo Horizonte. Desfrute de um piquenique na grama, alugue um barco a remo para passear no lago ou observe seus filhos gastarem sua energia no playground. Praça da Savassi 0,23 mi / 0,3\"\nPergunta: \"Onde passear no Brasil?\"",
        },
        {
            'role': "assistant",
            'content': "Pontuação: 1; Razão: a passagem não responde à pergunta, indicando detalhes de uma região de Belo Horizonte que dificilmente vão interessar alguém buscando informações gerais sobre passeios no Brasil."
        }
    ]    
]

EVALUATION_FEW_SHOT_EXAMPLE_FORMAT="Exemplo {}:\n{}"

EVALUATION_SINGLE_FEW_SHOT_PROMPT_INITIAL_FORMAT="{} Siga os exemplos abaixo."

EVALUATION_QUERY_PASSAGE_FORMAT="Passagem: \"{}\"\nPergunta: \"{}\""
EVALUATION_OPENAI_RESPONSE_REGEX="[\n\r]*\{\s*[\"\']Razão[\"\']\s*:\s*\"(.+)\"\s*,\s*[\"\']Pontuação[\"\']\s*:\s*([0-9\.]+)\s*\}[\.\n\r]*"

EVALUATION_MAX_TOKENS_RESPONSE=500

#
# Queries LLM creation definitions
#

CREATION_SYSTEM_ROLE = {
    'role': "system", 
    'content': "Você sugere 2 perguntas a partir da leitura de uma passagem de texto. A primeira pergunta explora o tema da passagem, e a segunda pergunta explora uma informação ou conclusão específica possível a partir da leitura da passagem. Suas perguntas devem fazer sentido para alguém que não leu a passagem. Siga o formato do exemplo."
}

CREATION_ONE_SHOT_EXAMPLE=[
    {
        'role': "user",
        'content': "Exemplo: Passagem: \"Como acontece com todos os tratamentos naturais, a qualidade do produto utilizado para o tratamento de decide o resultado. Portanto, se você deseja obter os melhores resultados com o tratamento óleo de rosa mosqueta para a acne, você deve tentar encontrar o melhor e mais puro óleo de rosa mosqueta orgânica. Antes de comprar um produto, certifique-se que você leia os rótulos das embalagens adequadamente para verificar se ele contém óleo de rosa mosqueta puro ou de uma mistura de outros óleos essenciais. Leia as instruções de uso recomendadas pelo fabricante, porque alguns produtos requerem lavagem após alguns minutos da aplicação, enquanto que alguns precisam ser mantidos durante a noite.óleo de rosa mosqueta tem um cheiro desagradável e desagradável e muitas pessoas podem não gostar. Se você tem crianças em casa, eles podem ser desligados de você devido ao cheiro. Por isso, certifique-se de que você adicionar uma certa quantidade de óleo essencial aromático, tal como lavanda ou jasmim para travar para baixo o cheiro.[ Ler: Como usar o óleo de abacate para acne? ]Considerações ao usar o Óleo de Rosa Mosqueta\"", 
    },
    {  
        'role': "assistant",
        'content': "Pergunta 1: Quais tratamentos para acne?\nPergunta 2: Como é possível evitar o forte cheiro da rosa mosqueta no tratamento de pele?"
    }
]

CREATION_PASSAGE_FORMAT="Passagem: \"{}\""
CREATION_OPENAI_RESPONSE_REGEX="[\n\r]*[Pp]ergunta 1:\s*(.+)\s*[\n\r]+[Pp]ergunta 2:\s*(.+)[\n\r]*"

CREATION_MAX_TOKENS_RESPONSE=200

LLM_EVALUATION_KEY_FORMAT = "{}_{}"



def initialize_openai(which_key="OPENAI_API_KEY"):

    if which_key == "OPENAI_API_KEY_2":
        api_keys_filename = API_KEYS_FILE_2
    else:
        api_keys_filename = API_KEYS_FILE

    with open(api_keys_filename) as inputFile:
        api_keys = json.load(inputFile)

    openai.api_key = api_keys[which_key]



def compute_openai_api_usage_cost(api_usage_dict, which_model):

    cost =  api_usage_dict['prompt_tokens'] / 1000 * API_PRICING_INPUT[which_model] + \
            api_usage_dict['completion_tokens'] / 1000 * API_PRICING_OUTPUT[which_model]
    
    return cost



def execute_LLM_passage_relevance_evaluation(which_query, 
                                             which_passage, 
                                             model=MODEL_GPT3, 
                                             number_of_completions=1,
                                             add_examples=True,
                                             verbose=True):

    start_time = time.time()

    query_passage_to_evaluate=EVALUATION_QUERY_PASSAGE_FORMAT.format(which_passage, which_query)

    if verbose:
        print("++++++++++++++++++++++++++")
        print(query_passage_to_evaluate)
        print("++++++++++++++++++++++++++")


    if number_of_completions > 1:
        temperature = 1
    else:
        temperature = 0

    if model == MODEL_GPT3:
        messages_to_send = [EVALUATION_SYSTEM_ROLE]

        for i, example in enumerate(EVALUATION_FEW_SHOT_EXAMPLES):
            for example_role in example:
                if example_role['role'] == "user":
                    messages_to_send.append({'role': "user", 
                                             'content': EVALUATION_FEW_SHOT_EXAMPLE_FORMAT.format(i + 1, example_role['content'])})
                else:
                    messages_to_send.append(example_role)

        messages_to_send.append({'role': "user", 'content': query_passage_to_evaluate})

    
        if verbose:
            print("\n")
            print(messages_to_send)
        
        response = openai.ChatCompletion.create(model=model,
                                                messages=messages_to_send,
                                                temperature=temperature,
                                                n=number_of_completions,
                                                max_tokens=EVALUATION_MAX_TOKENS_RESPONSE)      

    elif model == MODEL_GPT4:
        messages_to_send = [EVALUATION_SYSTEM_ROLE]

        if add_examples:
            for i, example in enumerate([EVALUATION_FEW_SHOT_EXAMPLES[0], EVALUATION_FEW_SHOT_EXAMPLES[2]]):
                for example_role in example:
                    if example_role['role'] == "user":
                        messages_to_send.append({'role': "user", 
                                                'content': EVALUATION_FEW_SHOT_EXAMPLE_FORMAT.format(i + 1, example_role['content'])})
                    else:
                        messages_to_send.append(example_role)

        messages_to_send.append({'role': "user", 'content': query_passage_to_evaluate})

    
        if verbose:
            print("\n")
            print(messages_to_send)
        
        response = openai.ChatCompletion.create(model=model,
                                                messages=messages_to_send,
                                                temperature=temperature,
                                                n=number_of_completions,
                                                max_tokens=EVALUATION_MAX_TOKENS_RESPONSE)

    elif model == MODEL_DAVINCI3:
        prompt_to_send = EVALUATION_SINGLE_FEW_SHOT_PROMPT_INITIAL_FORMAT.format(EVALUATION_SYSTEM_ROLE['content'])

        for i, example in enumerate([EVALUATION_FEW_SHOT_EXAMPLES[0], EVALUATION_FEW_SHOT_EXAMPLES[1] ,EVALUATION_FEW_SHOT_EXAMPLES[2]]):
            for example_role in example:
                if example_role['role'] == "user":
                    prompt_to_send += "\n\n" + EVALUATION_FEW_SHOT_EXAMPLE_FORMAT.format(i + 1, example_role['content'])
                else:
                    prompt_to_send += "\n\n" + example_role['content']

        prompt_to_send += "\n\n" + query_passage_to_evaluate

        if verbose:
            print("\n")
            print(prompt_to_send)

        response = openai.Completion.create(model=model,
                                            prompt=prompt_to_send,
                                            temperature=temperature,
                                            n=number_of_completions,
                                            max_tokens=EVALUATION_MAX_TOKENS_RESPONSE,
                                            top_p=1,
                                            frequency_penalty=0,
                                            presence_penalty=0)  
        
        

    else:
        raise ValueError("Cannot handle OPENAI model {}...".format(model))


    if verbose:
        print("\n")
        print(response['choices'])

    LLM_responses = []

    for i in range(number_of_completions):
        if model == MODEL_DAVINCI3:
            response_text = response['choices'][i]['text']
        else:
            response_text = response['choices'][i]['message']['content']

        m = re.match(EVALUATION_OPENAI_RESPONSE_REGEX, response_text)

        if len(m.groups()) == 2:
            score = int(round(float(m.group(2))))
            reasoning = m.group(1)
        else:
            score = None
            reasoning = None

        LLM_responses.append({'score': score,
                              'reasoning': reasoning})

    final_time = time.time()
    final_cost = compute_openai_api_usage_cost(response['usage'], model)

    print("\nLLM query relevance evaluation duration: {}; cost: {}; usage{}\n\n".format(final_time - start_time, final_cost, response['usage']))

    if number_of_completions > 1:
        final_result = {'LLM_responses': LLM_responses,
                        'usage': response['usage'].copy(),
                        'cost': final_cost,
                        'duration': final_time - start_time}
    else:
        final_result = {'score': LLM_responses[0]['score'],
                        'reasoning': LLM_responses[0]['reasoning'],
                        'usage': response['usage'].copy(),
                        'cost': final_cost,
                        'duration': final_time - start_time}

    return final_result



def execute_LLM_query_creation(which_passage, 
                               model=MODEL_GPT3, 
                               verbose=True):

    start_time = time.time()

    passage_to_create_question=CREATION_PASSAGE_FORMAT.format(which_passage)

    if verbose:
        print("++++++++++++++++++++++++++")
        print(passage_to_create_question)
        print("++++++++++++++++++++++++++")

    if model == MODEL_GPT3:
        messages_to_send = [CREATION_SYSTEM_ROLE]

        for example_role in CREATION_ONE_SHOT_EXAMPLE:
            if example_role['role'] == "user":
                messages_to_send.append({'role': "user", 
                                         'content': example_role['content']})
            else:
                messages_to_send.append(example_role)

        messages_to_send.append({'role': "user", 'content': passage_to_create_question})

    
        if verbose:
            print("\n")
            print(messages_to_send)
        
        response = openai.ChatCompletion.create(model=model,
                                                messages=messages_to_send,
                                                temperature=0,
                                                max_tokens=CREATION_MAX_TOKENS_RESPONSE)
        
        response_text = response['choices'][0]['message']['content']

    else:
        raise ValueError("Cannot handle OPENAI model {}...".format(model))


    if verbose:
        print("\n")
        print(response_text)

    m = re.match(CREATION_OPENAI_RESPONSE_REGEX, response_text)

    if (m is not None) and (len(m.groups()) == 2):
        question_theme = m.group(1)
        question_specific = m.group(2)
    else:
        question_theme = None
        question_specific = None

    final_time = time.time()

    final_cost = compute_openai_api_usage_cost(response['usage'], model)

    print("\nLLM passage queries creation duration: {}; cost: {}\n\n".format(final_time - start_time, final_cost))

    return {'question_theme': question_theme,
            'question_specific': question_specific,
            'usage': response['usage'].copy(),
            'cost': final_cost,
            'duration': final_time - start_time}



def extract_query_passages_from_doccano(which_file):
    annotations_df = pd.read_csv(which_file)
    
    query_passages = []

    for i, row in annotations_df.iterrows():
        m = re.match("Query: \n(.+)\n\nPassage:\n(.+)", row['text'])

        # print("query={}".format(m.group(1)))
        # print("passage={}".format(m.group(2)))

        query_passages.append({
                               'doccano_id': row['id'],
                               'query': m.group(1),
                               'passage': m.group(2),
                               'passage_id': row['passage-id']
                              })
    
    return annotations_df, pd.DataFrame(query_passages)



def LLM_query_passage_evaluation(query_passages_df, 
                                 which_model, 
                                 output_file, 
                                 LLM_evaluations=None,
                                 number_of_completions=1,
                                 add_examples=True):

    if LLM_evaluations is None:
        LLM_evaluations = {}
    
    for i, row in query_passages_df.iterrows():
        
        document_key = LLM_EVALUATION_KEY_FORMAT.format(row['query'], row['passage_id'])

        print("Query/Passage evaluation {}; document_key={}...".format(i, document_key))

        if document_key not in LLM_evaluations:
            try:
                relevance_results = execute_LLM_passage_relevance_evaluation(row['query'], 
                                                                             row['passage'], 
                                                                             model=which_model, 
                                                                             number_of_completions=number_of_completions,
                                                                             add_examples=add_examples,
                                                                             verbose=True)
            except Exception as e:
                print(e)

                if number_of_completions > 1:
                    time.sleep(60)
                else:
                    time.sleep(30)

                relevance_results = execute_LLM_passage_relevance_evaluation(row['query'], 
                                                                             row['passage'], 
                                                                             model=which_model, 
                                                                             number_of_completions=number_of_completions,
                                                                             add_examples=add_examples,
                                                                             verbose=True)
        else:
            print("-- LLM already evaluated document {}...\n".format(document_key))

            relevance_results = LLM_evaluations[document_key]

        if (number_of_completions > 1) and ('score' not in relevance_results):
            
            rounded_score = 0
            
            for LLM_response in relevance_results['LLM_responses']:
                rounded_score += LLM_response['score']

            relevance_results['score'] = int(round(rounded_score / number_of_completions))

            print(">> Rounded score: {}\n\n\n".format(relevance_results['score']))

        LLM_evaluations[document_key] = relevance_results
        
        validation_results_df = pd.concat([query_passages_df.reset_index(drop=True), 
                                        pd.DataFrame.from_dict(LLM_evaluations, orient='index').reset_index(drop=True)], axis=1)
        
        validation_results_df.to_csv(output_file, sep='\t', index=False)
    
    return validation_results_df



def check_agreement_per_questions(evaluation_a, evaluation_b, prefix=None, correlation_function="cohen_kappa_score"):
    merged_df = evaluation_a.merge(evaluation_b, left_on='doccano_id', right_on='doccano_id')[['query_x', 'passage_x', 'passage_id_x', 'score_x', 'score_y']]
    
    correlations = []
    
    for group_name, group_df in merged_df.groupby('query_x'):

        correlation_result = np.array(globals()[correlation_function](group_df['score_x'], group_df['score_y']))

        if len(correlation_result.shape) > 1:
            correlations.append({'query': group_name,
                                 '{}{}'.format(correlation_function, prefix): correlation_result[:, 0],
                                 '{}{}_pvalue'.format(correlation_function, prefix): correlation_result[:, 1]})
        else:
            correlations.append({'query': group_name,
                                '{}{}'.format(correlation_function, prefix): correlation_result})
        
    return merged_df, pd.DataFrame(correlations).fillna(1.0)