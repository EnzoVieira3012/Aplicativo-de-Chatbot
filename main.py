# noinspection PyUnresolvedReferences
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download dos recursos necessários
nltk.download('punkt')


class PythonChatbot:
    def __init__(self):
        # Dados de treinamento (exemplos de perguntas e respostas)
        self.training_data = [
            ("O que é Python?",
             "Python é uma linguagem de programação de alto nível, interpretada e orientada a objetos."),
            ("Como instalo o Python?",
             "Você pode baixar o Python do site oficial python.org e seguir as instruções de instalação."),
            ("Qual é a última versão do Python?",
             "Você pode verificar a última versão do Python no site oficial python.org."),
            ("Como eu escrevo uma função em Python?",
             "Para escrever uma função em Python, use a palavra-chave 'def', seguida pelo nome da função e parênteses. Por exemplo: def minha_funcao():"),
            ("O que é uma lista em Python?",
             "Uma lista em Python é uma coleção ordenada e mutável de itens. Você pode criar uma lista usando colchetes [] e separar os itens por vírgulas.")
        ]

        # Preparação dos dados
        self.questions, self.answers = zip(*self.training_data)
        self.vectorizer = CountVectorizer()
        self.X_train = self.vectorizer.fit_transform(self.questions)
        self.model = MultinomialNB()
        self.model.fit(self.X_train, self.answers)

    def get_response(self, user_input):
        # Verifique se o modelo está treinado
        if not hasattr(self, 'model') or not hasattr(self, 'vectorizer'):
            raise RuntimeError("O modelo não foi treinado.")

        # Transforme o input do usuário para a matriz de características
        X_test = self.vectorizer.transform([user_input])

        # Verifique se a transformação foi feita corretamente
        if X_test.shape[1] != self.X_train.shape[1]:
            raise ValueError("A dimensão do teste não corresponde à dimensão do treinamento.")

        # Obtenha a previsão
        predicted = self.model.predict(X_test)
        return predicted[0]


def main():
    chatbot = PythonChatbot()
    print("Olá! Sou um chatbot sobre programação em Python. Pergunte-me algo sobre Python.")

    while True:
        user_input = input("\nVocê: ")
        if user_input.lower() in ['sair', 'exit', 'quit']:
            print("Chatbot: Até logo!")
            break

        response = chatbot.get_response(user_input)
        print(f"Chatbot: {response}")


if __name__ == "__main__":
    main()
