import numpy as np

class Perceptron:
	
	def __init__(self,taxa_aprendizado = 1e-2, max_epochs = 10000, limiar_ativacao = 0):
		
		self.taxa_apendizado = taxa_aprendizado
		self.max_epochs = max_epochs

		self.qt_cols = 0
		self.qt_rows = 0
		self.limiar_ativacao = 0
		self.pesos = np.array([])
		self.limiar_ativacao = limiar_ativacao
		self.histpeso = {}

	
	def FuncaoAtivacao(self, sum: float)-> int:
		'''Função de ativação degrau bipolar'''

		return 1 if sum > 0 else -1


	def fit(self,X,label) -> bool:
		'''Treinando a rede neural'''

		self.qt_cols = X.shape[1]
		self.qt_rows = X.shape[0]
		total = X.size

		# Iniciando o vetor de pesos aleatóriamente
		self.pesos = np.random.rand(self.qt_cols)

		# Iniciando o vetor de limiares
		limiar = np.array([self.limiar_ativacao])
		
		# Adicionando o limiar no 1 elemento do vetor de pesos
		self.pesos = np.hstack([limiar,self.pesos])

		# Adicionando o valor -1 no inicio de todos os registros
		X = np.hstack([-1*np.ones(self.qt_rows).reshape(-1,1),X])

		#Iniciando contador de epoca
		epoch = 0

		# Iniciando o erro como True, mas na verdade ele ainda inexiste
		erro = True

		# Iniciando o processo de treinamento
		while(erro and epoch < self.max_epochs):
			print(f"Peso Epoch {epoch}: {self.pesos}")
			self.histpeso[f"{epoch}"] = self.pesos	

			# Erro, a príncipio, inexiste	
			erro = False 
			#Ponderando a entrada com os pesos
			for i in range(self.qt_rows):
				u = (X[i]*self.pesos).sum()

				# Enviando para  a função de ativação
				y = self.FuncaoAtivacao(u)

				#Comparando o resultado gerado pelo neuronio com o original
				if(y != label[i]):
					
					#Erro ainda existe
					erro = True
					
					# Atualizando os pesos conforme a regra da Hebb
					self.pesos =self.pesos + (self.taxa_apendizado*(label[i] - y) * X[i])


			#Incrementando a época
			epoch += 1




				

	def predict(self,X):
		'''Realizando predições com a rede neural'''

		X = np.hstack([-1*np.ones(X.shape[0]).reshape(-1,1),X])
		
		resultado = []
		for i in range(X.shape[0]):
			sum = (X[i]*self.pesos).sum()
			# Enviando para  a função de ativação
			y = self.FuncaoAtivacao(sum)
			resultado.append(y)
		
		return np.array(resultado)
			

	
